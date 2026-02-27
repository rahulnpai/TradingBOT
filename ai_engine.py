"""
ai_engine.py — Ollama LLM integration for AI-assisted trade decisions.

Responsibilities:
  • Analyse indicator snapshot and validate / enrich a trade signal.
  • Explain trade rationale in plain English.
  • Periodically suggest strategy optimisations.

Design goals:
  • Only called at decision points (signal generated), NOT every tick.
  • Per-symbol cooldown prevents spamming the model.
  • Gracefully degrades: if Ollama is unreachable, trading continues
    with the raw technical signal (ai_decision = "PASS_THROUGH").
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

import requests

from config import config
from logger import get_logger
from strategy import Signal

log = get_logger("AIEngine")


# ---------------------------------------------------------------------------
# Response dataclass
# ---------------------------------------------------------------------------

@dataclass
class AIAnalysis:
    decision: str            # "CONFIRM" | "REJECT" | "MODIFY" | "PASS_THROUGH"
    confidence: float        # 0.0 – 1.0  (AI's own confidence)
    reasoning: str
    suggested_sl: Optional[float] = None
    suggested_target: Optional[float] = None
    key_risks: List[str]    = field(default_factory=list)
    optimization_notes: str = ""
    raw_response: str       = ""
    latency_ms: int         = 0

    @property
    def is_confirmed(self) -> bool:
        return self.decision in ("CONFIRM", "PASS_THROUGH")

    @property
    def is_rejected(self) -> bool:
        return self.decision == "REJECT"


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_TRADE_PROMPT = """You are an expert quantitative trader specialising in Indian equities (NSE).
Analyse the following intraday signal and respond ONLY with a valid JSON object.

---
SYMBOL: {symbol}
EXCHANGE: NSE
SIGNAL: {action}
STRATEGY: {strategy_type}
CURRENT PRICE: ₹{entry_price:.2f}
PROPOSED STOPLOSS: ₹{sl:.2f}  ({sl_pct:.1f}% from entry)
PROPOSED TARGET:   ₹{tgt:.2f} ({tgt_pct:.1f}% from entry)
RISK:REWARD RATIO: 1:{rr:.2f}

TECHNICAL INDICATORS (latest candle):
  RSI(14)       : {rsi:.2f}  {"(OVERSOLD)" if {rsi:.2f} < 30 else "(OVERBOUGHT)" if {rsi:.2f} > 70 else ""}
  MACD          : {macd:.4f}
  MACD Signal   : {macd_signal:.4f}
  MACD Hist     : {macd_hist:.4f}
  EMA9 / EMA21  : {ema_9:.2f} / {ema_21:.2f}
  EMA50 / EMA200: {ema_50:.2f} / {ema_200:.2f}
  VWAP          : {vwap:.2f}
  Price vs VWAP : {price_vs_vwap}
  Volume Ratio  : {vol_ratio:.2f}x average

SIGNAL REASON: {reason}
SIGNAL CONFIDENCE (technical): {tech_confidence:.2f}
---

Respond with exactly this JSON structure (no markdown, no extra text):
{{
  "decision": "CONFIRM" | "REJECT" | "MODIFY",
  "confidence": <0.0 to 1.0>,
  "reasoning": "<2-3 sentence explanation>",
  "suggested_sl": <price or null>,
  "suggested_target": <price or null>,
  "key_risks": ["<risk1>", "<risk2>"],
  "optimization": "<one actionable suggestion to improve this strategy>"
}}
"""

_OPTIMIZATION_PROMPT = """You are a quantitative strategy advisor for Indian stock markets.
Review this recent trading performance and suggest 3 concrete improvements.

Performance snapshot:
{performance_json}

Recent trades:
{trades_json}

Respond with a JSON object:
{{
  "suggestions": [
    {{"title": "...", "detail": "...", "expected_impact": "..."}},
    {{"title": "...", "detail": "...", "expected_impact": "..."}},
    {{"title": "...", "detail": "...", "expected_impact": "..."}}
  ],
  "overall_assessment": "<brief paragraph>"
}}
"""


# ---------------------------------------------------------------------------
# AI Engine
# ---------------------------------------------------------------------------

class AIEngine:

    def __init__(self) -> None:
        self._cfg = config.ai
        self._last_call: Dict[str, float] = {}    # symbol → timestamp
        self._session = requests.Session()
        self._session.headers.update({"Content-Type": "application/json"})

    # ------------------------------------------------------------------
    # Primary: analyse a trade signal
    # ------------------------------------------------------------------

    def analyse_signal(self, signal: Signal) -> AIAnalysis:
        if not self._cfg.enabled:
            return AIAnalysis(decision="PASS_THROUGH", confidence=signal.confidence,
                              reasoning="AI disabled in config.")

        # Throttle: respect per-symbol cooldown
        last = self._last_call.get(signal.symbol, 0.0)
        elapsed = time.time() - last
        if elapsed < self._cfg.call_interval_sec:
            remaining = int(self._cfg.call_interval_sec - elapsed)
            log.debug("AI throttled for %s (%ds remaining)", signal.symbol, remaining)
            return AIAnalysis(
                decision="PASS_THROUGH", confidence=signal.confidence,
                reasoning=f"AI cooldown active ({remaining}s remaining).",
            )

        prompt = self._build_trade_prompt(signal)
        t0 = time.time()
        raw = self._call_ollama(prompt)
        latency = int((time.time() - t0) * 1000)

        self._last_call[signal.symbol] = time.time()

        if raw is None:
            return AIAnalysis(
                decision="PASS_THROUGH", confidence=signal.confidence,
                reasoning="Ollama unreachable — using technical signal only.",
                latency_ms=latency,
            )

        analysis = self._parse_trade_response(raw, signal, latency)
        log.info(
            "AI [%s] %s %s → %s (conf=%.2f, %.0fms) | %s",
            self._cfg.model, signal.action, signal.symbol,
            analysis.decision, analysis.confidence, latency,
            analysis.reasoning[:80],
        )
        return analysis

    # ------------------------------------------------------------------
    # Periodic optimisation suggestions
    # ------------------------------------------------------------------

    def suggest_optimisations(
        self,
        performance: dict,
        recent_trades: list,
    ) -> Optional[dict]:
        if not self._cfg.enabled:
            return None

        prompt = _OPTIMIZATION_PROMPT.format(
            performance_json=json.dumps(performance, indent=2),
            trades_json=json.dumps(recent_trades[-20:], indent=2),
        )
        raw = self._call_ollama(prompt)
        if raw is None:
            return None
        return self._parse_json_safely(raw)

    # ------------------------------------------------------------------
    # Ollama HTTP call
    # ------------------------------------------------------------------

    def _call_ollama(self, prompt: str) -> Optional[str]:
        url = f"{self._cfg.ollama_url}/api/generate"
        payload = {
            "model":  self._cfg.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.2,
                "top_p": 0.9,
                "num_predict": 512,
            },
        }
        try:
            resp = self._session.post(url, json=payload, timeout=self._cfg.timeout)
            resp.raise_for_status()
            data = resp.json()
            return data.get("response", "")
        except requests.exceptions.ConnectionError:
            log.warning("Ollama not running at %s", self._cfg.ollama_url)
        except requests.exceptions.Timeout:
            log.warning("Ollama request timed out after %ds", self._cfg.timeout)
        except Exception as e:
            log.error("Ollama call failed: %s", e)
        return None

    # ------------------------------------------------------------------
    # Prompt builder
    # ------------------------------------------------------------------

    def _build_trade_prompt(self, signal: Signal) -> str:
        ind = signal.indicators
        entry = signal.entry_price
        sl    = signal.suggested_sl
        tgt   = signal.suggested_target

        if entry > 0 and sl > 0 and tgt > 0:
            sl_pct  = abs(entry - sl)  / entry * 100
            tgt_pct = abs(tgt - entry) / entry * 100
            rr      = tgt_pct / sl_pct if sl_pct > 0 else 0
        else:
            sl_pct = tgt_pct = rr = 0

        vwap_val = ind.get("vwap", 0)
        price_vs_vwap = "ABOVE VWAP" if entry > vwap_val else "BELOW VWAP"

        # Build the prompt by substituting values directly (no .format() to avoid
        # issues with nested braces in the template)
        prompt = (
            "You are an expert quantitative trader specialising in Indian equities (NSE).\n"
            "Analyse the following intraday signal and respond ONLY with a valid JSON object.\n\n"
            "---\n"
            f"SYMBOL: {signal.symbol}\n"
            f"EXCHANGE: NSE\n"
            f"SIGNAL: {signal.action}\n"
            f"STRATEGY: {signal.strategy_type}\n"
            f"CURRENT PRICE: ₹{entry:.2f}\n"
            f"PROPOSED STOPLOSS: ₹{sl:.2f}  ({sl_pct:.1f}% from entry)\n"
            f"PROPOSED TARGET:   ₹{tgt:.2f} ({tgt_pct:.1f}% from entry)\n"
            f"RISK:REWARD RATIO: 1:{rr:.2f}\n\n"
            "TECHNICAL INDICATORS (latest candle):\n"
            f"  RSI(14)       : {ind.get('rsi', 0):.2f}\n"
            f"  MACD          : {ind.get('macd', 0):.4f}\n"
            f"  MACD Signal   : {ind.get('macd_signal', 0):.4f}\n"
            f"  MACD Hist     : {ind.get('macd_hist', 0):.4f}\n"
            f"  EMA9 / EMA21  : {ind.get('ema_9', 0):.2f} / {ind.get('ema_21', 0):.2f}\n"
            f"  EMA50 / EMA200: {ind.get('ema_50', 0):.2f} / {ind.get('ema_200', 0):.2f}\n"
            f"  VWAP          : {vwap_val:.2f}\n"
            f"  Price vs VWAP : {price_vs_vwap}\n"
            f"  Volume Ratio  : {ind.get('vol_ratio', 1):.2f}x average\n\n"
            f"SIGNAL REASON: {signal.reason}\n"
            f"SIGNAL CONFIDENCE (technical): {signal.confidence:.2f}\n"
            "---\n\n"
            'Respond with exactly this JSON structure (no markdown, no extra text):\n'
            '{\n'
            '  "decision": "CONFIRM" or "REJECT" or "MODIFY",\n'
            '  "confidence": <0.0 to 1.0>,\n'
            '  "reasoning": "<2-3 sentence explanation>",\n'
            '  "suggested_sl": <price or null>,\n'
            '  "suggested_target": <price or null>,\n'
            '  "key_risks": ["<risk1>", "<risk2>"],\n'
            '  "optimization": "<one actionable improvement suggestion>"\n'
            '}'
        )
        return prompt

    # ------------------------------------------------------------------
    # Response parsers
    # ------------------------------------------------------------------

    def _parse_trade_response(
        self, raw: str, signal: Signal, latency: int
    ) -> AIAnalysis:
        data = self._parse_json_safely(raw)
        if data is None:
            log.warning("Could not parse AI JSON response; passing through.")
            return AIAnalysis(
                decision="PASS_THROUGH", confidence=signal.confidence,
                reasoning="AI response parse failed.",
                raw_response=raw[:500], latency_ms=latency,
            )

        decision   = str(data.get("decision", "PASS_THROUGH")).upper()
        confidence = float(data.get("confidence", signal.confidence))
        reasoning  = str(data.get("reasoning", ""))

        suggested_sl  = data.get("suggested_sl")
        suggested_tgt = data.get("suggested_target")

        # If AI confidence is too low, treat as rejection
        if decision == "CONFIRM" and confidence < self._cfg.min_confidence:
            decision = "REJECT"
            reasoning = f"[Auto-downgraded] AI confidence {confidence:.2f} < threshold {self._cfg.min_confidence}. " + reasoning

        return AIAnalysis(
            decision=decision,
            confidence=confidence,
            reasoning=reasoning,
            suggested_sl=float(suggested_sl) if suggested_sl else None,
            suggested_target=float(suggested_tgt) if suggested_tgt else None,
            key_risks=list(data.get("key_risks", [])),
            optimization_notes=str(data.get("optimization", "")),
            raw_response=raw[:1000],
            latency_ms=latency,
        )

    @staticmethod
    def _parse_json_safely(text: str) -> Optional[dict]:
        """Extract and parse the first JSON object in *text*."""
        if not text:
            return None
        # Strip markdown code fences
        text = re.sub(r"```(?:json)?", "", text).strip()
        # Try to find the outermost JSON object
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return None
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            return None

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        try:
            resp = self._session.get(
                f"{self._cfg.ollama_url}/api/tags", timeout=5
            )
            return resp.status_code == 200
        except Exception:
            return False


# Singleton
ai_engine = AIEngine()
