
"""
ai_engine.py — Ollama LLM integration for AI-assisted trade decisions.
Optimized for 2-model setup with longer timeout (60s).
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import requests

from config import config
from logger import get_logger
from strategy import Signal

log = get_logger("AIEngine")

# ---------------------------------------------------------------------------
# Timeout settings (60 seconds per model)
# ---------------------------------------------------------------------------

_MODEL_TIMEOUTS: Dict[str, int] = {
    "mistral": 60,
    "mistral:latest": 60,
    "llama3": 60,
    "llama3:8b": 60,
}


# ---------------------------------------------------------------------------
# Response dataclass
# ---------------------------------------------------------------------------

@dataclass
class AIAnalysis:
    decision: str
    confidence: float
    reasoning: str
    suggested_sl: Optional[float] = None
    suggested_target: Optional[float] = None
    key_risks: List[str] = field(default_factory=list)
    optimization_notes: str = ""
    raw_response: str = ""
    latency_ms: int = 0

    @property
    def is_confirmed(self) -> bool:
        return self.decision in ("CONFIRM", "PASS_THROUGH")

    @property
    def is_rejected(self) -> bool:
        return self.decision == "REJECT"


# ---------------------------------------------------------------------------
# AI Engine
# ---------------------------------------------------------------------------

class AIEngine:

    def __init__(self) -> None:
        self._cfg = config.ai
        self._last_call: Dict[str, float] = {}
        self._active_model: str = (
            self._cfg.models[0] if self._cfg.models else self._cfg.model
        )

        self._session = requests.Session()
        self._session.headers.update({"Content-Type": "application/json"})

    # ------------------------------------------------------------------
    # Analyse trade signal
    # ------------------------------------------------------------------

    def analyse_signal(self, signal: Signal) -> AIAnalysis:

        if not self._cfg.enabled:
            return AIAnalysis(
                decision="PASS_THROUGH",
                confidence=signal.confidence,
                reasoning="AI disabled in config."
            )

        last = self._last_call.get(signal.symbol, 0.0)
        elapsed = time.time() - last

        if elapsed < self._cfg.call_interval_sec:
            remaining = int(self._cfg.call_interval_sec - elapsed)

            return AIAnalysis(
                decision="PASS_THROUGH",
                confidence=signal.confidence,
                reasoning=f"AI cooldown active ({remaining}s remaining)."
            )

        prompt = self._build_trade_prompt(signal)

        start = time.time()
        raw = self._call_with_fallback(prompt)
        latency = int((time.time() - start) * 1000)

        self._last_call[signal.symbol] = time.time()

        if raw is None:
            return AIAnalysis(
                decision="PASS_THROUGH",
                confidence=signal.confidence,
                reasoning="AI unavailable — using technical signal.",
                latency_ms=latency,
            )

        return self._parse_trade_response(raw, signal, latency)

    # ------------------------------------------------------------------
    # Model fallback chain
    # ------------------------------------------------------------------

    def _call_with_fallback(self, prompt: str) -> Optional[str]:

        models = self._cfg.models or [self._cfg.model]

        for model in models:

            model = self._normalize_model_name(model)

            timeout = _MODEL_TIMEOUTS.get(model, 60)

            result = self._call_model(model, prompt, timeout)

            if result:
                self._active_model = model
                return result

        log.warning("All AI models failed — using rule-based signal")
        return None

    # ------------------------------------------------------------------
    # Normalize model names
    # ------------------------------------------------------------------

    def _normalize_model_name(self, model: str) -> str:

        if model == "mistral":
            return "mistral:latest"

        if model == "llama3":
            return "llama3:8b"

        return model

    # ------------------------------------------------------------------
    # Single Ollama call
    # ------------------------------------------------------------------

    def _call_model(self, model: str, prompt: str, timeout: int) -> Optional[str]:

        url = f"{self._cfg.ollama_url}/api/generate"

        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.2,
                "top_p": 0.9,
                "num_predict": 512
            }
        }

        try:

            response = self._session.post(url, json=payload, timeout=timeout)

            response.raise_for_status()

            data = response.json()

            return data.get("response")

        except requests.exceptions.ConnectionError:

            log.warning("Ollama server not running at %s", self._cfg.ollama_url)

        except requests.exceptions.Timeout:

            log.warning("AI timeout for %s — trying next model", model)

        except Exception as e:

            log.error("Ollama call failed for %s: %s", model, str(e))

        return None

    # ------------------------------------------------------------------
    # Prompt builder
    # ------------------------------------------------------------------

    def _build_trade_prompt(self, signal: Signal) -> str:

        ind = signal.indicators
        entry = signal.entry_price
        sl = signal.suggested_sl
        tgt = signal.suggested_target

        prompt = f"""
You are a professional quantitative trader analysing an intraday signal.

SYMBOL: {signal.symbol}
SIGNAL: {signal.action}

Entry: {entry}
Stop Loss: {sl}
Target: {tgt}

Indicators:
RSI: {ind.get('rsi')}
MACD: {ind.get('macd')}
VWAP: {ind.get('vwap')}
Volume Ratio: {ind.get('vol_ratio')}

Technical Confidence: {signal.confidence}

Respond ONLY with JSON:

{{
 "decision":"CONFIRM | REJECT | MODIFY",
 "confidence":0.0,
 "reasoning":"short explanation",
 "suggested_sl":null,
 "suggested_target":null,
 "key_risks":[""],
 "optimization":""
}}
"""

        return prompt

    # ------------------------------------------------------------------
    # Parse response
    # ------------------------------------------------------------------

    def _parse_trade_response(self, raw: str, signal: Signal, latency: int) -> AIAnalysis:

        data = self._parse_json_safely(raw)

        if data is None:
            return AIAnalysis(
                decision="PASS_THROUGH",
                confidence=signal.confidence,
                reasoning="AI response parse failed",
                raw_response=raw[:400],
                latency_ms=latency,
            )

        decision = str(data.get("decision", "PASS_THROUGH")).upper()

        confidence = float(data.get("confidence", signal.confidence))

        reasoning = str(data.get("reasoning", ""))

        return AIAnalysis(
            decision=decision,
            confidence=confidence,
            reasoning=reasoning,
            suggested_sl=data.get("suggested_sl"),
            suggested_target=data.get("suggested_target"),
            key_risks=list(data.get("key_risks", [])),
            optimization_notes=str(data.get("optimization", "")),
            raw_response=raw[:1000],
            latency_ms=latency,
        )

    # ------------------------------------------------------------------
    # Safe JSON parser
    # ------------------------------------------------------------------

    def _parse_json_safely(self, text: str) -> Optional[dict]:

        if not text:
            return None

        text = re.sub(r"```(?:json)?", "", text).strip()

        match = re.search(r"\{.*\}", text, re.DOTALL)

        if not match:
            return None

        try:
            return json.loads(match.group())
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------

    def is_available(self) -> bool:

        try:
            r = self._session.get(
                f"{self._cfg.ollama_url}/api/tags",
                timeout=5
            )

            return r.status_code == 200

        except Exception:
            return False


ai_engine = AIEngine()
