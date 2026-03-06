#!/bin/bash

# run.sh — helper script for mode switching

source venv/bin/activate

MODE=${1:-paper}

echo "Starting bot in $MODE mode..."

python main.py $MODE
