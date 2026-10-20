#!/usr/bin/env bash
# Script to launch SmartRoute FastAPI Gateway
set -e

# Change to script directory
cd "$(dirname "$0")"

# Activate virtual environment if present
if [ -d "venv" ]; then
    source venv/bin/activate
fi

echo "🚀 Starting SmartRoute AI Gateway on http://localhost:8000 ..."
python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
