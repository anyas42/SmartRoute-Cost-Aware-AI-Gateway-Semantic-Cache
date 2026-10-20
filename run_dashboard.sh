#!/usr/bin/env bash
# Script to launch SmartRoute Streamlit Dashboard
set -e

# Change to script directory
cd "$(dirname "$0")"

# Activate virtual environment if present
if [ -d "venv" ]; then
    source venv/bin/activate
fi

echo "📊 Launching SmartRoute Streamlit Monitoring Dashboard on http://localhost:8501 ..."
streamlit run dashboard/app.py --server.port 8501 --server.address 0.0.0.0
