#!/bin/bash
# launch.sh — Activate venv and start the Streamlit dashboard
# Usage: bash launch.sh

cd "$(dirname "$0")"

# Activate virtualenv
source .venv/bin/activate

# Make sure tensorflow is available (uses the system install or whatever is in path)
echo "🐍 Python: $(python3 --version)"
echo "📦 Streamlit: $(python3 -m streamlit --version)"
echo ""
echo "🚀 Starting Sentiment Analysis Dashboard…"
echo "   Open http://localhost:8501 in your browser"
echo ""

python3 -m streamlit run app.py --server.headless false
