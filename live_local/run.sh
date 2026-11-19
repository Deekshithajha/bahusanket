#!/bin/bash
# Starts BahuSanket at http://localhost:8000 (opens the browser when ready). Stop with Ctrl+C.
cd "$(dirname "$0")"
[ -x .venv/bin/python ] || { echo "Run setup first: bash setup.sh"; exit 1; }
exec .venv/bin/python server.py
