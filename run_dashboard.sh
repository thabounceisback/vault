#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

REQUIREMENTS_HASH_FILE=".venv/requirements.sha256"

if [ ! -x ".venv/bin/streamlit" ]; then
  python3 -m venv .venv
fi

if ! sha256sum -c "$REQUIREMENTS_HASH_FILE" >/dev/null 2>&1; then
  .venv/bin/pip install -r requirements.txt
  sha256sum requirements.txt > "$REQUIREMENTS_HASH_FILE"
fi

exec .venv/bin/streamlit run app.py --server.address 127.0.0.1 --server.port 8501
