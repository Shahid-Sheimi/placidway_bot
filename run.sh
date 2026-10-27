#!/usr/bin/env bash
# Run the PlacidWay chatbot server.
#
# Usage:
#   ./run.sh            # uses .venv if present, else system python
#   PORT=9000 ./run.sh # custom port
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

# Prefer a local virtualenv if it exists.
if [ -d "$DIR/.venv" ] && [ -f "$DIR/.venv/bin/python" ]; then
  PY="$DIR/.venv/bin/python"
else
  PY="python3"
fi

# Load .env if present (simple KEY=VALUE parser).
if [ -f "$DIR/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  . "$DIR/.env"
  set +a
fi

if [ ! -f "$DIR/data/kb_index.json" ]; then
  echo "[run] Knowledge base not found — building it now..."
  "$PY" "$DIR/src/scraper.py"
  "$PY" "$DIR/src/build_kb.py"
fi

exec "$PY" "$DIR/src/app.py"