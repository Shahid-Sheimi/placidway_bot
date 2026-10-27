#!/usr/bin/env bash
# Create a virtual environment and install dependencies.
#
# Usage:
#   ./setup_venv.sh
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

VENV_DIR="$DIR/.venv"

if [ ! -d "$VENV_DIR" ]; then
  echo "[venv] creating $VENV_DIR"
  python3 -m venv "$VENV_DIR"
fi

echo "[venv] upgrading pip"
"$VENV_DIR/bin/python" -m pip install --upgrade pip -q

echo "[venv] installing dependencies"
"$VENV_DIR/bin/pip" install -r "$DIR/requirements.txt" -q

echo "[venv] done."
echo "Activate with: source $VENV_DIR/bin/activate"