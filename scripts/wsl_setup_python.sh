#!/usr/bin/env bash
set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"

# shellcheck source=scripts/wsl_common.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/wsl_common.sh"

echo "=== uv python list ==="
uv python list | tail -25

echo "=== installing standalone CPython 3.12 ==="
uv python install 3.12

PY312="$(uv python find 3.12)"
echo "Found Python: $PY312"
"$PY312" -c "import zlib, sys; print('zlib ok', sys.executable)"

rm -rf "$WSL_VENV"
uv venv "$WSL_VENV" --python "$PY312"
"$STEPCOVNET_WSL_PYTHON" -c "import zlib; print('venv zlib ok')"
