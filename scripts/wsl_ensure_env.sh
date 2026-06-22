#!/usr/bin/env bash
# Ensure the WSL GPU Python environment exists for stepcovnet.
set -euo pipefail

# shellcheck source=scripts/wsl_common.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/wsl_common.sh"
cd "$REPO_ROOT"

UV="${UV:-$HOME/.local/bin/uv}"
PY312="${PY312:-$HOME/.local/share/uv/python/cpython-3.12-linux-x86_64-gnu/bin/python3.12}"

if [[ ! -x "$UV" ]]; then
  echo "uv not found at $UV (required for WSL GPU environment)." >&2
  exit 1
fi

if [[ ! -x "$PY312" ]]; then
  "$UV" python install 3.12
fi

if [[ ! -x "$WSL_VENV/bin/python" ]]; then
  "$UV" venv "$WSL_VENV" --python "$PY312"
fi

PY="$STEPCOVNET_WSL_PYTHON"
"$PY" -c "import zlib"

"$UV" pip install --python "$PY" -U pip wheel setuptools -q
"$UV" pip install --python "$PY" -e ".[ssl,gpu]" -q
