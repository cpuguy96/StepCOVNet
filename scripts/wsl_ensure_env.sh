#!/usr/bin/env bash
# Ensure the WSL GPU Python environment exists for stepcovnet.
set -euo pipefail

# shellcheck source=scripts/wsl_common.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/wsl_common.sh"
cd "$REPO_ROOT"

UV="${UV:-$HOME/.local/bin/uv}"
PY_VER="${STEPCOVNET_WSL_PYTHON_VERSION:-3.13}"

if [[ ! -x "$UV" ]]; then
  echo "uv not found at $UV (required for WSL GPU environment)." >&2
  exit 1
fi

PY_BASE="$("$UV" python find "$PY_VER" 2>/dev/null || true)"
if [[ -z "$PY_BASE" || ! -x "$PY_BASE" ]]; then
  "$UV" python install "$PY_VER"
  PY_BASE="$("$UV" python find "$PY_VER")"
fi

if [[ ! -x "$WSL_VENV/bin/python" ]]; then
  "$UV" venv "$WSL_VENV" --python "$PY_BASE"
fi

PY="$STEPCOVNET_WSL_PYTHON"
"$PY" -c "import zlib"

"$UV" pip install --python "$PY" -U pip -q
"$UV" pip install --python "$PY" -e ".[ssl,dataset-prep]" -q
