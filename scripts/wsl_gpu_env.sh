#!/usr/bin/env bash
# Export LD_LIBRARY_PATH so TensorFlow can load nvidia-* wheel libraries in WSL.
# Source after wsl_ensure_env.sh (venv must exist).

# shellcheck source=scripts/wsl_common.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/wsl_common.sh"

if [[ ! -d "${WSL_VENV}" ]]; then
  return 0 2>/dev/null || exit 0
fi

_wsl_nvidia_libs="$(
  find "${WSL_VENV}" -path '*/nvidia/*/lib*' -type d 2>/dev/null | sort -u | paste -sd: -
)"

if [[ -n "${_wsl_nvidia_libs}" ]]; then
  export LD_LIBRARY_PATH="${_wsl_nvidia_libs}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

unset _wsl_nvidia_libs
