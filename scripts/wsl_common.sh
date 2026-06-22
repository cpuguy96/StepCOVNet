#!/usr/bin/env bash
# Portable paths for WSL helper scripts (any clone location, any WSL user).
# Source from other scripts/wsl_*.sh after setting REPO_ROOT, or let this file set it.

_wsl_common_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${_wsl_common_dir}/.." && pwd)}"
WSL_VENV="${WSL_VENV:-$HOME/stepcovnet-venv-wsl}"
STEPCOVNET_WSL_PYTHON="${STEPCOVNET_WSL_PYTHON:-$WSL_VENV/bin/python}"
export REPO_ROOT WSL_VENV STEPCOVNET_WSL_PYTHON
unset _wsl_common_dir
