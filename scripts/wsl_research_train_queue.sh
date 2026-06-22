#!/usr/bin/env bash
# Sequential research training: gaussian then arch_large (after binary completes externally).
set -euo pipefail

# shellcheck source=scripts/wsl_common.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/wsl_common.sh"
cd "$REPO_ROOT"

bash "$REPO_ROOT/scripts/wsl_ensure_env.sh"
# shellcheck source=scripts/wsl_gpu_env.sh
source "$REPO_ROOT/scripts/wsl_gpu_env.sh"
export STEPCOVNET_IN_WSL=1
PY="$STEPCOVNET_WSL_PYTHON"

run_train() {
  local config="$1"
  local name="$2"
  local outdir="models_wsl/research/${name}"
  mkdir -p "$outdir"
  echo "=== training ${config} ==="
  "$PY" scripts/train_onset.py --config="$config" 2>&1 | tee "${outdir}/train.log"
}

run_train configs/research/gaussian_10train_40ep.json gaussian_10train_40ep
run_train configs/research/arch_large_10train_40ep.json arch_large_10train_40ep
echo "=== research queue complete ==="
