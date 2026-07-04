#!/usr/bin/env bash
# Tide overfit speed ablation: incremental consistency on/off + mixed precision.
set -euo pipefail

# shellcheck source=scripts/wsl_common.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/wsl_common.sh"
cd "$REPO_ROOT"

bash "$REPO_ROOT/scripts/wsl_ensure_env.sh"
# shellcheck source=scripts/wsl_gpu_env.sh
source "$REPO_ROOT/scripts/wsl_gpu_env.sh"
export STEPCOVNET_IN_WSL=1
PY="$STEPCOVNET_WSL_PYTHON"
LOG="$REPO_ROOT/logs/ar_tide_speed_ablation.log"
mkdir -p "$(dirname "$LOG")"

run_arm() {
  local name="$1"
  local config="$2"
  echo "=== ARM ${name} $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
  local start
  start=$(date +%s)
  "$PY" scripts/train_onset_ar.py --config="$config"
  local end
  end=$(date +%s)
  echo "=== ARM ${name} complete elapsed_s=$((end - start)) $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
}

{
  echo "=== AR tide speed ablation start $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
  run_arm "inc_on_fp32" "configs/ar/versions/tide_overfit/ab_inc_on.json"
  run_arm "inc_off_fp32" "configs/ar/versions/tide_overfit/ab_inc_off.json"
  run_arm "inc_on_fp16_xla" "configs/ar/versions/tide_overfit/ab_inc_on_fp16.json"
  echo "=== AR tide speed ablation complete $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
} 2>&1 | tee "$LOG"
