#!/usr/bin/env bash
# MERT extract + dense baseline train for the 50/100 scoreboard subset.
set -euo pipefail

# shellcheck source=scripts/wsl_common.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/wsl_common.sh"
cd "$REPO_ROOT"

bash "$REPO_ROOT/scripts/wsl_ensure_env.sh"
# shellcheck source=scripts/wsl_gpu_env.sh
source "$REPO_ROOT/scripts/wsl_gpu_env.sh"
export STEPCOVNET_IN_WSL=1
PY="$STEPCOVNET_WSL_PYTHON"
LOG="$REPO_ROOT/logs/scoreboard_50t_100v_pipeline.log"
mkdir -p "$(dirname "$LOG")"

{
  echo "=== MERT extract $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
  "$PY" scripts/extract_mert_features.py \
    --training_index_path=data/final_data/training_index_scoreboard_50t_100v.json \
    --beside_audio --device=cuda --skip_existing
  echo "=== train $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
  "$PY" scripts/train_onset.py \
    --config=configs/onset_final_data_mert_bilstm_scoreboard_50t_100v.json
  echo "=== complete $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
} 2>&1 | tee "$LOG"
