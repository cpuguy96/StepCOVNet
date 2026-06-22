#!/usr/bin/env bash
set -euo pipefail

# shellcheck source=scripts/wsl_common.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/wsl_common.sh"
cd "$REPO_ROOT"

export STEPCOVNET_IN_WSL=1
bash "$REPO_ROOT/scripts/wsl_ensure_env.sh"
# shellcheck source=scripts/wsl_gpu_env.sh
source "$REPO_ROOT/scripts/wsl_gpu_env.sh"
PY="$STEPCOVNET_WSL_PYTHON"

echo "TensorFlow GPU devices:"
"$PY" - <<'PY'
import tensorflow as tf
print("tensorflow", tf.__version__)
print("gpus", tf.config.list_physical_devices("GPU"))
PY

echo ""
"$PY" scripts/probe_onset_event_sample_rate.py --device=gpu
