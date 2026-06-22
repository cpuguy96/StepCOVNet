#!/usr/bin/env bash
set -euo pipefail

# shellcheck source=scripts/wsl_common.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/wsl_common.sh"
cd "$REPO_ROOT"

bash "$REPO_ROOT/scripts/wsl_ensure_env.sh"
# shellcheck source=scripts/wsl_gpu_env.sh
source "$REPO_ROOT/scripts/wsl_gpu_env.sh"
PY="$STEPCOVNET_WSL_PYTHON"
AUDIO="$REPO_ROOT/tests/testdata/mayu.ogg"
OUT_DIR="$REPO_ROOT/e2e_mert/bench"

mkdir -p "$OUT_DIR"

"$PY" - <<'PY'
import torch
print("cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device_name", torch.cuda.get_device_name(0))
PY

for dev in cpu cuda; do
  out="$OUT_DIR/mayu_${dev}.mert.npy"
  rm -f "$out"
  echo "=== device=$dev ==="
  start=$(date +%s.%N)
  "$PY" - <<PY
from stepcovnet import ssl_features
ssl_features.extract_and_save_mert_features(
    "$AUDIO",
    "$out",
    device="$dev",
)
PY
  end=$(date +%s.%N)
  elapsed=$(awk -v s="$start" -v e="$end" 'BEGIN { printf "%.2f", e - s }')
  echo "elapsed_sec $elapsed"
  ls -lh "$out"
  echo
done
