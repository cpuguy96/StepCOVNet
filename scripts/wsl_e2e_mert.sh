#!/usr/bin/env bash
set -euo pipefail

# shellcheck source=scripts/wsl_common.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/wsl_common.sh"
cd "$REPO_ROOT"

bash "$REPO_ROOT/scripts/wsl_ensure_env.sh"
# shellcheck source=scripts/wsl_gpu_env.sh
source "$REPO_ROOT/scripts/wsl_gpu_env.sh"
PY="$STEPCOVNET_WSL_PYTHON"

echo "Checking PyTorch CUDA..."
"$PY" - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0))
PY

echo "Extracting MERT features on GPU..."
"$PY" scripts/extract_mert_features.py \
  --data_dir=tests/testdata \
  --output_dir=e2e_mert/mert_wsl \
  --device=cuda

echo "Verifying features..."
"$PY" - <<'PY'
import numpy as np
from pathlib import Path
path = Path("e2e_mert/mert_wsl/mayu.mert.npy")
arr = np.load(path)
print("features", path, "shape", arr.shape, "dtype", arr.dtype)
PY

echo "Training onset model on MERT features..."
"$PY" scripts/train_onset.py --config=configs/local_e2e_mert_wsl.json

echo "Done."
