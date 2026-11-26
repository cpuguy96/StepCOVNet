"""Training reproducibility helpers (PyTorch, NumPy, and Python RNG state)."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def apply_training_seed(seed: int) -> None:
    """Set Python, NumPy, and PyTorch RNG state before model build.

    Call once at the start of a training run, before constructing models or
    reading data that depends on randomness.

    Args:
        seed: Integer random seed.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
