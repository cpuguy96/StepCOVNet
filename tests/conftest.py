"""Pytest configuration shared by the whole suite."""

from __future__ import annotations

import os

# Several tests import a module under scripts/ to exercise its helpers. Those
# scripts call wsl_gpu.bootstrap_gpu_script at import time, which re-execs the
# interpreter inside WSL and raises SystemExit, aborting collection on any
# machine where WSL is available. Tests only cover CPU code paths.
os.environ.setdefault("STEPCOVNET_NO_WSL", "1")
