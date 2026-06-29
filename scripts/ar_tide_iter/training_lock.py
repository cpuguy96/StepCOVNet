"""Exclusive lock so only one AR tide iteration trains on the GPU at a time.

``nvidia-smi`` alone is not enough: two ``run_exp.py`` processes can both pass
the busy check before either TensorFlow process appears on the GPU.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

from stepcovnet import wsl_gpu

REPO = Path(__file__).resolve().parents[2]
LOCK_PATH = REPO / "logs" / "ar_tide_iter" / "gpu_training.lock"


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def read_lock() -> dict | None:
    if not LOCK_PATH.is_file():
        return None
    try:
        payload = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _write_lock_atomic(payload: dict) -> None:
    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    fd = os.open(str(LOCK_PATH), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    try:
        os.write(fd, text.encode("utf-8"))
    finally:
        os.close(fd)


def clear_stale_lock() -> bool:
    """Remove lock file when the holder process is gone. Returns True if cleared."""
    payload = read_lock()
    if payload is None:
        return False
    pid = int(payload.get("pid", 0))
    if _pid_alive(pid):
        return False
    try:
        LOCK_PATH.unlink(missing_ok=True)
    except OSError:
        return False
    return True


def acquire_training_lock(exp_id: str) -> None:
    """Take the iteration training lock or raise ``RuntimeError``."""
    payload = {
        "exp_id": exp_id,
        "pid": os.getpid(),
        "started_at": datetime.now().isoformat(timespec="seconds"),
    }
    if clear_stale_lock():
        pass
    if LOCK_PATH.is_file():
        _raise_lock_busy(read_lock())
    try:
        _write_lock_atomic(payload)
    except FileExistsError:
        clear_stale_lock()
        if LOCK_PATH.is_file():
            _raise_lock_busy(read_lock())
        _write_lock_atomic(payload)


def release_training_lock(exp_id: str | None = None) -> None:
    """Drop the lock when this process holds it."""
    payload = read_lock()
    if payload is None:
        return
    if int(payload.get("pid", -1)) != os.getpid():
        return
    if exp_id is not None and str(payload.get("exp_id")) != exp_id:
        return
    LOCK_PATH.unlink(missing_ok=True)


def _raise_lock_busy(payload: dict | None) -> None:
    if payload is None:
        raise RuntimeError(
            f"GPU training lock exists at {LOCK_PATH.relative_to(REPO)} "
            "but could not be read.",
        )
    exp_id = payload.get("exp_id", "?")
    pid = payload.get("pid", "?")
    started = payload.get("started_at", "?")
    raise RuntimeError(
        "Another AR tide iteration is already training "
        f"({exp_id}, pid={pid}, started={started}). "
        f"Lock: {LOCK_PATH.relative_to(REPO)}. "
        "Stop the other run_exp/run_overnight shell before starting a new job.",
    )


def assert_gpu_training_available(*, exp_id: str, force: bool = False) -> None:
    """Require a free GPU and no active iteration training lock."""
    if force:
        clear_stale_lock()
        wsl_gpu.assert_wsl_gpu_free_for_training(force=True)
        return
    if LOCK_PATH.is_file() and not clear_stale_lock():
        _raise_lock_busy(read_lock())
    wsl_gpu.assert_wsl_gpu_free_for_training(force=False)
