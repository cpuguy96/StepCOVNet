"""Exclusive file lock for WSL GPU jobs (training, decode, MERT extract).

``nvidia-smi`` alone races when two processes start in the same window before
either appears on the GPU. Lock path: ``logs/gpu_wsl.lock`` (repo-relative).
"""

from __future__ import annotations

import atexit
import json
import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from stepcovnet import wsl_gpu

LOCK_REL = Path("logs") / "gpu_wsl.lock"
GPU_LOCK_HELD_ENV = "STEPCOVNET_GPU_LOCK_HELD"
_registered_lock_job: str | None = None
_registered_lock_start: str | None = None


def lock_path(start: str | None = None) -> Path:
    """Return absolute path to the shared GPU lock file."""
    return wsl_gpu.find_repo_root(start) / LOCK_REL


def read_lock(start: str | None = None) -> dict | None:
    path = lock_path(start)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _write_lock_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    try:
        os.write(fd, text.encode("utf-8"))
    finally:
        os.close(fd)


def _pid_alive(pid: int, holder_platform: str) -> bool:
    if pid <= 0:
        return False
    if holder_platform == "win32":
        if wsl_gpu.is_windows():
            try:
                os.kill(pid, 0)
            except OSError:
                return False
            return True
        # Windows parent holds lock while WSL subprocess runs; do not clear from WSL.
        return True
    if wsl_gpu.is_windows():
        return wsl_gpu.wsl_pid_is_alive(pid)
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def clear_stale_lock(start: str | None = None) -> bool:
    """Remove lock when the holder process is gone. Returns True if cleared."""
    path = lock_path(start)
    payload = read_lock(start)
    if payload is None:
        return False
    pid = int(payload.get("pid", 0))
    holder_platform = str(payload.get("platform", "linux"))
    if _pid_alive(pid, holder_platform):
        return False
    try:
        path.unlink(missing_ok=True)
    except OSError:
        return False
    return True


def _raise_lock_busy(payload: dict | None, *, path: Path) -> None:
    if payload is None:
        raise RuntimeError(
            f"GPU lock exists at {path} but could not be read.",
        )
    job = payload.get("job", "?")
    pid = payload.get("pid", "?")
    started = payload.get("started_at", "?")
    raise RuntimeError(
        f"Another GPU job is already running ({job}, pid={pid}, started={started}). "
        f"Lock: {path}. Wait for it to finish or set STEPCOVNET_FORCE_GPU=1.",
    )


def assert_gpu_lock_available(*, start: str | None = None, force: bool = False) -> None:
    """Raise when the shared GPU lock file is held by a live process."""
    if force or wsl_gpu.gpu_force_enabled():
        clear_stale_lock(start)
        return
    if gpu_lock_held_by_parent():
        return
    path = lock_path(start)
    if path.is_file() and not clear_stale_lock(start):
        _raise_lock_busy(read_lock(start), path=path)


def acquire_gpu_lock(job: str, *, start: str | None = None) -> None:
    """Take the GPU lock or raise ``RuntimeError``."""
    path = lock_path(start)
    payload = {
        "job": job,
        "pid": os.getpid(),
        "platform": sys.platform,
        "started_at": datetime.now().isoformat(timespec="seconds"),
    }
    clear_stale_lock(start)
    if path.is_file():
        _raise_lock_busy(read_lock(start), path=path)
    try:
        _write_lock_atomic(path, payload)
    except FileExistsError:
        clear_stale_lock(start)
        if path.is_file():
            _raise_lock_busy(read_lock(start), path=path)
        _write_lock_atomic(path, payload)


def release_gpu_lock(job: str | None = None, *, start: str | None = None) -> None:
    """Drop the lock when this process holds it."""
    path = lock_path(start)
    payload = read_lock(start)
    if payload is None:
        return
    if int(payload.get("pid", -1)) != os.getpid():
        return
    if job is not None and str(payload.get("job")) != job:
        return
    path.unlink(missing_ok=True)


def gpu_lock_held_by_parent() -> bool:
    """True when Windows dispatch already holds the lock for this WSL child."""
    return os.environ.get(GPU_LOCK_HELD_ENV) == "1"


def _release_registered_lock() -> None:
    global _registered_lock_job, _registered_lock_start
    if _registered_lock_job is None:
        return
    release_gpu_lock(_registered_lock_job, start=_registered_lock_start)
    _registered_lock_job = None
    _registered_lock_start = None


def ensure_gpu_job_lock(job: str, *, start: str | None = None) -> None:
    """Acquire the GPU lock once per process; release on interpreter exit."""
    global _registered_lock_job, _registered_lock_start
    if (
        _registered_lock_job is not None
        or gpu_lock_held_by_parent()
        or wsl_gpu.gpu_force_enabled()
    ):
        return
    acquire_gpu_lock(job, start=start)
    _registered_lock_job = job
    _registered_lock_start = start
    atexit.register(_release_registered_lock)


@contextmanager
def gpu_job_lock(job: str, *, start: str | None = None) -> Iterator[None]:
    """Ensure the shared GPU lock for this process (released via ``atexit``)."""
    ensure_gpu_job_lock(job, start=start)
    yield
