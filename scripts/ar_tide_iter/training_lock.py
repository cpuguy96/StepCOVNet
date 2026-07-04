"""Exclusive lock for AR tide iteration training (delegates to ``wsl_gpu_lock``)."""

from __future__ import annotations

import time

from stepcovnet import wsl_gpu, wsl_gpu_lock

REPO = wsl_gpu.find_repo_root(__file__)
LOCK_PATH = wsl_gpu_lock.lock_path(__file__)


def read_lock() -> dict | None:
    return wsl_gpu_lock.read_lock(__file__)


def clear_stale_lock() -> bool:
    return wsl_gpu_lock.clear_stale_lock(__file__)


def acquire_training_lock(exp_id: str) -> None:
    wsl_gpu_lock.acquire_gpu_lock(exp_id, start=__file__)


def release_training_lock(exp_id: str | None = None) -> None:
    wsl_gpu_lock.release_gpu_lock(exp_id, start=__file__)


def assert_gpu_training_available(*, exp_id: str, force: bool = False) -> None:
    """Require a free GPU and no active iteration training lock."""
    if force:
        clear_stale_lock()
        wsl_gpu.assert_wsl_gpu_free_for_training(force=True)
        return
    wsl_gpu_lock.assert_gpu_lock_available(start=__file__)
    wsl_gpu.assert_wsl_gpu_free_for_training(force=False)


def wait_gpu_training_available(
    *,
    exp_id: str,
    timeout_sec: float,
    force: bool = False,
    poll_sec: float = 15.0,
) -> None:
    """Poll until the GPU and training lock are free or ``timeout_sec`` elapses."""
    if timeout_sec <= 0:
        assert_gpu_training_available(exp_id=exp_id, force=force)
        return
    deadline = time.monotonic() + timeout_sec
    while True:
        try:
            assert_gpu_training_available(exp_id=exp_id, force=force)
            return
        except RuntimeError as exc:
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    f"GPU still busy after {timeout_sec:.0f}s wait: {exc}",
                ) from exc
            remaining = deadline - time.monotonic()
            sleep_for = min(poll_sec, max(remaining, 0.0))
            print(
                f"GPU busy — retry in {sleep_for:.0f}s ({remaining:.0f}s left): {exc}",
            )
            time.sleep(sleep_for)
