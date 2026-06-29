"""Tests for AR tide iteration GPU training lock."""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest import mock

_ITER_PKG = Path(__file__).resolve().parents[1] / "scripts" / "ar_tide_iter"
if str(_ITER_PKG) not in sys.path:
    sys.path.insert(0, str(_ITER_PKG))

import training_lock  # noqa: E402


class TrainingLockTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = training_lock.REPO / "logs" / "ar_tide_iter" / "_lock_test"
        self._tmpdir.mkdir(parents=True, exist_ok=True)
        self._orig_lock = training_lock.LOCK_PATH
        training_lock.LOCK_PATH = self._tmpdir / "gpu_training.lock"
        training_lock.LOCK_PATH.unlink(missing_ok=True)

    def tearDown(self) -> None:
        training_lock.LOCK_PATH.unlink(missing_ok=True)
        training_lock.LOCK_PATH = self._orig_lock

    def test_acquire_and_release(self) -> None:
        training_lock.acquire_training_lock("iter99")
        self.assertTrue(training_lock.LOCK_PATH.is_file())
        payload = training_lock.read_lock()
        assert payload is not None
        self.assertEqual(payload["exp_id"], "iter99")
        self.assertEqual(payload["pid"], os.getpid())
        training_lock.release_training_lock("iter99")
        self.assertFalse(training_lock.LOCK_PATH.is_file())

    def test_second_acquire_raises_while_held(self) -> None:
        training_lock.acquire_training_lock("iter99")
        try:
            with self.assertRaises(RuntimeError):
                training_lock.acquire_training_lock("iter100")
        finally:
            training_lock.release_training_lock("iter99")

    def test_stale_lock_cleared(self) -> None:
        training_lock.LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
        training_lock.LOCK_PATH.write_text(
            '{"exp_id": "old", "pid": 999999999, "started_at": "x"}\n',
            encoding="utf-8",
        )
        with mock.patch.object(training_lock, "_pid_alive", return_value=False):
            self.assertTrue(training_lock.clear_stale_lock())
        training_lock.acquire_training_lock("iter99")
        training_lock.release_training_lock("iter99")

    def test_assert_available_checks_lock(self) -> None:
        training_lock.acquire_training_lock("iter99")
        try:
            with mock.patch.object(
                training_lock.wsl_gpu,
                "assert_wsl_gpu_free_for_training",
                autospec=True,
            ) as assert_gpu:
                with self.assertRaises(RuntimeError):
                    training_lock.assert_gpu_training_available(
                        exp_id="iter100",
                        force=False,
                    )
                assert_gpu.assert_not_called()
        finally:
            training_lock.release_training_lock("iter99")


if __name__ == "__main__":
    unittest.main()
