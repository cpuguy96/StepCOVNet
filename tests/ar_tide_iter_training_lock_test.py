"""Tests for AR tide iteration GPU training lock."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

_ITER_PKG = Path(__file__).resolve().parents[1] / "scripts" / "ar_tide_iter"
if str(_ITER_PKG) not in sys.path:
    sys.path.insert(0, str(_ITER_PKG))

import training_lock  # noqa: E402

from stepcovnet import wsl_gpu_lock  # noqa: E402


class TrainingLockTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self._lock_path = Path(self._tmpdir.name) / "gpu_wsl.lock"
        self._lock_patcher = mock.patch.object(
            wsl_gpu_lock,
            "lock_path",
            autospec=True,
            return_value=self._lock_path,
        )
        self._lock_patcher.start()

    def tearDown(self) -> None:
        self._lock_patcher.stop()
        self._tmpdir.cleanup()

    def test_acquire_and_release(self) -> None:
        training_lock.acquire_training_lock("iter99")
        self.assertTrue(self._lock_path.is_file())
        payload = training_lock.read_lock()
        assert payload is not None
        self.assertEqual(payload["job"], "iter99")
        self.assertEqual(payload["pid"], os.getpid())
        training_lock.release_training_lock("iter99")
        self.assertFalse(self._lock_path.is_file())

    def test_second_acquire_raises_while_held(self) -> None:
        training_lock.acquire_training_lock("iter99")
        try:
            with self.assertRaises(RuntimeError):
                training_lock.acquire_training_lock("iter100")
        finally:
            training_lock.release_training_lock("iter99")

    def test_stale_lock_cleared(self) -> None:
        self._lock_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock_path.write_text(
            '{"job": "old", "pid": 999999999, "platform": "linux", "started_at": "x"}\n',
            encoding="utf-8",
        )
        with mock.patch.object(wsl_gpu_lock, "_pid_alive", return_value=False):
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
