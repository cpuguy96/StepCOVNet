import os
import pathlib
import tempfile
import unittest
from unittest import mock

from stepcovnet import wsl_gpu_lock


class WslGpuLockTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self._lock_path = pathlib.Path(self._tmpdir.name) / "gpu_wsl.lock"
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
        wsl_gpu_lock.acquire_gpu_lock("job_a")
        payload = wsl_gpu_lock.read_lock()
        assert payload is not None
        self.assertEqual(payload["job"], "job_a")
        self.assertEqual(int(payload["pid"]), os.getpid())
        wsl_gpu_lock.release_gpu_lock("job_a")
        self.assertIsNone(wsl_gpu_lock.read_lock())

    def test_second_acquire_raises(self) -> None:
        wsl_gpu_lock.acquire_gpu_lock("job_a")
        try:
            with self.assertRaises(RuntimeError):
                wsl_gpu_lock.acquire_gpu_lock("job_b")
        finally:
            wsl_gpu_lock.release_gpu_lock("job_a")

    def test_clear_stale_lock_when_pid_dead(self) -> None:
        self._lock_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock_path.write_text(
            '{"job":"old","pid":999999,"platform":"linux","started_at":"2026-01-01"}\n',
            encoding="utf-8",
        )
        with mock.patch.object(wsl_gpu_lock, "_pid_alive", return_value=False, autospec=True):
            self.assertTrue(wsl_gpu_lock.clear_stale_lock())
        self.assertFalse(self._lock_path.is_file())

    def test_gpu_job_lock_skips_when_parent_holds(self) -> None:
        with (
            mock.patch.dict(os.environ, {"STEPCOVNET_GPU_LOCK_HELD": "1"}),
            wsl_gpu_lock.gpu_job_lock("child"),
        ):
            pass
        self.assertIsNone(wsl_gpu_lock.read_lock())

    def test_ensure_gpu_job_lock_idempotent(self) -> None:
        wsl_gpu_lock.ensure_gpu_job_lock("job_a")
        try:
            wsl_gpu_lock.ensure_gpu_job_lock("job_a")
            payload = wsl_gpu_lock.read_lock()
            assert payload is not None
            self.assertEqual(payload["job"], "job_a")
        finally:
            wsl_gpu_lock._release_registered_lock()
        self.assertIsNone(wsl_gpu_lock.read_lock())


if __name__ == "__main__":
    unittest.main()
