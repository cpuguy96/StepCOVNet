import os
import pathlib
import tempfile
import unittest
from unittest import mock

import tensorflow

from stepcovnet import wsl_gpu


class WslGpuTest(unittest.TestCase):
    def test_nvidia_library_dirs_empty_when_venv_missing(self):
        missing = wsl_gpu.nvidia_library_dirs(pathlib.Path("/nonexistent/venv"))
        self.assertEqual(missing, [])

    def test_apply_tensorflow_gpu_library_path_prepends_dirs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            lib_dir = (
                pathlib.Path(tmpdir)
                / "lib"
                / "python3.12"
                / "site-packages"
                / "nvidia"
                / "cudnn"
                / "lib"
            )
            lib_dir.mkdir(parents=True)
            os.environ.pop("LD_LIBRARY_PATH", None)
            applied = wsl_gpu.apply_tensorflow_gpu_library_path(pathlib.Path(tmpdir))
            self.assertTrue(applied)
            self.assertIn(str(lib_dir), os.environ["LD_LIBRARY_PATH"])

    def test_reexec_skips_when_gpu_env_marker_set(self):
        with (
            mock.patch.object(
                wsl_gpu, "is_running_in_wsl", return_value=True, autospec=True
            ),
            mock.patch.object(wsl_gpu, "find_repo_root", autospec=True) as find_root,
            mock.patch.dict(os.environ, {"STEPCOVNET_WSL_GPU_ENV": "1"}),
            mock.patch.object(wsl_gpu.os, "execvp", autospec=True) as execvp,
        ):
            find_root.return_value = pathlib.Path("/repo")
            wsl_gpu.reexec_with_tensorflow_gpu_env_if_needed(["/repo/script.py"])
            execvp.assert_not_called()

    def test_require_tensorflow_gpu_exits_in_wsl_when_no_device(self):
        with (
            mock.patch.object(
                wsl_gpu, "is_running_in_wsl", return_value=True, autospec=True
            ),
            mock.patch.object(
                tensorflow.config,
                "list_physical_devices",
                return_value=[],
                autospec=True,
            ),
            self.assertRaises(SystemExit),
        ):
            wsl_gpu.require_tensorflow_gpu()

    def test_wsl_gpu_compute_busy_when_active_apps(self):
        with mock.patch.object(
            wsl_gpu,
            "active_wsl_gpu_compute_apps",
            return_value=[(17, "python", "620 MiB")],
            autospec=True,
        ):
            self.assertTrue(wsl_gpu.wsl_gpu_compute_busy())

    def test_active_compute_apps_skip_dead_pids(self):
        with (
            mock.patch.object(
                wsl_gpu,
                "list_wsl_gpu_compute_apps",
                return_value=[(99, "python", "100 MiB")],
                autospec=True,
            ),
            mock.patch.object(
                wsl_gpu, "wsl_pid_is_alive", return_value=False, autospec=True
            ),
        ):
            self.assertEqual(wsl_gpu.active_wsl_gpu_compute_apps(), [])

    def test_assert_wsl_gpu_free_raises_when_training_script_running(self):
        with (
            mock.patch.object(
                wsl_gpu,
                "list_wsl_training_pids",
                return_value=[6973],
                autospec=True,
            ),
            self.assertRaises(RuntimeError),
        ):
            wsl_gpu.assert_wsl_gpu_free_for_training()

    def test_assert_wsl_gpu_free_raises_when_busy(self):
        with (
            mock.patch.object(
                wsl_gpu,
                "list_wsl_training_pids",
                return_value=[],
                autospec=True,
            ),
            mock.patch.object(
                wsl_gpu,
                "active_wsl_gpu_compute_apps",
                return_value=[(17, "python", "620 MiB")],
                autospec=True,
            ),
            self.assertRaises(RuntimeError),
        ):
            wsl_gpu.assert_wsl_gpu_free_for_training()


if __name__ == "__main__":
    unittest.main()
