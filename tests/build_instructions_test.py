import io
import pathlib
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import pytest

# Allow importing the script module (scripts/build_generate_ui_binary.py)
_SCRIPT_DIR = pathlib.Path(__file__).resolve().parent.parent / "scripts"
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

_PROJECT_ROOT = _SCRIPT_DIR.parent
_SPEC_PATH = _SCRIPT_DIR / "generate_ui.spec"
_ENTRY_SCRIPT_PATH = _SCRIPT_DIR / "generate_ui.py"
_BUILD_HELPER_PATH = _SCRIPT_DIR / "build_generate_ui_binary.py"
_EXE_NAME = "generate_ui.exe" if sys.platform == "win32" else "generate_ui"

import build_generate_ui_binary  # noqa: E402


class BuildArtifactsExistTest(unittest.TestCase):
    """Check that all files referenced in the README build instructions exist."""

    def test_spec_file_exists(self):
        self.assertTrue(
            _SPEC_PATH.is_file(),
            f"Spec file should exist at scripts/generate_ui.spec: {_SPEC_PATH}",
        )

    def test_entry_script_exists(self):
        self.assertTrue(
            _ENTRY_SCRIPT_PATH.is_file(),
            f"Entry script should exist at scripts/generate_ui.py: {_ENTRY_SCRIPT_PATH}",
        )

    def test_build_helper_script_exists(self):
        self.assertTrue(
            _BUILD_HELPER_PATH.is_file(),
            f"Build helper should exist at scripts/build_generate_ui_binary.py: {_BUILD_HELPER_PATH}",
        )

    def test_spec_refers_to_entry_script(self):
        with _SPEC_PATH.open(encoding="utf-8") as f:
            content = f.read()
        self.assertIn(
            "generate_ui.py", content, "Spec should reference the entry script"
        )


class BuildGenerateUiBinaryMainTest(unittest.TestCase):
    """Unit tests for build_generate_ui_binary.main() (error handling and return codes)."""

    def test_main_exits_1_when_spec_file_missing(self):
        """When the spec file does not exist, main() exits with 1 and prints to stderr."""
        stderr = io.StringIO()
        with (
            mock.patch.object(pathlib.Path, "is_file", return_value=False),
            mock.patch.object(sys, "stderr", stderr),
            self.assertRaises(SystemExit) as cm,
        ):
            build_generate_ui_binary.main()
        self.assertEqual(cm.exception.code, 1)
        self.assertIn("Spec file not found", stderr.getvalue())
        self.assertIn("generate_ui.spec", stderr.getvalue())

    def test_main_propagates_subprocess_return_code(self):
        """main() exits with the same return code as the PyInstaller subprocess."""
        with (
            mock.patch.object(pathlib.Path, "is_file", return_value=True),
            mock.patch.object(build_generate_ui_binary.os, "chdir", autospec=True),
            mock.patch.object(
                build_generate_ui_binary.subprocess,
                "run",
                return_value=subprocess.CompletedProcess(
                    args=[], returncode=3, stdout=b"", stderr=b"",
                ),
                autospec=True,
            ) as run_mock,
            self.assertRaises(SystemExit) as cm,
        ):
            build_generate_ui_binary.main()
        self.assertEqual(cm.exception.code, 3)
        run_mock.assert_called_once()
        call_args = run_mock.call_args[0][0]
        self.assertEqual(call_args[0], sys.executable)
        self.assertEqual(call_args[1], "-m")
        self.assertEqual(call_args[2], "PyInstaller")
        self.assertTrue(str(call_args[3]).endswith("generate_ui.spec"))

    def test_main_exits_0_on_success(self):
        """When PyInstaller succeeds, main() exits with 0."""
        with (
            mock.patch.object(pathlib.Path, "is_file", return_value=True),
            mock.patch.object(build_generate_ui_binary.os, "chdir", autospec=True),
            mock.patch.object(
                build_generate_ui_binary.subprocess,
                "run",
                return_value=subprocess.CompletedProcess(
                    args=[], returncode=0, stdout=b"", stderr=b"",
                ),
                autospec=True,
            ) as run_mock,
            self.assertRaises(SystemExit) as cm,
        ):
            build_generate_ui_binary.main()
        self.assertEqual(cm.exception.code, 0)
        run_mock.assert_called_once()
        kwargs = run_mock.call_args[1]
        self.assertEqual(kwargs["cwd"], _PROJECT_ROOT)


class BuildInstructionsRunTest(unittest.TestCase):
    """Run the build and verify output. Requires PyInstaller and stepcovnet; marked slow."""

    @pytest.mark.slow
    def test_build_helper_produces_executable(self):
        """Running the build (per README) should succeed and create generate_ui[.exe] in a temp dir."""
        try:
            import PyInstaller  # noqa: F401
        except ImportError:
            self.skipTest(
                "PyInstaller not installed; install with pip install .[build]"
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            distpath = pathlib.Path(tmpdir) / "dist"
            workpath = pathlib.Path(tmpdir) / "build"
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "PyInstaller",
                    "--distpath",
                    distpath,
                    "--workpath",
                    workpath,
                    _SPEC_PATH,
                ],
                cwd=_PROJECT_ROOT,
                capture_output=True,
                text=True,
                timeout=600,
            )
            self.assertEqual(
                result.returncode,
                0,
                f"Build should exit 0. stderr: {result.stderr!r} stdout: {result.stdout!r}",
            )
            exe_path = pathlib.Path(distpath) / _EXE_NAME
            self.assertTrue(
                exe_path.is_file(),
                f"Executable should exist at {exe_path} after build",
            )
