"""Tests that verify the standalone binary build instructions are correct."""

import os
import subprocess
import sys
import tempfile
import unittest
from io import StringIO
from unittest import mock

import pytest

# Allow importing the script module (scripts/build_generate_ui_binary.py)
_SCRIPT_DIR = os.path.join(os.path.dirname(__file__), "..", "scripts")
_SCRIPT_DIR = os.path.abspath(_SCRIPT_DIR)
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
_SPEC_PATH = os.path.join(_SCRIPT_DIR, "generate_ui.spec")
_ENTRY_SCRIPT_PATH = os.path.join(_SCRIPT_DIR, "generate_ui.py")
_BUILD_HELPER_PATH = os.path.join(_SCRIPT_DIR, "build_generate_ui_binary.py")
_EXE_NAME = "generate_ui.exe" if sys.platform == "win32" else "generate_ui"

import build_generate_ui_binary  # noqa: E402


class BuildArtifactsExistTest(unittest.TestCase):
    """Check that all files referenced in the README build instructions exist."""

    def test_spec_file_exists(self):
        self.assertTrue(
            os.path.isfile(_SPEC_PATH),
            f"Spec file should exist at scripts/generate_ui.spec: {_SPEC_PATH}",
        )

    def test_entry_script_exists(self):
        self.assertTrue(
            os.path.isfile(_ENTRY_SCRIPT_PATH),
            f"Entry script should exist at scripts/generate_ui.py: {_ENTRY_SCRIPT_PATH}",
        )

    def test_build_helper_script_exists(self):
        self.assertTrue(
            os.path.isfile(_BUILD_HELPER_PATH),
            f"Build helper should exist at scripts/build_generate_ui_binary.py: {_BUILD_HELPER_PATH}",
        )

    def test_spec_refers_to_entry_script(self):
        with open(_SPEC_PATH, encoding="utf-8") as f:
            content = f.read()
        self.assertIn(
            "generate_ui.py", content, "Spec should reference the entry script"
        )


class BuildGenerateUiBinaryMainTest(unittest.TestCase):
    """Unit tests for build_generate_ui_binary.main() (error handling and return codes)."""

    def test_main_exits_1_when_spec_file_missing(self):
        """When the spec file does not exist, main() exits with 1 and prints to stderr."""
        with mock.patch(
            "build_generate_ui_binary.os.path.isfile", return_value=False
        ) as isfile_mock:
            stderr = StringIO()
            with mock.patch.object(sys, "stderr", stderr):
                with self.assertRaises(SystemExit) as cm:
                    build_generate_ui_binary.main()
        self.assertEqual(cm.exception.code, 1)
        self.assertIn("Spec file not found", stderr.getvalue())
        self.assertIn("generate_ui.spec", stderr.getvalue())
        isfile_mock.assert_called_once()

    def test_main_propagates_subprocess_return_code(self):
        """main() exits with the same return code as the PyInstaller subprocess."""
        with (
            mock.patch("build_generate_ui_binary.os.path.isfile", return_value=True),
            mock.patch("build_generate_ui_binary.os.chdir"),
        ):
            with mock.patch(
                "build_generate_ui_binary.subprocess.run",
                return_value=mock.MagicMock(returncode=3),
            ) as run_mock:
                with self.assertRaises(SystemExit) as cm:
                    build_generate_ui_binary.main()
        self.assertEqual(cm.exception.code, 3)
        run_mock.assert_called_once()
        call_args = run_mock.call_args[0][0]
        self.assertEqual(call_args[0], sys.executable)
        self.assertEqual(call_args[1], "-m")
        self.assertEqual(call_args[2], "PyInstaller")
        self.assertTrue(call_args[3].endswith("generate_ui.spec"))

    def test_main_exits_0_on_success(self):
        """When PyInstaller succeeds, main() exits with 0."""
        with (
            mock.patch("build_generate_ui_binary.os.path.isfile", return_value=True),
            mock.patch("build_generate_ui_binary.os.chdir"),
        ):
            with mock.patch(
                "build_generate_ui_binary.subprocess.run",
                return_value=mock.MagicMock(returncode=0),
            ) as run_mock:
                with self.assertRaises(SystemExit) as cm:
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
            distpath = os.path.join(tmpdir, "dist")
            workpath = os.path.join(tmpdir, "build")
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
            exe_path = os.path.join(distpath, _EXE_NAME)
            self.assertTrue(
                os.path.isfile(exe_path),
                f"Executable should exist at {exe_path} after build",
            )
