"""Tests that verify the standalone binary build instructions are correct."""

import os
import subprocess
import sys
import tempfile
import unittest

import pytest

# Paths relative to project root (parent of tests/)
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_SCRIPTS_DIR = os.path.join(_PROJECT_ROOT, "scripts")
_SPEC_PATH = os.path.join(_SCRIPTS_DIR, "generate_ui.spec")
_ENTRY_SCRIPT_PATH = os.path.join(_SCRIPTS_DIR, "generate_ui.py")
_BUILD_HELPER_PATH = os.path.join(_SCRIPTS_DIR, "build_generate_ui_binary.py")
_EXE_NAME = "generate_ui.exe" if sys.platform == "win32" else "generate_ui"


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
