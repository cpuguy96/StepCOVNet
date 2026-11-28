"""Tests for scripts/colab_run_remote.sh provisioning timeout and diagnostics."""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT_PATH = _REPO_ROOT / "scripts" / "colab_run_remote.sh"
_SAMPLE_CONFIG = (
    "configs/ddc/itgpt_mert_placement_fraxtil_orig_cosine64_l2_6_10_16_ema.json"
)


def _write_executable(path: Path, content: str) -> None:
    """Write an executable script file.

    Args:
        path: Destination path for the executable script.
        content: Script contents to write.
    """
    path.write_text(content, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


class ColabRunRemoteTests(unittest.TestCase):
    """Tests for provisioning timeout and colab.log diagnostic output."""

    def test_script_sets_default_provision_timeout_and_redacts_tokens(self) -> None:
        """Verify 240s default provisioning timeout, flag, and redaction."""
        text = _SCRIPT_PATH.read_text(encoding="utf-8")
        self.assertIn('PROVISION_TIMEOUT="${COLAB_PROVISION_TIMEOUT:-240}"', text)
        self.assertIn('--provision-timeout) PROVISION_TIMEOUT="$2"; shift 2 ;;', text)
        self.assertIn("Bearer REDACTED", text)
        self.assertIn("token=REDACTED", text)

    @unittest.skipIf(
        sys.platform == "win32"
        or shutil.which("bash") is None
        or shutil.which("timeout") is None,
        "POSIX bash and timeout required",
    )
    def test_provision_timeout_surfaces_redacted_colab_log_diagnostics(self) -> None:
        """Timeout during colab new exits 124 and prints redacted diagnostics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            bin_dir = tmp / "bin"
            bin_dir.mkdir()
            log_file = tmp / "colab.log"
            log_file.write_text("2026-01-01 ERROR - stale old line\n", encoding="utf-8")

            _write_executable(
                bin_dir / "colab",
                "#!/usr/bin/env bash\n"
                "while [[ $# -gt 0 ]]; do\n"
                '  case "$1" in\n'
                "    --auth|--config) shift 2 ;;\n"
                "    *) break ;;\n"
                "  esac\n"
                "done\n"
                'cmd="${1:-}"\n'
                'case "$cmd" in\n'
                '  status) echo "not found"; exit 1 ;;\n'
                "  stop) exit 0 ;;\n"
                "  new)\n"
                '    cat >> "$COLAB_LOG_FILE" <<EOF\n'
                "2026-11-28 colab_cli.client - DEBUG - Response: 503 Service Unavailable\n"
                "2026-11-28 colab_cli.client - WARNING - Assignment request failed; retrying in 10s\n"
                "2026-11-28 colab_cli.client - ERROR - authorization: Bearer super-secret-tok\n"
                "EOF\n"
                "    sleep 5\n"
                "    ;;\n"
                "  *) exit 0 ;;\n"
                "esac\n",
            )
            _write_executable(bin_dir / "rclone", "#!/usr/bin/env bash\nexit 0\n")

            env = os.environ.copy()
            env["PATH"] = f"{bin_dir}:{env.get('PATH', '')}"
            env["COLAB_LOG_FILE"] = str(log_file)

            proc = subprocess.run(
                [
                    "bash",
                    str(_SCRIPT_PATH),
                    "--session",
                    "test-s",
                    "--gpu",
                    "T4",
                    "--run-name",
                    "test-r",
                    "--config",
                    _SAMPLE_CONFIG,
                    "--provision-timeout",
                    "1",
                ],
                cwd=_REPO_ROOT,
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(proc.returncode, 124, msg=proc.stderr)
            self.assertIn("timed out after 1s", proc.stderr)
            self.assertIn("503 Service Unavailable", proc.stderr)
            self.assertIn("Assignment request failed", proc.stderr)
            self.assertIn("Bearer REDACTED", proc.stderr)
            self.assertNotIn("super-secret-tok", proc.stderr)
            self.assertNotIn("stale old line", proc.stderr)

    @unittest.skipIf(
        sys.platform == "win32"
        or shutil.which("bash") is None
        or shutil.which("timeout") is None,
        "POSIX bash and timeout required",
    )
    def test_provision_failure_surfaces_nonzero_exit_and_diagnostics(self) -> None:
        """Non-124 failure during colab new propagates exit code and diagnostics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            bin_dir = tmp / "bin"
            bin_dir.mkdir()
            log_file = tmp / "colab.log"
            log_file.write_text("2026-01-01 ERROR - stale old line\n", encoding="utf-8")

            _write_executable(
                bin_dir / "colab",
                "#!/usr/bin/env bash\n"
                "while [[ $# -gt 0 ]]; do\n"
                '  case "$1" in\n'
                "    --auth|--config) shift 2 ;;\n"
                "    *) break ;;\n"
                "  esac\n"
                "done\n"
                'cmd="${1:-}"\n'
                'case "$cmd" in\n'
                '  status) echo "not found"; exit 1 ;;\n'
                "  stop) exit 0 ;;\n"
                "  new)\n"
                '    echo "2026-11-28 colab_cli - ERROR - Quota exceeded" >> "$COLAB_LOG_FILE"\n'
                "    exit 2\n"
                "    ;;\n"
                "  *) exit 0 ;;\n"
                "esac\n",
            )
            _write_executable(bin_dir / "rclone", "#!/usr/bin/env bash\nexit 0\n")

            env = os.environ.copy()
            env["PATH"] = f"{bin_dir}:{env.get('PATH', '')}"
            env["COLAB_LOG_FILE"] = str(log_file)

            proc = subprocess.run(
                [
                    "bash",
                    str(_SCRIPT_PATH),
                    "--session",
                    "test-s",
                    "--gpu",
                    "T4",
                    "--run-name",
                    "test-r",
                    "--config",
                    _SAMPLE_CONFIG,
                    "--provision-timeout",
                    "5",
                ],
                cwd=_REPO_ROOT,
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(proc.returncode, 2, msg=proc.stderr)
            self.assertIn("failed (exit 2)", proc.stderr)
            self.assertIn("Quota exceeded", proc.stderr)
            self.assertNotIn("timed out", proc.stderr)
            self.assertNotIn("stale old line", proc.stderr)
