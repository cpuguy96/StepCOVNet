from __future__ import annotations

import pathlib
import subprocess
import sys
import unittest
from unittest import mock

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pre_submit  # noqa: E402


class PreSubmitTest(unittest.TestCase):
    def test_find_code_changes_without_tests(self) -> None:
        self.assertEqual(
            pre_submit.find_code_changes_without_tests(["src/stepcovnet/foo.py"]),
            ["src/stepcovnet/foo.py"],
        )
        self.assertEqual(
            pre_submit.find_code_changes_without_tests(["scripts/foo.py"]),
            ["scripts/foo.py"],
        )
        self.assertEqual(
            pre_submit.find_code_changes_without_tests(
                ["src/stepcovnet/foo.py", "tests/foo_test.py"]
            ),
            [],
        )
        self.assertEqual(
            pre_submit.find_code_changes_without_tests(
                ["docs/a.md", "configs/x.json", "scripts/foo.sh"]
            ),
            [],
        )
        self.assertEqual(
            pre_submit.find_code_changes_without_tests(["src\\stepcovnet\\a.py"]),
            ["src/stepcovnet/a.py"],
        )

    def test_git_changed_paths_parses_porcelain_and_skips_deletes(self) -> None:
        porcelain = "\n".join(
            [
                " M src/stepcovnet/a.py",
                "?? tests/a_test.py",
                "R  scripts/old.py -> scripts/new.py",
                " D src/stepcovnet/removed.py",
                "D  scripts/deleted.py",
                '?? "scripts/quoted path.py"',
            ]
        )
        completed = subprocess.CompletedProcess(
            args=["git", "status", "--porcelain", "-uall"],
            returncode=0,
            stdout=porcelain,
            stderr="",
        )
        with mock.patch.object(
            pre_submit.subprocess,
            "run",
            autospec=True,
            return_value=completed,
        ) as mock_run:
            paths = pre_submit.git_changed_paths(REPO_ROOT)

        mock_run.assert_called_once()
        self.assertEqual(
            paths,
            [
                "src/stepcovnet/a.py",
                "tests/a_test.py",
                "scripts/new.py",
                "scripts/quoted path.py",
            ],
        )

    def test_git_changed_paths_returns_empty_on_git_error(self) -> None:
        completed = subprocess.CompletedProcess(
            args=["git", "status", "--porcelain", "-uall"],
            returncode=128,
            stdout="",
            stderr="fatal: not a git repository",
        )
        with mock.patch.object(
            pre_submit.subprocess,
            "run",
            autospec=True,
            return_value=completed,
        ):
            self.assertEqual(pre_submit.git_changed_paths(REPO_ROOT), [])

    def test_main_test_diff_check_gate_under_fast(self) -> None:
        base_argv = ["--fast", "--skip-ruff", "--skip-pyright"]
        with (
            mock.patch.object(
                pre_submit,
                "git_changed_paths",
                autospec=True,
                return_value=["src/stepcovnet/a.py"],
            ),
            self.assertRaises(SystemExit) as ctx,
        ):
            pre_submit.main(base_argv)
        self.assertEqual(ctx.exception.code, 1)

        with mock.patch.object(
            pre_submit,
            "git_changed_paths",
            autospec=True,
            return_value=["src/stepcovnet/a.py", "tests/a_test.py"],
        ):
            pre_submit.main(base_argv)

        with mock.patch.object(
            pre_submit,
            "git_changed_paths",
            autospec=True,
            return_value=["src/stepcovnet/a.py"],
        ):
            pre_submit.main([*base_argv, "--skip-test-diff-check"])


if __name__ == "__main__":
    unittest.main()
