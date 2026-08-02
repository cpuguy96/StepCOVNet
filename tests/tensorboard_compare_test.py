"""Tests for scripts/tensorboard_compare.py."""

from __future__ import annotations

import argparse
import io
import json
import pathlib
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest import mock

_SCRIPT_DIR = pathlib.Path(__file__).resolve().parent.parent / "scripts"
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import tensorboard_compare as tb  # noqa: E402


def _write_tfevents(
    directory: pathlib.Path, name: str = "events.out.tfevents.0"
) -> None:
    """Create a minimal TensorBoard event file marker under ``directory``."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / name).write_bytes(b"")


class LabelHelpersTest(unittest.TestCase):
    def test_label_strips_callbacks_prefix(self) -> None:
        self.assertEqual(
            tb._label_for_callback_root("callbacks/ar/ladder_50t_50v"),
            "ar/ladder_50t_50v",
        )

    def test_label_sanitizes_special_characters(self) -> None:
        self.assertEqual(
            tb._label_for_callback_root("callbacks/weird name!"),
            "weird_name_",
        )

    def test_sanitize_tb_label_removes_spec_separators(self) -> None:
        self.assertEqual(tb._sanitize_tb_label("a:b,c"), "a_b_c")


class DiscoveryTest(unittest.TestCase):
    def _make_callback_tree(
        self,
        root: pathlib.Path,
        callback_rel: str,
        run_names: list[str],
    ) -> pathlib.Path:
        logs_dir = root / callback_rel / "logs"
        for run_name in run_names:
            _write_tfevents(logs_dir / run_name)
        return logs_dir

    def test_has_tfevents_false_for_empty_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            empty = pathlib.Path(tmp)
            self.assertFalse(tb._has_tfevents(empty))

    def test_count_runs_counts_timestamped_subfolders(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            logs_dir = self._make_callback_tree(
                root,
                "callbacks/ar/ladder_10t_50v",
                ["20260725-run1", "20260726-run2"],
            )
            self.assertEqual(tb._count_runs(logs_dir), 2)

    def test_discover_log_groups_finds_nested_callbacks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            self._make_callback_tree(
                root,
                "callbacks/ar/ladder_10t_50v",
                ["20260725-run1"],
            )
            self._make_callback_tree(
                root,
                "callbacks/ar/ladder_50t_50v",
                ["20260726-run1", "20260727-run2"],
            )
            with mock.patch.object(tb, "REPO", root):
                groups = tb._discover_log_groups((root / "callbacks",))
            labels = {label for label, _logs, _count in groups}
            self.assertIn("ar/ladder_10t_50v", labels)
            self.assertIn("ar/ladder_50t_50v", labels)
            by_label = {label: count for label, _logs, count in groups}
            self.assertEqual(by_label["ar/ladder_50t_50v"], 2)


class ConfigAndPathResolutionTest(unittest.TestCase):
    def test_read_callback_logdir_from_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            callback_root = root / "callbacks" / "ar" / "demo"
            run_dir = callback_root / "logs" / "20260725-run"
            _write_tfevents(run_dir)
            config_path = root / "demo.json"
            config_path.write_text(
                json.dumps({"run": {"callback_root_dir": "callbacks/ar/demo"}}),
                encoding="utf-8",
            )
            with mock.patch.object(tb, "REPO", root):
                logs_dir = tb._read_callback_logdir(config_path)
            self.assertEqual(logs_dir, (callback_root / "logs").resolve())

    def test_read_callback_logdir_returns_none_without_events(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            config_path = root / "demo.json"
            config_path.write_text(
                json.dumps({"run": {"callback_root_dir": "callbacks/ar/demo"}}),
                encoding="utf-8",
            )
            with mock.patch.object(tb, "REPO", root):
                self.assertIsNone(tb._read_callback_logdir(config_path))

    def test_resolve_group_path_accepts_callback_parent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            logs_dir = root / "callbacks" / "ar" / "demo" / "logs"
            _write_tfevents(logs_dir / "20260725-run")
            with mock.patch.object(tb, "REPO", root):
                resolved = tb._resolve_group_path("callbacks/ar/demo")
            self.assertEqual(resolved, logs_dir.resolve())


class SelectGroupsTest(unittest.TestCase):
    def _seed_ladder_tree(self, root: pathlib.Path) -> None:
        for name in ("ladder_10t_50v", "ladder_50t_50v", "smoke_50t_50v"):
            logs_dir = root / "callbacks" / "ar" / name / "logs"
            _write_tfevents(logs_dir / "20260725-run")

    def test_preset_ladder_filters_discovered_groups(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            self._seed_ladder_tree(root)
            args = argparse.Namespace(
                config=[],
                group=[],
                preset="ladder",
                filter="",
                scan_root=[str(root / "callbacks")],
                list=True,
                root=[],
            )
            with mock.patch.object(tb, "REPO", root):
                groups = tb._select_groups(args)
            labels = {label for label, _logs, _count in groups}
            self.assertEqual(labels, {"ar/ladder_10t_50v", "ar/ladder_50t_50v"})

    def test_filter_regex_narrows_discovery(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            self._seed_ladder_tree(root)
            args = argparse.Namespace(
                config=[],
                group=[],
                preset="",
                filter="smoke_50t",
                scan_root=[str(root / "callbacks")],
                list=True,
                root=[],
            )
            with mock.patch.object(tb, "REPO", root):
                groups = tb._select_groups(args)
            self.assertEqual(len(groups), 1)
            self.assertEqual(groups[0][0], "ar/smoke_50t_50v")


class BuildTensorboardArgvTest(unittest.TestCase):
    def test_single_group_uses_logdir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logs_dir = pathlib.Path(tmp) / "logs"
            _write_tfevents(logs_dir / "run-a")
            args = argparse.Namespace(
                root=[],
                port=6006,
                reload_interval=30,
            )
            groups = [("ar/demo", logs_dir, 1)]
            fake_tb = pathlib.Path(tmp) / "tensorboard.exe"
            fake_tb.write_bytes(b"")
            with mock.patch.object(tb, "_tensorboard_executable", return_value=fake_tb):
                argv = tb._build_tensorboard_argv(args, groups)
            self.assertEqual(argv[0], str(fake_tb))
            self.assertIn("--logdir", argv)
            self.assertEqual(argv[argv.index("--logdir") + 1], logs_dir.as_posix())

    def test_multiple_groups_use_logdir_spec(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            logs_a = root / "a" / "logs"
            logs_b = root / "b" / "logs"
            _write_tfevents(logs_a / "run-a")
            _write_tfevents(logs_b / "run-b")
            args = argparse.Namespace(
                root=[],
                port=6006,
                reload_interval=30,
            )
            groups = [
                ("group/a", logs_a, 1),
                ("group/b", logs_b, 1),
            ]
            fake_tb = root / "tensorboard.exe"
            fake_tb.write_bytes(b"")
            with mock.patch.object(tb, "_tensorboard_executable", return_value=fake_tb):
                argv = tb._build_tensorboard_argv(args, groups)
            spec = argv[argv.index("--logdir_spec") + 1]
            self.assertIn("group/a:", spec)
            self.assertIn("group/b:", spec)

    def test_root_mode_uses_recursive_logdir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            args = argparse.Namespace(
                root=["callbacks/ar"],
                port=6006,
                reload_interval=30,
            )
            fake_tb = root / "tensorboard.exe"
            fake_tb.write_bytes(b"")
            with (
                mock.patch.object(tb, "REPO", root),
                mock.patch.object(
                    tb,
                    "_tensorboard_executable",
                    return_value=fake_tb,
                ),
            ):
                argv = tb._build_tensorboard_argv(args, [])
            self.assertEqual(
                argv[argv.index("--logdir") + 1],
                (root / "callbacks" / "ar").resolve().as_posix(),
            )

    def test_duplicate_labels_get_numeric_suffix(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            logs_a = root / "a" / "logs"
            logs_b = root / "b" / "logs"
            _write_tfevents(logs_a / "run-a")
            _write_tfevents(logs_b / "run-b")
            args = argparse.Namespace(
                root=[],
                port=6006,
                reload_interval=30,
            )
            groups = [
                ("same", logs_a, 1),
                ("same", logs_b, 1),
            ]
            fake_tb = root / "tensorboard.exe"
            fake_tb.write_bytes(b"")
            with mock.patch.object(tb, "_tensorboard_executable", return_value=fake_tb):
                argv = tb._build_tensorboard_argv(args, groups)
            spec = argv[argv.index("--logdir_spec") + 1]
            self.assertIn("same:", spec)
            self.assertIn("same_2:", spec)


class MainCliTest(unittest.TestCase):
    def test_main_list_prints_discovered_groups(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            logs_dir = root / "callbacks" / "ar" / "ladder_10t_50v" / "logs"
            _write_tfevents(logs_dir / "20260725-run")
            with (
                mock.patch.object(tb, "REPO", root),
                redirect_stdout(io.StringIO()) as out,
            ):
                code = tb.main(
                    [
                        "--list",
                        "--preset",
                        "ladder",
                        "--scan-root",
                        str(root / "callbacks"),
                    ],
                )
            self.assertEqual(code, 0)
            self.assertIn("ar/ladder_10t_50v", out.getvalue())

    def test_main_requires_scope_without_list(self) -> None:
        with self.assertRaises(SystemExit):
            tb.main([])

    def test_main_print_cmd_does_not_launch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            logs_dir = root / "callbacks" / "ar" / "demo" / "logs"
            _write_tfevents(logs_dir / "20260725-run")
            fake_tb = root / "tensorboard.exe"
            fake_tb.write_bytes(b"")
            with (
                mock.patch.object(tb, "REPO", root),
                mock.patch.object(
                    tb,
                    "_tensorboard_executable",
                    return_value=fake_tb,
                ),
                mock.patch.object(tb.subprocess, "call") as call_mock,
            ):
                code = tb.main(
                    [
                        "--root",
                        "callbacks/ar",
                        "--print-cmd",
                    ],
                )
            call_mock.assert_not_called()
            self.assertEqual(code, 0)


if __name__ == "__main__":
    unittest.main()
