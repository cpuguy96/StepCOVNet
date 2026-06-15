"""Tests for dataset_prep.discovery."""

import json
import os
import pathlib
import tempfile
import unittest

from stepcovnet.dataset_prep import config, constants, discovery


def _touch(path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")


class DiscoveryHelpersTest(unittest.TestCase):
    def test_choose_simfile_prefers_ssc_over_sm(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pack = pathlib.Path(tmpdir) / "pack"
            pack.mkdir()
            _touch(pack / "chart.sm")
            _touch(pack / "chart.ssc")
            self.assertEqual(discovery.choose_simfile(pack), "chart.ssc")

    def test_is_pack_dir_false_for_empty_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pack = pathlib.Path(tmpdir) / "pack"
            pack.mkdir()
            self.assertFalse(discovery.is_pack_dir(pack))

    def test_choose_simfile_raises_when_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pack = pathlib.Path(tmpdir) / "pack"
            pack.mkdir()
            with self.assertRaises(FileNotFoundError):
                discovery.choose_simfile(pack)

    def test_list_pack_dirs_returns_empty_for_missing_bundle(self):
        self.assertEqual(discovery.list_pack_dirs(pathlib.Path("nonexistent")), [])

    def test_is_pack_dir_false_for_file_path(self):
        with tempfile.NamedTemporaryFile() as handle:
            self.assertFalse(discovery.is_pack_dir(pathlib.Path(handle.name)))


class BuildPacksManifestTest(unittest.TestCase):
    def _write_single_bundle_tree(self, root: pathlib.Path) -> None:
        bundle = root / "ITL Online 2026"
        pack_a = bundle / "[12] Expanded"
        pack_b = bundle / "[07] Nightmare"
        _touch(pack_a / "sm.ssc")
        _touch(pack_a / "song.ogg")
        _touch(pack_b / "only.sm")

    def _write_multi_bundle_tree(self, root: pathlib.Path) -> None:
        bundle_a = root / "bundle_a"
        bundle_b = root / "bundle_b"
        empty = root / "empty_bundle"
        empty.mkdir(parents=True)
        (empty / "readme.txt").write_text("no simfiles", encoding="utf-8")
        _touch(bundle_a / "song_one" / "a.ssc")
        _touch(bundle_b / "song_two" / "b.sm")

    def test_single_bundle_discovery(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = pathlib.Path(tmpdir)
            self._write_single_bundle_tree(root / "input")
            input_dir = root / "input" / "ITL Online 2026"
            manifest = discovery.build_packs_manifest(input_dir)
        self.assertEqual(manifest.discovery_mode, discovery.DiscoveryMode.SINGLE_BUNDLE)
        self.assertEqual(len(manifest.bundles), 1)
        self.assertEqual(manifest.bundles[0].pack_count, 2)
        self.assertEqual(len(manifest.packs), 2)
        relpaths = {entry.pack_relpath for entry in manifest.packs}
        self.assertIn("[12] Expanded", relpaths)
        expanded = next(
            entry for entry in manifest.packs if entry.pack_relpath.endswith("Expanded")
        )
        self.assertEqual(expanded.simfile, "sm.ssc")
        self.assertEqual(expanded.source_bundle, "ITL Online 2026")
        self.assertEqual(expanded.bundle_relpath, "ITL Online 2026")

    def test_multi_bundle_discovery_with_empty_bundle_warning(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = pathlib.Path(tmpdir) / "raw_data"
            self._write_multi_bundle_tree(root)
            manifest = discovery.build_packs_manifest(root)
        self.assertEqual(manifest.discovery_mode, discovery.DiscoveryMode.MULTI_BUNDLE)
        self.assertEqual(len(manifest.bundles), 2)
        self.assertEqual(len(manifest.packs), 2)
        self.assertTrue(any(w.startswith("empty_bundle:") for w in manifest.warnings))
        self.assertEqual(
            {entry.source_bundle for entry in manifest.packs},
            {"bundle_a", "bundle_b"},
        )

    def test_build_packs_manifest_missing_input_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            missing = os.path.join(tmpdir, "missing")
        with self.assertRaises(FileNotFoundError):
            discovery.build_packs_manifest(missing)


class PacksManifestIoTest(unittest.TestCase):
    def test_save_load_and_run_discovery_round_trip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = pathlib.Path(tmpdir)
            input_dir = root / "input" / "ITL Online 2026"
            pack = input_dir / "song"
            _touch(pack / "sm.ssc")
            output_dir = root / "output"
            cfg = config.PrepConfig(
                input_dir=str(input_dir),
                output_dir=str(output_dir),
            )
            written = discovery.run_discovery(cfg)
            path = discovery.packs_manifest_path(output_dir)
            self.assertTrue(path.is_file())
            loaded = discovery.load_packs_manifest(path)
        self.assertEqual(loaded.schema_version, constants.SCHEMA_VERSION)
        self.assertEqual(len(loaded.packs), len(written.packs))
        self.assertEqual(loaded.packs[0].simfile, "sm.ssc")

    def test_multi_bundle_skips_non_directory_children(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = pathlib.Path(tmpdir)
            (root / "stray.txt").write_text("x", encoding="utf-8")
            bundle = root / "bundle_a"
            _touch(bundle / "song" / "a.ssc")
            manifest = discovery.build_packs_manifest(root)
        self.assertEqual(len(manifest.packs), 1)

    def test_load_packs_manifest_rejects_missing_schema_version(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = pathlib.Path(tmpdir) / "packs_manifest.json"
            with open(path, "w", encoding="utf-8") as handle:
                json.dump({"packs": []}, handle)
            with self.assertRaises(ValueError):
                discovery.load_packs_manifest(path)

    def test_load_packs_manifest_rejects_unsupported_schema_version(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = pathlib.Path(tmpdir) / "packs_manifest.json"
            with open(path, "w", encoding="utf-8") as handle:
                json.dump({"schema_version": 99, "packs": []}, handle)
            with self.assertRaises(ValueError):
                discovery.load_packs_manifest(path)


if __name__ == "__main__":
    unittest.main()
