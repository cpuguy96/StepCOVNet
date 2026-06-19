"""Tests for dataset_prep.normalize."""

import pathlib
import tempfile
import unittest

from stepcovnet.dataset_prep import (
    config,
    discovery,
    normalize,
    pack_results,
)


def _touch(path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")


class SlugifyTest(unittest.TestCase):
    def test_slugify_folds_unicode_and_punctuation(self):
        self.assertEqual(normalize.slugify("ITL Online 2026"), "itl_online_2026")
        self.assertEqual(normalize.slugify("Expanded!!"), "expanded")

    def test_finalize_slug_rewrites_reserved_name(self):
        slug, warnings = normalize.finalize_slug("nul")
        self.assertEqual(slug, "nul_dir")
        self.assertIn("reserved_slug_rewritten", warnings)

    def test_assign_unique_slug_appends_suffix(self):
        used: set[str] = set()
        first = normalize.assign_unique_slug("expanded", used)
        second = normalize.assign_unique_slug("expanded", used)
        self.assertEqual(first, "expanded")
        self.assertEqual(second, "expanded_2")


class BuildNameMapTest(unittest.TestCase):
    def _write_pack(
        self, root: pathlib.Path, bundle: str, pack_name: str, title: str
    ) -> None:
        pack = root / bundle / pack_name
        pack.mkdir(parents=True, exist_ok=True)
        sim = (
            "#TITLE:"
            + title
            + ";\n#ARTIST:Artist;\n#MUSIC:song.ogg;\n#OFFSET:0.0;\n#BPMS:0.0=120.0;\n"
            "#SELECTABLE:YES;\n#NOTEDATA:;\n#STEPSTYPE:dance-single;\n#DIFFICULTY:Hard;\n"
            "#METER:8;\n#NOTES:\n0001\n0000\n0000\n0000\n,\n;\n"
        )
        (pack / "chart.ssc").write_text(sim, encoding="utf-8")
        _touch(pack / "song.ogg")

    def test_build_name_map_assigns_bundle_and_song_slugs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = pathlib.Path(tmpdir)
            self._write_pack(root, "ITL Online 2026", "[12] Expanded", "Expanded!!")
            self._write_pack(root, "ITL Online 2026", "[07] Nightmare", "Nightmare")
            manifest = discovery.build_packs_manifest(root / "ITL Online 2026")
            prep = config.PrepConfig(output_dir=str(root / "out"))
            name_map = normalize.build_name_map(manifest, prep)
        self.assertEqual(len(name_map.entries), 2)
        bundles = {entry.normalized_bundle for entry in name_map.entries}
        self.assertEqual(bundles, {"itl_online_2026"})
        ids = {entry.normalized_id for entry in name_map.entries}
        self.assertIn("expanded", ids)
        self.assertIn("nightmare", ids)
        for entry in name_map.entries:
            self.assertTrue(entry.source_pack.startswith("ITL Online 2026/"))

    def test_build_name_map_resolves_title_collisions_within_bundle(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = pathlib.Path(tmpdir)
            self._write_pack(root, "Bundle", "pack_a", "Same Title")
            self._write_pack(root, "Bundle", "pack_b", "Same Title")
            manifest = discovery.build_packs_manifest(root / "Bundle")
            prep = config.PrepConfig(output_dir=str(root / "out"))
            name_map = normalize.build_name_map(manifest, prep)
        ids = sorted(entry.normalized_id for entry in name_map.entries)
        self.assertEqual(ids, ["same_title", "same_title_2"])

    def test_build_name_map_limit_truncates_pack_list(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = pathlib.Path(tmpdir)
            self._write_pack(root, "Bundle", "aaa", "AAA")
            self._write_pack(root, "Bundle", "zzz", "ZZZ")
            manifest = discovery.build_packs_manifest(root / "Bundle")
            prep = config.PrepConfig(output_dir=str(root / "out"))
            name_map = normalize.build_name_map(manifest, prep, limit=1)
        self.assertEqual(len(name_map.entries), 1)
        self.assertTrue(name_map.entries[0].source_pack.endswith("/aaa"))

    def test_run_normalization_writes_name_map_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = pathlib.Path(tmpdir)
            self._write_pack(root, "Bundle", "song", "Song")
            input_dir = root / "Bundle"
            out_dir = root / "final_data"
            prep = config.PrepConfig(input_dir=str(input_dir), output_dir=str(out_dir))
            manifest = discovery.run_discovery(prep)
            name_map = normalize.run_normalization(manifest, prep)
            self.assertTrue(normalize.name_map_path(out_dir).is_file())
            self.assertTrue(normalize.name_map_csv_path(out_dir).is_file())
            self.assertEqual(
                name_map.entries[0].result, pack_results.PACK_RESULT_PENDING
            )
            self.assertIsNone(name_map.entries[0].reason)


if __name__ == "__main__":
    unittest.main()
