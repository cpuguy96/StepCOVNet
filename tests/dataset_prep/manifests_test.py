"""Tests for dataset_prep.manifests merge helpers."""

import pathlib
import tempfile
import unittest

from stepcovnet.dataset_prep import (
    config,
    constants,
    discovery,
    manifests,
    normalize,
    pack_results,
)


def _touch(path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")


class ManifestMergeTest(unittest.TestCase):
    def test_manifest_raw_input_root_single_bundle_uses_parent(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = pathlib.Path(tmpdir)
            bundle = root / "raw" / "ITL Online 2026" / "song"
            _touch(bundle / "chart.ssc")
            manifest = discovery.build_packs_manifest(root / "raw" / "ITL Online 2026")
        self.assertEqual(
            str(manifests.manifest_raw_input_root(manifest)),
            str((root / "raw").resolve()),
        )

    def test_merge_name_maps_keeps_entries_from_both_runs(self):
        existing = normalize.NameMap(
            schema_version=constants.SCHEMA_VERSION,
            raw_input_root="/data/raw",
            output_dir="/data/final",
            entries=[
                normalize.NameMapEntry(
                    normalized_bundle="itl_online_2026",
                    normalized_id="song_a",
                    output_relpath="itl_online_2026/song_a",
                    source_bundle="ITL Online 2026",
                    source_pack="ITL Online 2026/[01] Song A",
                    source_simfile="chart.ssc",
                    title="Song A",
                    artist="",
                    audio_source="",
                    result=pack_results.PACK_RESULT_EXPORTED,
                    reason=None,
                    warnings=[],
                    charts_exported=1,
                )
            ],
        )
        new = normalize.NameMap(
            schema_version=constants.SCHEMA_VERSION,
            raw_input_root="/data/raw",
            output_dir="/data/final",
            entries=[
                normalize.NameMapEntry(
                    normalized_bundle="mizuki_s_simfiles",
                    normalized_id="song_b",
                    output_relpath="mizuki_s_simfiles/song_b",
                    source_bundle="Mizuki S Simfiles",
                    source_pack="Mizuki S Simfiles/[01] Song B",
                    source_simfile="chart.ssc",
                    title="Song B",
                    artist="",
                    audio_source="",
                    result=pack_results.PACK_RESULT_EXPORTED,
                    reason=None,
                    warnings=[],
                    charts_exported=2,
                )
            ],
        )
        merged = manifests.merge_name_maps(existing, new)
        self.assertEqual(len(merged.entries), 2)
        self.assertEqual(
            {entry.source_pack for entry in merged.entries},
            {
                "ITL Online 2026/[01] Song A",
                "Mizuki S Simfiles/[01] Song B",
            },
        )

    def test_build_preprocess_report_aggregates_merged_name_map(self):
        name_map = normalize.NameMap(
            schema_version=constants.SCHEMA_VERSION,
            raw_input_root="/data/raw",
            output_dir="/data/final",
            entries=[
                normalize.NameMapEntry(
                    normalized_bundle="itl_online_2026",
                    normalized_id="exported",
                    output_relpath="itl_online_2026/exported",
                    source_bundle="ITL Online 2026",
                    source_pack="ITL Online 2026/exported",
                    source_simfile="chart.ssc",
                    title="Exported",
                    artist="",
                    audio_source="",
                    result=pack_results.PACK_RESULT_EXPORTED,
                    reason=None,
                    warnings=[],
                    charts_exported=1,
                ),
                normalize.NameMapEntry(
                    normalized_bundle="itl_online_2026",
                    normalized_id="skipped",
                    output_relpath="itl_online_2026/skipped",
                    source_bundle="ITL Online 2026",
                    source_pack="ITL Online 2026/skipped",
                    source_simfile="chart.ssc",
                    title="Skipped",
                    artist="",
                    audio_source="",
                    result=pack_results.PACK_RESULT_SKIPPED,
                    reason=pack_results.REASON_NO_DANCE_SINGLE,
                    warnings=[],
                ),
            ],
        )
        report = manifests.build_preprocess_report(
            name_map,
            packs_manifest=None,
            chart_skips=[],
            started_at="t0",
            finished_at="t1",
            dry_run=False,
        )
        self.assertEqual(report["counts"]["packs_exported"], 1)
        self.assertEqual(report["counts"]["packs_skipped"], 1)
        self.assertEqual(report["counts"]["charts_exported"], 1)

    def test_supplement_name_map_adds_skipped_worker_results(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = pathlib.Path(tmpdir)
            pack_dir = root / "raw" / "ITL Online 2026" / "[07] Delusion"
            _touch(pack_dir / "delusion.ssc")
            manifest = discovery.build_packs_manifest(root / "raw" / "ITL Online 2026")
            name_map = normalize.NameMap(
                schema_version=constants.SCHEMA_VERSION,
                raw_input_root=str((root / "raw").resolve()),
                output_dir=str((root / "out").resolve()),
                entries=[],
            )
            worker_results = [
                {
                    "normalized_bundle": "itl_online_2026",
                    "normalized_id": "1870_07_delusion_easy",
                    "output_relpath": "itl_online_2026/1870_07_delusion_easy",
                    "source_pack": "[07] Delusion",
                    "result": pack_results.PACK_RESULT_SKIPPED,
                    "reason": pack_results.REASON_NO_DANCE_SINGLE,
                    "warnings": [],
                    "charts_exported": 0,
                    "charts_skipped": 0,
                    "chart_skips": [],
                    "message": "",
                }
            ]
            supplemented = manifests.supplement_name_map_from_worker_results(
                name_map,
                worker_results,
                manifest,
            )
            report = manifests.build_preprocess_report(
                supplemented,
                packs_manifest=manifest,
                chart_skips=[],
                started_at="t0",
                finished_at="t1",
                dry_run=False,
            )
        self.assertEqual(len(supplemented.entries), 1)
        self.assertEqual(
            supplemented.entries[0].result, pack_results.PACK_RESULT_SKIPPED
        )
        self.assertEqual(report["counts"]["packs_skipped"], 1)
        self.assertEqual(len(report["skipped_packs"]), 1)
        self.assertEqual(
            report["skipped_packs"][0]["reason"],
            pack_results.REASON_NO_DANCE_SINGLE,
        )

    def test_canonicalize_source_pack_adds_bundle_prefix(self):
        entry = normalize.NameMapEntry(
            normalized_bundle="itl_online_2026",
            normalized_id="song",
            output_relpath="itl_online_2026/song",
            source_bundle="ITL Online 2026",
            source_pack="[01] Song",
            source_simfile="chart.ssc",
            title="Song",
            artist="",
            audio_source="",
            result=pack_results.PACK_RESULT_EXPORTED,
            reason=None,
            warnings=[],
        )
        canonical = manifests.canonicalize_source_pack(entry)
        self.assertEqual(canonical.source_pack, "ITL Online 2026/[01] Song")

    def test_run_normalization_merge_preserves_previous_bundle_rows(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = pathlib.Path(tmpdir)
            bundle_a = root / "raw" / "Bundle A" / "song_a"
            bundle_b = root / "raw" / "Bundle B" / "song_b"
            _touch(bundle_a / "a.ssc")
            _touch(bundle_b / "b.ssc")
            out_dir = root / "final_data"
            prep_a = config.PrepConfig(
                input_dir=str(root / "raw" / "Bundle A"),
                output_dir=str(out_dir),
            )
            prep_b = config.PrepConfig(
                input_dir=str(root / "raw" / "Bundle B"),
                output_dir=str(out_dir),
            )
            manifest_a = discovery.run_discovery(prep_a)
            name_map_a = normalize.run_normalization(manifest_a, prep_a)
            name_map_a.entries[0].result = pack_results.PACK_RESULT_EXPORTED
            normalize.save_name_map(name_map_a, out_dir, merge=False)

            manifest_b = discovery.run_discovery(prep_b)
            merged = normalize.run_normalization(manifest_b, prep_b)

        self.assertEqual(len(merged.entries), 2)
        self.assertEqual(
            {entry.source_pack for entry in merged.entries},
            {"Bundle A/song_a", "Bundle B/song_b"},
        )
        exported = next(
            entry for entry in merged.entries if entry.source_pack == "Bundle A/song_a"
        )
        self.assertEqual(exported.result, pack_results.PACK_RESULT_EXPORTED)

    def test_merge_name_maps_preserves_terminal_existing_entries(self):
        existing = normalize.NameMap(
            schema_version=constants.SCHEMA_VERSION,
            raw_input_root="/data/raw",
            output_dir="/data/final",
            entries=[
                normalize.NameMapEntry(
                    normalized_bundle="itl_online_2026",
                    normalized_id="song_a",
                    output_relpath="itl_online_2026/song_a",
                    source_bundle="ITL Online 2026",
                    source_pack="ITL Online 2026/[01] Song A",
                    source_simfile="chart.ssc",
                    title="Song A",
                    artist="",
                    audio_source="",
                    result=pack_results.PACK_RESULT_EXPORTED,
                    reason=None,
                    warnings=[],
                    charts_exported=1,
                )
            ],
        )
        new = normalize.NameMap(
            schema_version=constants.SCHEMA_VERSION,
            raw_input_root="/data/raw",
            output_dir="/data/final",
            entries=[
                normalize.NameMapEntry(
                    normalized_bundle="itl_online_2026",
                    normalized_id="song_a",
                    output_relpath="itl_online_2026/song_a",
                    source_bundle="ITL Online 2026",
                    source_pack="ITL Online 2026/[01] Song A",
                    source_simfile="chart.ssc",
                    title="Song A",
                    artist="",
                    audio_source="",
                    result=pack_results.PACK_RESULT_PENDING,
                    reason=None,
                    warnings=[],
                )
            ],
        )
        merged = manifests.merge_name_maps(existing, new)
        self.assertEqual(len(merged.entries), 1)
        self.assertEqual(merged.entries[0].result, pack_results.PACK_RESULT_EXPORTED)


if __name__ == "__main__":
    unittest.main()
