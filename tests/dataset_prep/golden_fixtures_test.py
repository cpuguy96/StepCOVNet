"""Golden fixture integration tests for dataset_prep (P6)."""

from __future__ import annotations

import contextlib
import dataclasses
import pathlib
import tempfile
import unittest

import numpy as np

from stepcovnet import datasets
from stepcovnet.dataset_prep import (
    config,
    constants,
    export,
    models,
    normalize,
    pipeline,
)

_FIXTURES_ROOT = (
    pathlib.Path(__file__).resolve().parent.parent / "fixtures" / "dataset_prep"
)


def _legacy_txt_for_default_chart(pack: models.ParsedSongPack) -> str:
    """Render legacy v2 text for the default chart only."""
    chart = pack.charts[pack.default_chart_index]
    single = dataclasses.replace(pack, charts=[chart])
    return export.render_legacy_txt(single)


def _assert_default_chart_round_trips(pack: models.ParsedSongPack) -> None:
    chart = pack.charts[pack.default_chart_index]
    with tempfile.TemporaryDirectory() as tmpdir:
        txt_path = pathlib.Path(tmpdir) / f"{pack.normalized_id}.txt"
        txt_path.write_text(_legacy_txt_for_default_chart(pack), encoding="utf-8")
        times, cols = datasets._parse_step_chart(str(txt_path), binary_timings=False)
    np.testing.assert_allclose(times, np.asarray(chart.times_sec, dtype=float))
    np.testing.assert_array_equal(
        cols,
        np.asarray(chart.column_codes, dtype=np.int32),
    )


class GoldenFixturesTest(unittest.TestCase):
    @contextlib.contextmanager
    def _fixture_output(
        self,
        bundle_dir: str,
        *,
        export_legacy_txt: bool = False,
    ):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = pathlib.Path(tmpdir) / "out"
            prep = config.PrepConfig(
                input_dir=str(_FIXTURES_ROOT / bundle_dir),
                output_dir=str(out_dir),
                export_legacy_txt=export_legacy_txt,
                overwrite=True,
            )
            report = pipeline.run_preprocess(prep)
            yield out_dir, report

    def test_itl_challenge_ssc_exports_ssc_pack(self):
        with self._fixture_output("itl_challenge_ssc") as (out_dir, report):
            self.assertEqual(report.counts["packs_exported"], 1)
            pack = models.load_parsed_song(out_dir, "itl_challenge_ssc", "golden_song")
            self.assertEqual(pack.source_simfile, "chart.ssc")
            self.assertEqual(len(pack.charts), 1)
            self.assertEqual(pack.charts[0].summary.difficulty.lower(), "challenge")
            self.assertEqual(pack.default_chart_index, 0)
            self.assertAlmostEqual(pack.metadata.offset_sec, 0.5)
            self.assertEqual(pack.charts[0].arrow_rows, ["0001"])
            self.assertEqual(pack.charts[0].column_codes, [1])
            audio_path = (
                out_dir / "itl_challenge_ssc" / "golden_song" / "golden_song.ogg"
            )
            self.assertTrue(audio_path.is_file())

    def test_vocaloid_multi_sm_exports_all_singles(self):
        with self._fixture_output("vocaloid_multi_sm") as (out_dir, report):
            self.assertEqual(report.counts["packs_exported"], 1)
            self.assertEqual(report.counts["charts_exported"], 2)
            pack = models.load_parsed_song(out_dir, "vocaloid_multi_sm", "multi_diff")
            self.assertEqual(pack.source_simfile, "chart.sm")
            self.assertEqual(len(pack.charts), 2)
            difficulties = [chart.summary.difficulty.lower() for chart in pack.charts]
            self.assertEqual(difficulties, ["beginner", "challenge"])
            self.assertEqual(pack.default_chart_index, 1)
            self.assertEqual(pack.charts[1].arrow_rows, ["0100"])
            self.assertEqual(pack.charts[1].column_codes, [16])

    def test_edge_nul_inferred_audio_and_reserved_slug(self):
        with self._fixture_output("edge_nul_inferred") as (out_dir, report):
            self.assertEqual(report.counts["packs_exported"], 1)
            pack = models.load_parsed_song(out_dir, "edge_nul_inferred", "nul_dir")
            self.assertEqual(pack.normalized_id, "nul_dir")
            self.assertEqual(pack.audio_source, constants.AUDIO_SOURCE_INFERRED)
            self.assertEqual(pack.audio_resolved_relpath, "only_audio.ogg")
            self.assertEqual(pack.audio_filename, "nul_dir.ogg")
            name_map = normalize.load_name_map(normalize.name_map_path(out_dir))
            entry = name_map.entries[0]
            self.assertIn("reserved_slug_rewritten", entry.warnings)

    def _assert_fixture_round_trips(
        self,
        bundle_dir: str,
        normalized_bundle: str,
        normalized_id: str,
    ) -> None:
        with self._fixture_output(bundle_dir) as (out_dir, report):
            self.assertEqual(report.counts["packs_exported"], 1)
            pack = models.load_parsed_song(
                out_dir,
                normalized_bundle,
                normalized_id,
            )
            _assert_default_chart_round_trips(pack)

    def test_default_chart_legacy_txt_round_trips_all_fixtures(self):
        cases = (
            ("itl_challenge_ssc", "itl_challenge_ssc", "golden_song"),
            ("vocaloid_multi_sm", "vocaloid_multi_sm", "multi_diff"),
            ("edge_nul_inferred", "edge_nul_inferred", "nul_dir"),
        )
        for bundle_dir, normalized_bundle, normalized_id in cases:
            with self.subTest(fixture=bundle_dir):
                self._assert_fixture_round_trips(
                    bundle_dir,
                    normalized_bundle,
                    normalized_id,
                )

    def test_export_legacy_txt_writes_round_trip_file(self):
        with self._fixture_output(
            "itl_challenge_ssc",
            export_legacy_txt=True,
        ) as (out_dir, report):
            self.assertEqual(report.counts["packs_exported"], 1)
            pack = models.load_parsed_song(out_dir, "itl_challenge_ssc", "golden_song")
            song_dir = out_dir / "itl_challenge_ssc" / "golden_song"
            txt_path = song_dir / "golden_song.txt"
            self.assertTrue(txt_path.is_file())
            times, cols = datasets._parse_step_chart(
                str(txt_path), binary_timings=False
            )
            default = pack.charts[pack.default_chart_index]
            np.testing.assert_allclose(
                times,
                np.asarray(default.times_sec, dtype=float),
            )
            np.testing.assert_array_equal(
                cols,
                np.asarray(default.column_codes, dtype=np.int32),
            )


if __name__ == "__main__":
    unittest.main()
