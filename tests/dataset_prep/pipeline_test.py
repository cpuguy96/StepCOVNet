"""Tests for dataset_prep export, validate, and pipeline dry-run."""

import dataclasses
import json
import pathlib
import tempfile
import unittest

from stepcovnet.dataset_prep import (
    config,
    constants,
    discovery,
    export,
    models,
    normalize,
    pack_results,
    pipeline,
    validate,
)


def _minimal_pack() -> models.ParsedSongPack:
    metadata = models.SimfileMetadata(
        title="Test Song",
        artist="Artist",
        subtitle="",
        music_filename="song.ogg",
        offset_sec=0.0,
        initial_bpm=120.0,
        bpm_segments=[models.BpmSegment(start_beat=0.0, bpm=120.0)],
        selectable=True,
    )
    summary = models.ChartSummary(
        stepstype="dance-single",
        difficulty="hard",
        difficulty_kind=constants.DIFFICULTY_KIND_STANDARD,
        meter=8,
        chart_name="",
        credit="",
        num_steps=1,
    )
    chart = models.ParsedChart(
        summary=summary,
        times_sec=[1.0],
        arrow_rows=["0001"],
        column_codes=[1],
    )
    return models.ParsedSongPack(
        schema_version=constants.SCHEMA_VERSION,
        normalized_bundle="bundle",
        normalized_id="test_song",
        source_pack_relpath="bundle/pack",
        source_simfile="chart.ssc",
        metadata=metadata,
        charts=[chart],
        default_chart_index=0,
        available_charts=[],
        audio_filename="song.ogg",
        audio_source=constants.AUDIO_SOURCE_MUSIC_TAG,
        audio_resolved_relpath="song.ogg",
        warnings=[],
    )


class ValidateExportTest(unittest.TestCase):
    def test_validate_parsed_pack_detects_non_monotonic_times(self):
        pack = _minimal_pack()
        pack.charts[0].times_sec = [2.0, 1.0]
        self.assertIn(
            "chart_0_non_monotonic_times", validate.validate_parsed_pack(pack)
        )

    def test_render_legacy_txt_matches_parser_layout(self):
        pack = _minimal_pack()
        text = export.render_legacy_txt(pack)
        self.assertIn("TITLE Test Song", text)
        self.assertIn("BPM 120.0", text)
        self.assertIn("DIFFICULTY Hard", text)
        self.assertIn("0001 1.0", text)

    def test_output_audio_filename_uses_normalized_id(self):
        self.assertEqual(
            export.output_audio_filename("6894_12_expanded", "Expanded.ogg"),
            "6894_12_expanded.ogg",
        )

    def test_write_song_pack_creates_chart_json_and_audio(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = pathlib.Path(tmpdir)
            raw_pack = root / "raw" / "pack"
            raw_pack.mkdir(parents=True)
            (raw_pack / "song.ogg").write_bytes(b"audio")
            out_dir = root / "final_data"
            pack = _minimal_pack()
            export.write_song_pack(
                pack,
                raw_pack_dir=raw_pack,
                output_dir=out_dir,
                prep_config=config.PrepConfig(),
            )
            song_dir = out_dir / "bundle" / "test_song"
            self.assertTrue((song_dir / "test_song.chart.json").is_file())
            self.assertTrue((song_dir / "test_song.ogg").is_file())


class PipelineDryRunTest(unittest.TestCase):
    def test_entry_needs_processing_respects_overwrite(self):
        exported = normalize.NameMapEntry(
            normalized_bundle="bundle",
            normalized_id="song",
            output_relpath="bundle/song",
            source_bundle="Bundle",
            source_pack="Bundle/song",
            source_simfile="sm.ssc",
            title="Song",
            artist="",
            audio_source="music_tag",
            result=pack_results.PACK_RESULT_EXPORTED,
            reason=None,
            warnings=[],
        )
        pending = dataclasses.replace(
            exported,
            result=pack_results.PACK_RESULT_PENDING,
        )
        skipped = dataclasses.replace(
            exported,
            result=pack_results.PACK_RESULT_SKIPPED,
            reason=pack_results.REASON_NO_DANCE_SINGLE,
        )
        self.assertTrue(pipeline.entry_needs_processing(pending, overwrite=False))
        self.assertFalse(pipeline.entry_needs_processing(exported, overwrite=False))
        self.assertTrue(pipeline.entry_needs_processing(exported, overwrite=True))
        self.assertFalse(pipeline.entry_needs_processing(skipped, overwrite=True))

    def test_dry_run_writes_manifests_without_pack_output(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = pathlib.Path(tmpdir)
            bundle = root / "ITL Online 2026" / "[01] Song"
            bundle.mkdir(parents=True)
            sim = (
                "#TITLE:Song One;\n#ARTIST:A;\n#MUSIC:song.ogg;\n#OFFSET:0.0;\n"
                "#BPMS:0.0=120.0;\n#SELECTABLE:YES;\n#NOTEDATA:;\n#STEPSTYPE:dance-single;\n"
                "#DIFFICULTY:Challenge;\n#METER:10;\n#NOTES:\n0001\n0000\n0000\n0000\n,\n;\n"
            )
            (bundle / "sm.ssc").write_text(sim, encoding="utf-8")
            (bundle / "song.ogg").write_bytes(b"audio")
            out_dir = root / "final_data"
            prep = config.PrepConfig(
                input_dir=str(root / "ITL Online 2026"),
                output_dir=str(out_dir),
                dry_run=True,
                limit_packs=1,
            )
            report = pipeline.run_preprocess(prep)
            self.assertTrue(discovery.packs_manifest_path(out_dir).is_file())
            self.assertTrue(normalize.name_map_path(out_dir).is_file())
            self.assertTrue(pipeline.preprocess_report_path(out_dir).is_file())
            self.assertTrue(report.dry_run)
            self.assertEqual(report.counts["packs_scheduled"], 1)
            nm_path = normalize.name_map_path(out_dir)
            with nm_path.open(encoding="utf-8") as handle:
                name_map = json.load(handle)
            self.assertEqual(
                name_map["entries"][0]["result"], pack_results.PACK_RESULT_PENDING
            )
            self.assertIsNone(name_map["entries"][0]["reason"])
            self.assertFalse((out_dir / "itl_online_2026").exists())


if __name__ == "__main__":
    unittest.main()
