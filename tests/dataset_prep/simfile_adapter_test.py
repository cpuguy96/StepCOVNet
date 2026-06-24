"""Tests for dataset_prep.simfile_adapter."""

import pathlib
import tempfile
import unittest
import unittest.mock

from stepcovnet.dataset_prep import (
    config,
    constants,
    models,
    pack_results,
    simfile_adapter,
)


def _write_pack(
    root: pathlib.Path,
    *,
    sim_text: str,
    audio_name: str = "song.ogg",
) -> pathlib.Path:
    pack = root / "pack"
    pack.mkdir(parents=True, exist_ok=True)
    (pack / "chart.ssc").write_text(sim_text, encoding="utf-8")
    (pack / audio_name).write_bytes(b"audio")
    return pack


_BASE_SSC = """#VERSION:0.83;
#TITLE:Test Song;
#ARTIST:Artist;
#MUSIC:song.ogg;
#OFFSET:0.0;
#BPMS:0.0=120.0;
#SELECTABLE:YES;
"""


def _single_chart_notes(
    notes_block: str, *, difficulty: str = "Hard", meter: int = 8
) -> str:
    return (
        _BASE_SSC
        + f"""
#NOTEDATA:;
#STEPSTYPE:dance-single;
#DIFFICULTY:{difficulty};
#METER:{meter};
#NOTES:
{notes_block}
;
"""
    )


class SimfileAdapterHelpersTest(unittest.TestCase):
    def test_parse_metadata_reads_bpm_segments(self):
        sim_text = _single_chart_notes("0000\n0001\n0000\n0000\n,")
        with tempfile.TemporaryDirectory() as tmpdir:
            pack = _write_pack(pathlib.Path(tmpdir), sim_text=sim_text)
            sim = simfile_adapter.open_simfile(pack / "chart.ssc")
            metadata = simfile_adapter.parse_metadata(sim)
            self.assertEqual(metadata.initial_bpm, 120.0)
            self.assertEqual(len(metadata.bpm_segments), 1)
            self.assertTrue(metadata.selectable)

    def test_default_chart_index_prefers_challenge(self):
        easy = simfile_adapter.build_chart_summary(
            type(
                "Chart",
                (),
                {
                    "stepstype": "dance-single",
                    "difficulty": "Easy",
                    "meter": 5,
                    "chartname": "",
                    "credit": "",
                },
            )(),
            num_steps=1,
        )
        challenge = simfile_adapter.build_chart_summary(
            type(
                "Chart",
                (),
                {
                    "stepstype": "dance-single",
                    "difficulty": "Challenge",
                    "meter": 12,
                    "chartname": "",
                    "credit": "",
                },
            )(),
            num_steps=1,
        )
        charts = [
            models.ParsedChart(
                summary=easy,
                times_sec=[0.0],
                arrow_rows=["1000"],
                column_codes=[int("1000", 4)],
            ),
            models.ParsedChart(
                summary=challenge,
                times_sec=[0.0],
                arrow_rows=["0100"],
                column_codes=[int("0100", 4)],
            ),
        ]
        self.assertEqual(simfile_adapter.default_chart_index(charts), 1)

    def test_default_chart_index_empty_returns_zero(self):
        self.assertEqual(simfile_adapter.default_chart_index([]), 0)

    def test_build_available_charts_lists_non_single_inventory(self):
        sim_text = (
            _BASE_SSC
            + """
#NOTEDATA:;
#STEPSTYPE:dance-double;
#DIFFICULTY:Challenge;
#METER:10;
#NOTES:
0000
0000
0000
0000
,
0000
0000
0000
0000
;

#NOTEDATA:;
#STEPSTYPE:dance-single;
#DIFFICULTY:Hard;
#METER:8;
#NOTES:
0000
0001
0000
0000
,
0000
0000
0000
0000
;
"""
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            pack = _write_pack(pathlib.Path(tmpdir), sim_text=sim_text)
            sim = simfile_adapter.open_simfile(pack / "chart.ssc")
            available = simfile_adapter.build_available_charts(sim)
            self.assertEqual(len(available), 1)
            self.assertEqual(available[0].stepstype, "dance-double")


class ParseSongPackTest(unittest.TestCase):
    def test_parse_song_pack_exports_valid_chart(self):
        sim_text = _single_chart_notes(
            """0000
0001
0000
0000
,
0000
0000
0100
0000
,"""
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            pack = _write_pack(pathlib.Path(tmpdir), sim_text=sim_text)
            result = simfile_adapter.parse_song_pack(
                pack,
                simfile_name="chart.ssc",
                normalized_bundle="bundle",
                normalized_id="test_song",
                source_pack_relpath="bundle/pack",
            )
            self.assertIsNone(result.reason)
            self.assertIsNotNone(result.pack)
            pack_obj = result.pack
            assert pack_obj is not None
            self.assertEqual(len(pack_obj.charts), 1)
            chart = pack_obj.charts[0]
            self.assertEqual(chart.summary.num_steps, 2)
            self.assertEqual(chart.arrow_rows[0], "0001")
            self.assertEqual(chart.arrow_rows[1], "0100")
            self.assertEqual(pack_obj.audio_filename, "test_song.ogg")
            self.assertEqual(pack_obj.audio_resolved_relpath, "song.ogg")

    def test_parse_song_pack_skips_invalid_hold_chart(self):
        sim_text = _single_chart_notes(
            """0000
0003
0000
0000
,"""
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            pack = _write_pack(pathlib.Path(tmpdir), sim_text=sim_text)
            result = simfile_adapter.parse_song_pack(
                pack,
                simfile_name="chart.ssc",
                normalized_bundle="bundle",
                normalized_id="bad_hold",
                source_pack_relpath="bundle/pack",
            )
            self.assertEqual(result.reason, pack_results.REASON_NO_EXPORTABLE_CHARTS)
            self.assertEqual(
                result.chart_skips[0].reason, constants.CHART_SKIP_INVALID_HOLDS
            )

    def test_parse_song_pack_no_dance_single(self):
        sim_text = (
            _BASE_SSC
            + """
#NOTEDATA:;
#STEPSTYPE:dance-double;
#DIFFICULTY:Challenge;
#METER:10;
#NOTES:
0000
0000
0000
0000
,
0000
0000
0000
0000
;
"""
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            pack = _write_pack(pathlib.Path(tmpdir), sim_text=sim_text)
            result = simfile_adapter.parse_song_pack(
                pack,
                simfile_name="chart.ssc",
                normalized_bundle="bundle",
                normalized_id="double_only",
                source_pack_relpath="bundle/pack",
            )
            self.assertEqual(result.reason, pack_results.REASON_NO_DANCE_SINGLE)

    def test_parse_song_pack_custom_difficulty_warning(self):
        sim_text = _single_chart_notes(
            "0000\n0001\n0000\n0000\n,",
            difficulty="Edit",
            meter=9,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            pack = _write_pack(pathlib.Path(tmpdir), sim_text=sim_text)
            result = simfile_adapter.parse_song_pack(
                pack,
                simfile_name="chart.ssc",
                normalized_bundle="bundle",
                normalized_id="edit_chart",
                source_pack_relpath="bundle/pack",
            )
            self.assertIsNone(result.reason)
            assert result.pack is not None
            self.assertIn("custom_difficulty", result.pack.warnings)
            self.assertEqual(result.pack.charts[0].summary.difficulty_kind, "custom")

    def test_parse_song_pack_no_audio(self):
        sim_text = _single_chart_notes("0000\n0001\n0000\n0000\n,")
        with tempfile.TemporaryDirectory() as tmpdir:
            pack = pathlib.Path(tmpdir) / "pack"
            pack.mkdir()
            (pack / "chart.ssc").write_text(sim_text, encoding="utf-8")
            result = simfile_adapter.parse_song_pack(
                pack,
                simfile_name="chart.ssc",
                normalized_bundle="bundle",
                normalized_id="no_audio",
                source_pack_relpath="bundle/pack",
            )
            self.assertEqual(result.reason, pack_results.REASON_NO_AUDIO)

    def test_parse_song_pack_parse_error_on_empty_bpms(self):
        sim_text = _single_chart_notes("0000\n0001\n0000\n0000\n,").replace(
            "#BPMS:0.0=120.0;\n", "#BPMS:;\n"
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            pack = _write_pack(pathlib.Path(tmpdir), sim_text=sim_text)
            result = simfile_adapter.parse_song_pack(
                pack,
                simfile_name="chart.ssc",
                normalized_bundle="bundle",
                normalized_id="no_bpms",
                source_pack_relpath="bundle/pack",
            )
            self.assertEqual(result.reason, pack_results.REASON_PARSE_ERROR)

    def test_parse_song_pack_over_cap_skip(self):
        notes = ",\n".join("0000\n0001\n0000\n0000" for _ in range(5))
        sim_text = _single_chart_notes(notes + ",")
        cfg = config.PrepConfig(max_steps_per_chart=3)
        with tempfile.TemporaryDirectory() as tmpdir:
            pack = _write_pack(pathlib.Path(tmpdir), sim_text=sim_text)
            result = simfile_adapter.parse_song_pack(
                pack,
                simfile_name="chart.ssc",
                normalized_bundle="bundle",
                normalized_id="over_cap",
                source_pack_relpath="bundle/pack",
                prep_config=cfg,
            )
            self.assertEqual(result.reason, pack_results.REASON_NO_EXPORTABLE_CHARTS)
            self.assertEqual(
                result.chart_skips[0].reason, constants.CHART_SKIP_OVER_CAP
            )

    def test_parse_song_pack_skips_mine_only_chart(self):
        sim_text = _single_chart_notes(
            """0000
000M
0000
0000
,"""
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            pack = _write_pack(pathlib.Path(tmpdir), sim_text=sim_text)
            result = simfile_adapter.parse_song_pack(
                pack,
                simfile_name="chart.ssc",
                normalized_bundle="bundle",
                normalized_id="mines_only",
                source_pack_relpath="bundle/pack",
            )
            self.assertEqual(result.reason, pack_results.REASON_NO_EXPORTABLE_CHARTS)
            self.assertEqual(result.chart_skips[0].reason, constants.CHART_SKIP_EMPTY)

    def test_parse_song_pack_encoding_error(self):
        sim_text = _single_chart_notes("0000\n0001\n0000\n0000\n,")
        with tempfile.TemporaryDirectory() as tmpdir:
            pack = _write_pack(pathlib.Path(tmpdir), sim_text=sim_text)
            with unittest.mock.patch(
                "stepcovnet.dataset_prep.simfile_adapter.simfile.open_with_detected_encoding",
                side_effect=UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid"),
            ):
                result = simfile_adapter.parse_song_pack(
                    pack,
                    simfile_name="chart.ssc",
                    normalized_bundle="bundle",
                    normalized_id="encoding",
                    source_pack_relpath="bundle/pack",
                )
            self.assertEqual(result.reason, pack_results.REASON_ENCODING_ERROR)

    def test_parse_song_pack_parse_error_on_invalid_simfile(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pack = pathlib.Path(tmpdir) / "pack"
            pack.mkdir()
            (pack / "chart.ssc").write_text(
                "#VERSION:0.83;\n#NOTES:broken;\n", encoding="utf-8"
            )
            (pack / "song.ogg").write_bytes(b"audio")
            result = simfile_adapter.parse_song_pack(
                pack,
                simfile_name="chart.ssc",
                normalized_bundle="bundle",
                normalized_id="broken",
                source_pack_relpath="bundle/pack",
            )
            self.assertEqual(result.reason, pack_results.REASON_PARSE_ERROR)

    def test_parse_song_pack_dedupes_custom_difficulty_warning(self):
        sim_text = (
            _BASE_SSC
            + """
#NOTEDATA:;
#STEPSTYPE:dance-single;
#DIFFICULTY:Edit;
#METER:9;
#NOTES:
0000
0001
0000
0000
,
0000
0000
0000
0000
;

#NOTEDATA:;
#STEPSTYPE:dance-single;
#DIFFICULTY:Edit;
#METER:10;
#NOTES:
0000
0100
0000
0000
,
0000
0000
0000
0000
;
"""
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            pack = _write_pack(pathlib.Path(tmpdir), sim_text=sim_text)
            result = simfile_adapter.parse_song_pack(
                pack,
                simfile_name="chart.ssc",
                normalized_bundle="bundle",
                normalized_id="two_edits",
                source_pack_relpath="bundle/pack",
            )
            self.assertIsNone(result.reason)
            assert result.pack is not None
            self.assertEqual(result.pack.warnings.count("custom_difficulty"), 1)

    def test_parse_song_pack_logs_fake_and_lift_warning_counts(self):
        sim_text = _single_chart_notes(
            """0000
0001
0000
0000
,
0000
000F
0000
0000
,
0000
000L
0000
0000
,"""
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            pack = _write_pack(pathlib.Path(tmpdir), sim_text=sim_text)
            result = simfile_adapter.parse_song_pack(
                pack,
                simfile_name="chart.ssc",
                normalized_bundle="bundle",
                normalized_id="fake_lift",
                source_pack_relpath="bundle/pack",
            )
            self.assertIsNone(result.reason)
            assert result.pack is not None
            self.assertTrue(
                any(
                    item.startswith("fake_notes_unencoded:")
                    for item in result.pack.warnings
                )
            )
            self.assertTrue(
                any(
                    item.startswith("lift_notes_unencoded:")
                    for item in result.pack.warnings
                )
            )
