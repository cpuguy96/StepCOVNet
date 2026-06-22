"""Tests for onset_events.charts."""

import json
import pathlib
import tempfile
import unittest

import numpy as np

from stepcovnet import datasets
from stepcovnet.dataset_prep import constants
from stepcovnet.onset_events import charts

TEST_DATA_DIR = pathlib.Path(__file__).resolve().parent.parent / "testdata"
MAYU_CHART = TEST_DATA_DIR / "mayu.txt"


def _write_chart(
    path: str, step_lines: list[str], difficulty: str = "Challenge"
) -> None:
    with pathlib.Path(path).open("w") as chart_file:
        chart_file.write("TITLE Test\nBPM 120.0\nNOTES\n")
        chart_file.write(f"DIFFICULTY {difficulty}\n")
        chart_file.write("".join(step_lines))


class ChartsTest(unittest.TestCase):
    def test_max_steps_constant(self):
        self.assertEqual(charts.MAX_STEPS_PER_CHART, 2048)

    def test_load_onset_times_inline_chart(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = pathlib.Path(tmpdir) / "chart.txt"
            _write_chart(
                path,
                ["1000 1.0\n", "0100 0.5\n", "0010 2.0\n"],
            )
            times = charts.load_onset_times(path, max_steps=None)
            self.assertIsNotNone(times)
            assert times is not None
            np.testing.assert_allclose(times, [0.5, 1.0, 2.0])

    def test_load_onset_times_matches_datasets_parser(self):
        times, _ = datasets._parse_step_chart(MAYU_CHART, binary_timings=True)
        loaded = charts.load_onset_times(MAYU_CHART, max_steps=None)
        self.assertIsNotNone(loaded)
        assert loaded is not None
        np.testing.assert_allclose(loaded, np.sort(times))

    def test_load_onset_times_stops_at_second_difficulty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = pathlib.Path(tmpdir) / "multi_diff.txt"
            with pathlib.Path(path).open("w") as chart_file:
                chart_file.write("TITLE X\nBPM 128.0\nNOTES\nDIFFICULTY Challenge\n")
                chart_file.write("0000 0.5\n")
                chart_file.write("DIFFICULTY Easy\n")
                chart_file.write("0000 1.0\n")
            times = charts.load_onset_times(path, max_steps=None)
            self.assertIsNotNone(times)
            assert times is not None
            np.testing.assert_allclose(times, [0.5])

    def test_count_steps(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = pathlib.Path(tmpdir) / "chart.txt"
            _write_chart(path, ["1000 0.0\n", "0100 0.5\n", "0010 1.0\n"])
            self.assertEqual(charts.count_steps(path), 3)

    def test_chart_exceeds_step_cap(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            small_path = pathlib.Path(tmpdir) / "small.txt"
            _write_chart(small_path, ["1000 0.0\n"])
            self.assertFalse(charts.chart_exceeds_step_cap(small_path, max_steps=1))
            self.assertTrue(charts.chart_exceeds_step_cap(small_path, max_steps=0))

            large_path = pathlib.Path(tmpdir) / "large.txt"
            step_lines = [f"1000 {i * 0.01}\n" for i in range(2049)]
            _write_chart(large_path, step_lines)
            self.assertTrue(charts.chart_exceeds_step_cap(large_path))
            self.assertFalse(charts.chart_exceeds_step_cap(large_path, max_steps=2049))

    def test_load_onset_times_returns_none_when_over_cap(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = pathlib.Path(tmpdir) / "large.txt"
            step_lines = [f"1000 {i * 0.01}\n" for i in range(2049)]
            _write_chart(path, step_lines)
            self.assertIsNone(charts.load_onset_times(path))
            loaded = charts.load_onset_times(path, max_steps=None)
            self.assertIsNotNone(loaded)
            assert loaded is not None
            self.assertEqual(len(loaded), 2049)

    def test_load_onset_times_allows_exact_cap(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = pathlib.Path(tmpdir) / "cap.txt"
            step_lines = [f"1000 {i * 0.01}\n" for i in range(2048)]
            _write_chart(path, step_lines)
            loaded = charts.load_onset_times(path)
            self.assertIsNotNone(loaded)
            assert loaded is not None
            self.assertEqual(len(loaded), 2048)

    def test_load_onset_times_empty_chart(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = pathlib.Path(tmpdir) / "empty.txt"
            _write_chart(path, [])
            loaded = charts.load_onset_times(path, max_steps=None)
            self.assertIsNotNone(loaded)
            assert loaded is not None
            self.assertEqual(len(loaded), 0)

    def test_mayu_chart_under_default_cap(self):
        loaded = charts.load_onset_times(MAYU_CHART)
        self.assertIsNotNone(loaded)
        assert loaded is not None
        self.assertLessEqual(len(loaded), charts.MAX_STEPS_PER_CHART)

    def test_load_onset_times_from_chart_json(self):
        payload = {
            "schema_version": constants.SCHEMA_VERSION,
            "normalized_bundle": "bundle",
            "normalized_id": "song",
            "source_pack_relpath": "bundle/song",
            "source_simfile": "chart.ssc",
            "metadata": {
                "title": "Song",
                "artist": "",
                "subtitle": "",
                "music_filename": "song.ogg",
                "offset_sec": 0.0,
                "initial_bpm": 120.0,
                "bpm_segments": [{"start_beat": 0.0, "bpm": 120.0}],
                "selectable": True,
            },
            "charts": [
                {
                    "summary": {
                        "stepstype": "dance-single",
                        "difficulty": "beginner",
                        "difficulty_kind": "standard",
                        "meter": 1,
                        "chart_name": "",
                        "credit": "",
                        "num_steps": 1,
                    },
                    "times_sec": [0.25],
                    "arrow_rows": ["0001"],
                    "column_codes": [1],
                },
                {
                    "summary": {
                        "stepstype": "dance-single",
                        "difficulty": "challenge",
                        "difficulty_kind": "standard",
                        "meter": 10,
                        "chart_name": "",
                        "credit": "",
                        "num_steps": 1,
                    },
                    "times_sec": [0.5],
                    "arrow_rows": ["0100"],
                    "column_codes": [16],
                },
            ],
            "default_chart_index": 1,
            "available_charts": [],
            "audio_filename": "song.ogg",
            "audio_source": "music_tag",
            "audio_resolved_relpath": "song.ogg",
            "warnings": [],
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = pathlib.Path(tmpdir) / "song.chart.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            beginner = charts.load_onset_times(path, chart_index=0, max_steps=None)
            challenge = charts.load_onset_times(path, chart_index=1, max_steps=None)
            self.assertIsNotNone(beginner)
            self.assertIsNotNone(challenge)
            assert beginner is not None
            assert challenge is not None
            np.testing.assert_allclose(beginner, [0.25])
            np.testing.assert_allclose(challenge, [0.5])
