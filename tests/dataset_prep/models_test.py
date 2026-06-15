"""Tests for dataset_prep.models."""

import json
import os
import tempfile
import unittest

from stepcovnet.dataset_prep import constants
from stepcovnet.dataset_prep import models


def _sample_song_pack() -> models.ParsedSongPack:
    return models.ParsedSongPack(
        schema_version=constants.SCHEMA_VERSION,
        normalized_bundle="itl_online_2026",
        normalized_id="expanded",
        source_pack_relpath="ITL Online 2026/[12] Expanded",
        source_simfile="sm.ssc",
        metadata=models.SimfileMetadata(
            title="Expanded!!",
            artist="Expander",
            subtitle="",
            music_filename="Expanded.ogg",
            offset_sec=0.0,
            initial_bpm=170.0,
            bpm_segments=[models.BpmSegment(start_beat=0.0, bpm=170.0)],
            selectable=True,
        ),
        charts=[
            models.ParsedChart(
                summary=models.ChartSummary(
                    stepstype="dance-single",
                    difficulty="challenge",
                    difficulty_kind=constants.DIFFICULTY_KIND_STANDARD,
                    meter=12,
                    chart_name="",
                    credit="",
                    num_steps=3,
                ),
                times_sec=[1.0, 2.0, 3.0],
                arrow_rows=["1000", "0200", "0030"],
                column_codes=[256, 32, 3],
            )
        ],
        default_chart_index=0,
        available_charts=[],
        audio_filename="expanded.ogg",
        audio_source=constants.AUDIO_SOURCE_MUSIC_TAG,
        audio_resolved_relpath="Expanded.ogg",
        warnings=[],
    )


class ModelsSerializationTest(unittest.TestCase):
    def test_parsed_song_pack_round_trip(self):
        original = _sample_song_pack()
        restored = models.ParsedSongPack.from_dict(original.as_dict())
        self.assertEqual(restored.normalized_id, "expanded")
        self.assertEqual(restored.metadata.initial_bpm, 170.0)
        self.assertEqual(len(restored.charts), 1)
        self.assertEqual(restored.charts[0].arrow_rows[0], "1000")

    def test_chart_json_path(self):
        path = models.chart_json_path("data/final_data", "itl_online_2026", "expanded")
        self.assertEqual(
            str(path).replace("\\", "/"),
            "data/final_data/itl_online_2026/expanded/expanded.chart.json",
        )

    def test_load_parsed_song_reads_nested_layout(self):
        pack = _sample_song_pack()
        with tempfile.TemporaryDirectory() as tmpdir:
            out = f"{tmpdir}/itl_online_2026/expanded"
            os.makedirs(out, exist_ok=True)
            json_path = f"{out}/expanded.chart.json"
            with open(json_path, "w", encoding="utf-8") as handle:
                json.dump(pack.as_dict(), handle)
            loaded = models.load_parsed_song(tmpdir, "itl_online_2026", "expanded")
        self.assertEqual(loaded.normalized_bundle, "itl_online_2026")
        self.assertEqual(loaded.charts[0].summary.meter, 12)

    def test_load_parsed_song_rejects_unsupported_schema_version(self):
        pack = _sample_song_pack()
        payload = pack.as_dict()
        payload["schema_version"] = 99
        with tempfile.TemporaryDirectory() as tmpdir:
            out = f"{tmpdir}/itl_online_2026/expanded"
            os.makedirs(out, exist_ok=True)
            json_path = f"{out}/expanded.chart.json"
            with open(json_path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle)
            with self.assertRaises(ValueError):
                models.load_parsed_song(tmpdir, "itl_online_2026", "expanded")

    def test_load_parsed_song_rejects_missing_schema_version(self):
        pack = _sample_song_pack()
        payload = pack.as_dict()
        del payload["schema_version"]
        with tempfile.TemporaryDirectory() as tmpdir:
            out = f"{tmpdir}/itl_online_2026/expanded"
            os.makedirs(out, exist_ok=True)
            json_path = f"{out}/expanded.chart.json"
            with open(json_path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle)
            with self.assertRaises(ValueError):
                models.load_parsed_song(tmpdir, "itl_online_2026", "expanded")

    def test_load_parsed_song_missing_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(FileNotFoundError):
                models.load_parsed_song(tmpdir, "missing_bundle", "missing_id")


if __name__ == "__main__":
    unittest.main()
