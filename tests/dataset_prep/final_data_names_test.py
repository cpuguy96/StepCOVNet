"""Tests for aligned normalized names under a preprocess output tree."""

from __future__ import annotations

import json
import pathlib
import tempfile
import unittest

from stepcovnet.dataset_prep import config, constants, export
from tests.dataset_prep.pipeline_test import _minimal_pack


def collect_final_data_name_issues(
    root: pathlib.Path,
) -> tuple[list[tuple], list[tuple], int]:
    """Collect path/audio misalignments under a preprocess output root.

    Args:
        root: Preprocess output directory containing exported song folders.

    Returns:
        Tuple of (misaligned records, legacy audio records, charts checked).
    """
    misaligned: list[tuple] = []
    legacy_audio: list[tuple] = []
    checked = 0

    for chart_path in root.rglob("*.chart.json"):
        if "_staging" in chart_path.parts:
            continue
        checked += 1
        data = json.loads(chart_path.read_text(encoding="utf-8"))
        normalized_id = data["normalized_id"]
        folder = chart_path.parent.name
        chart_name = chart_path.name
        expected_chart = f"{normalized_id}.chart.json"
        audio_filename = data.get("audio_filename", "")

        if folder != normalized_id or chart_name != expected_chart:
            misaligned.append(
                (
                    "path",
                    str(chart_path.relative_to(root)),
                    folder,
                    normalized_id,
                    chart_name,
                )
            )

        if audio_filename:
            expected_audio = (
                f"{normalized_id}{pathlib.Path(audio_filename).suffix.lower()}"
            )
            if audio_filename != expected_audio:
                misaligned.append(
                    (
                        "audio_field",
                        str(chart_path.relative_to(root)),
                        audio_filename,
                        expected_audio,
                    )
                )

        song_dir = chart_path.parent
        audio_files = [
            path
            for path in song_dir.iterdir()
            if path.suffix.lower() in constants.AUDIO_EXTENSIONS
        ]
        for audio in audio_files:
            expected_name = f"{normalized_id}{audio.suffix.lower()}"
            if audio.name != expected_name:
                legacy_audio.append(
                    (str(audio.relative_to(root)), audio.name, expected_name)
                )

        if audio_filename and not (song_dir / audio_filename).is_file():
            misaligned.append(
                (
                    "missing_audio",
                    str(chart_path.relative_to(root)),
                    audio_filename,
                    [path.name for path in audio_files],
                )
            )

    return misaligned, legacy_audio, checked


class FinalDataNamesTest(unittest.TestCase):
    def test_write_song_pack_output_passes_name_alignment(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = pathlib.Path(tmpdir)
            raw_pack = root / "raw" / "pack"
            raw_pack.mkdir(parents=True)
            (raw_pack / "song.ogg").write_bytes(b"audio")
            out_dir = root / "final_data"
            export.write_song_pack(
                _minimal_pack(),
                raw_pack_dir=raw_pack,
                output_dir=out_dir,
                prep_config=config.PrepConfig(),
            )
            misaligned, legacy_audio, checked = collect_final_data_name_issues(out_dir)
            self.assertEqual(checked, 1)
            self.assertEqual(misaligned, [])
            self.assertEqual(legacy_audio, [])

    def test_collect_final_data_name_issues_detects_legacy_audio(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = pathlib.Path(tmpdir) / "final_data"
            song_dir = root / "bundle" / "test_song"
            song_dir.mkdir(parents=True)
            chart_path = song_dir / "test_song.chart.json"
            chart_path.write_text(
                json.dumps(
                    {
                        "normalized_id": "test_song",
                        "audio_filename": "test_song.ogg",
                    }
                ),
                encoding="utf-8",
            )
            (song_dir / "Expanded.ogg").write_bytes(b"audio")
            misaligned, legacy_audio, checked = collect_final_data_name_issues(root)
            self.assertEqual(checked, 1)
            self.assertEqual(len(legacy_audio), 1)
            self.assertEqual(legacy_audio[0][1], "Expanded.ogg")
            missing = [item for item in misaligned if item[0] == "missing_audio"]
            self.assertEqual(len(missing), 1)

    def test_data_final_data_tree_if_present(self):
        root = pathlib.Path("data/final_data")
        if not root.is_dir():
            self.skipTest("data/final_data not found")
        misaligned, legacy_audio, checked = collect_final_data_name_issues(root)
        self.assertGreater(checked, 0)
        self.assertEqual(misaligned, [], misaligned[:5])
        self.assertEqual(legacy_audio, [], legacy_audio[:5])


if __name__ == "__main__":
    unittest.main()
