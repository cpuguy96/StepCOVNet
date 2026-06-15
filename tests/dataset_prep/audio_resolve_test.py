"""Tests for dataset_prep.audio_resolve."""

import pathlib
import tempfile
import unittest

from stepcovnet.dataset_prep import audio_resolve, constants


class AudioResolveTest(unittest.TestCase):
    def test_list_audio_files_returns_empty_for_missing_directory(self):
        self.assertEqual(audio_resolve.list_audio_files(pathlib.Path("missing")), [])

    def test_resolve_audio_uses_music_tag_match(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pack = pathlib.Path(tmpdir)
            (pack / "Song.ogg").write_bytes(b"ogg")
            (pack / "other.mp3").write_bytes(b"mp3")
            result = audio_resolve.resolve_audio(
                pack,
                music_filename="other.mp3",
                simfile_name="chart.ssc",
                title="Song",
            )
            self.assertIsNotNone(result)
            assert result is not None
            self.assertEqual(result.audio_filename, "other.mp3")
            self.assertEqual(result.audio_source, constants.AUDIO_SOURCE_MUSIC_TAG)
            self.assertEqual(result.warnings, [])

    def test_resolve_audio_matches_music_tag_case_insensitive(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pack = pathlib.Path(tmpdir)
            (pack / "SONG.OGG").write_bytes(b"ogg")
            result = audio_resolve.resolve_audio(
                pack,
                music_filename="song.ogg",
                simfile_name="chart.ssc",
                title="Song",
            )
            self.assertIsNotNone(result)
            assert result is not None
            self.assertEqual(result.audio_filename, "SONG.OGG")

    def test_resolve_audio_infers_single_candidate(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pack = pathlib.Path(tmpdir)
            (pack / "only.wav").write_bytes(b"wav")
            result = audio_resolve.resolve_audio(
                pack,
                music_filename="missing.ogg",
                simfile_name="chart.ssc",
                title="Example",
            )
            self.assertIsNotNone(result)
            assert result is not None
            self.assertEqual(result.audio_source, constants.AUDIO_SOURCE_INFERRED)
            self.assertEqual(result.warnings, ["audio_inferred_single_candidate"])

    def test_resolve_audio_heuristic_prefers_simfile_stem(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pack = pathlib.Path(tmpdir)
            (pack / "chart.mp3").write_bytes(b"a" * 10)
            (pack / "zzz.mp3").write_bytes(b"b" * 100)
            result = audio_resolve.resolve_audio(
                pack,
                music_filename="missing.ogg",
                simfile_name="chart.ssc",
                title="Unrelated",
            )
            self.assertIsNotNone(result)
            assert result is not None
            self.assertEqual(result.audio_filename, "chart.mp3")
            self.assertTrue(result.warnings[0].startswith("audio_inferred_heuristic"))

    def test_resolve_audio_heuristic_prefers_title_slug(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pack = pathlib.Path(tmpdir)
            (pack / "testsong.mp3").write_bytes(b"a" * 10)
            (pack / "zzz.mp3").write_bytes(b"b" * 100)
            result = audio_resolve.resolve_audio(
                pack,
                music_filename="missing.ogg",
                simfile_name="other.ssc",
                title="Test Song!",
            )
            self.assertIsNotNone(result)
            assert result is not None
            self.assertEqual(result.audio_filename, "testsong.mp3")

    def test_resolve_audio_normalizes_punctuation_in_music_tag(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pack = pathlib.Path(tmpdir)
            (pack / "songfile.ogg").write_bytes(b"ogg")
            result = audio_resolve.resolve_audio(
                pack,
                music_filename="song-file.ogg",
                simfile_name="chart.ssc",
                title="Song",
            )
            self.assertIsNotNone(result)
            assert result is not None
            self.assertEqual(result.audio_filename, "songfile.ogg")

    def test_resolve_audio_scores_music_stem_during_inference(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pack = pathlib.Path(tmpdir)
            (pack / "song-file.mp3").write_bytes(b"a" * 5)
            (pack / "other.mp3").write_bytes(b"b" * 50)
            result = audio_resolve.resolve_audio(
                pack,
                music_filename="song-file.ogg",
                simfile_name="chart.ssc",
                title="Unrelated",
            )
            self.assertIsNotNone(result)
            assert result is not None
            self.assertEqual(result.audio_filename, "song-file.mp3")

    def test_resolve_audio_returns_none_when_no_audio_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pack = pathlib.Path(tmpdir)
            (pack / "readme.txt").write_text("no audio", encoding="utf-8")
            self.assertIsNone(
                audio_resolve.resolve_audio(
                    pack,
                    music_filename="song.ogg",
                    simfile_name="chart.ssc",
                    title="Song",
                )
            )
