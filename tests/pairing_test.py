import pathlib
import tempfile
import unittest

from stepcovnet import datasets, pairing


class PairingTest(unittest.TestCase):
    def test_list_audio_chart_pairs_finds_match(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path = pathlib.Path(tmpdir) / "song.ogg"
            chart_path = pathlib.Path(tmpdir) / "song.txt"
            with pathlib.Path(audio_path).open("wb") as audio_file:
                audio_file.write(b"audio")
            with pathlib.Path(chart_path).open("w") as chart_file:
                chart_file.write("TITLE test\nBPM 120\nNOTES\n")
            pairs = pairing.list_audio_chart_pairs(tmpdir)
            self.assertEqual(pairs, [(str(audio_path), str(chart_path))])

    def test_datasets_reexports_pairing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path = pathlib.Path(tmpdir) / "track.wav"
            chart_path = pathlib.Path(tmpdir) / "track.txt"
            with pathlib.Path(audio_path).open("wb") as audio_file:
                audio_file.write(b"audio")
            with pathlib.Path(chart_path).open("w") as chart_file:
                chart_file.write("TITLE test\nBPM 120\nNOTES\n")
            self.assertEqual(
                datasets.list_audio_chart_pairs(tmpdir),
                pairing.list_audio_chart_pairs(tmpdir),
            )
            self.assertEqual(
                datasets.list_training_samples(tmpdir),
                pairing.list_training_samples(tmpdir),
            )

    def test_list_training_samples_falls_back_to_legacy_txt(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path = pathlib.Path(tmpdir) / "song.ogg"
            chart_path = pathlib.Path(tmpdir) / "song.txt"
            audio_path.write_bytes(b"audio")
            chart_path.write_text("TITLE test\nBPM 120\nNOTES\n")
            self.assertEqual(
                pairing.list_training_samples(tmpdir),
                [(str(audio_path), str(chart_path), 0)],
            )
