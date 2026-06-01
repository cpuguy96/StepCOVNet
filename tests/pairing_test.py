import os
import tempfile
import unittest

from stepcovnet import datasets, pairing


class PairingTest(unittest.TestCase):
    def test_list_audio_chart_pairs_finds_match(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path = os.path.join(tmpdir, "song.ogg")
            chart_path = os.path.join(tmpdir, "song.txt")
            with open(audio_path, "wb") as audio_file:
                audio_file.write(b"audio")
            with open(chart_path, "w") as chart_file:
                chart_file.write("TITLE test\nBPM 120\nNOTES\n")
            pairs = pairing.list_audio_chart_pairs(tmpdir)
            self.assertEqual(pairs, [(audio_path, chart_path)])

    def test_datasets_reexports_pairing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path = os.path.join(tmpdir, "track.wav")
            chart_path = os.path.join(tmpdir, "track.txt")
            with open(audio_path, "wb") as audio_file:
                audio_file.write(b"audio")
            with open(chart_path, "w") as chart_file:
                chart_file.write("TITLE test\nBPM 120\nNOTES\n")
            self.assertEqual(
                datasets.list_audio_chart_pairs(tmpdir),
                pairing.list_audio_chart_pairs(tmpdir),
            )
