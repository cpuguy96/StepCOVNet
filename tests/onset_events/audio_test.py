import pathlib
import tempfile
import unittest
from unittest import mock

import numpy as np
import scipy.io.wavfile

from stepcovnet import constants
from stepcovnet.onset_events import audio


def _write_wav(path: str, samples: np.ndarray, sample_rate: int) -> None:
    int_samples = np.clip(samples * 32767.0, -32768, 32767).astype(np.int16)
    scipy.io.wavfile.write(path, sample_rate, int_samples)


class AudioTest(unittest.TestCase):
    def test_default_max_samples(self):
        self.assertEqual(
            audio.DEFAULT_MAX_SAMPLES,
            300 * constants.TARGET_SR,
        )
        self.assertEqual(
            audio.max_samples_for_cap(),
            300 * 44100,
        )

    def test_max_samples_for_cap_custom(self):
        self.assertEqual(audio.max_samples_for_cap(10.0, 22050), 220500)

    def test_load_waveform_mono_peak_normalized(self):
        sr = constants.TARGET_SR
        t = np.linspace(0, 0.1, int(0.1 * sr), endpoint=False, dtype=np.float64)
        tone = 0.5 * np.sin(2 * np.pi * 440 * t)
        samples = np.column_stack([tone, 0.25 * tone]).astype(np.float32)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = pathlib.Path(tmpdir) / "tone.wav"
            _write_wav(path, samples, sr)
            waveform = audio.load_waveform(path)
        self.assertEqual(waveform.dtype, np.float32)
        self.assertEqual(waveform.ndim, 1)
        self.assertAlmostEqual(float(np.max(np.abs(waveform))), 1.0, places=5)

    def test_load_waveform_resample_when_reported_sr_mismatches(self):
        """Cover sr != target_sample_rate branch (librosa.load may still return other sr)."""
        y_orig = np.array([0.5, -0.25], dtype=np.float32)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = pathlib.Path(tmpdir) / "x.wav"
            _write_wav(path, y_orig, constants.TARGET_SR)
            with mock.patch.object(audio, "librosa", autospec=True) as mock_librosa:
                mock_librosa.load.return_value = (y_orig, 22050)
                mock_librosa.resample.return_value = np.array(
                    [0.5, -0.25, 0.0], dtype=np.float64
                )
                waveform = audio.load_waveform(
                    path, target_sample_rate=constants.TARGET_SR
                )
                mock_librosa.resample.assert_called_once_with(
                    y_orig, orig_sr=22050, target_sr=constants.TARGET_SR
                )
        self.assertEqual(waveform.dtype, np.float32)
        self.assertAlmostEqual(float(np.max(np.abs(waveform))), 1.0, places=5)

    def test_load_waveform_resamples_to_target_sr(self):
        orig_sr = 22050
        duration = 0.05
        n = int(duration * orig_sr)
        t = np.linspace(0, duration, n, endpoint=False, dtype=np.float64)
        samples = (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = pathlib.Path(tmpdir) / "low_sr.wav"
            _write_wav(path, samples, orig_sr)
            waveform = audio.load_waveform(path, target_sample_rate=constants.TARGET_SR)
        expected_len = int(round(duration * constants.TARGET_SR))
        self.assertAlmostEqual(waveform.shape[0], expected_len, delta=2)
        self.assertAlmostEqual(float(np.max(np.abs(waveform))), 1.0, places=5)

    def test_truncate_waveform_short_unchanged(self):
        waveform = np.array([0.25, -0.5, 0.75], dtype=np.float32)
        out = audio.truncate_waveform(waveform, max_samples=10)
        np.testing.assert_array_equal(out, waveform)

    def test_truncate_waveform_long(self):
        waveform = np.arange(8, dtype=np.float32)
        out = audio.truncate_waveform(waveform, max_samples=5)
        np.testing.assert_array_equal(out, np.arange(5, dtype=np.float32))

    def test_truncate_waveform_empty(self):
        out = audio.truncate_waveform(np.zeros(0, dtype=np.float32), max_samples=4)
        self.assertEqual(out.shape, (0,))
        self.assertEqual(out.dtype, np.float32)

    def test_pad_waveform_shorter(self):
        waveform = np.array([1.0, -1.0], dtype=np.float32)
        out = audio.pad_waveform(waveform, max_samples=5)
        np.testing.assert_array_equal(
            out,
            np.array([1.0, -1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        )

    def test_pad_waveform_exact_length(self):
        waveform = np.array([0.5, 0.5], dtype=np.float32)
        out = audio.pad_waveform(waveform, max_samples=2)
        np.testing.assert_array_equal(out, waveform)

    def test_pad_waveform_empty(self):
        out = audio.pad_waveform(np.zeros(0, dtype=np.float32), max_samples=3)
        np.testing.assert_array_equal(out, np.zeros(3, dtype=np.float32))

    def test_pad_waveform_raises_when_longer_than_cap(self):
        waveform = np.ones(6, dtype=np.float32)
        with self.assertRaises(ValueError):
            audio.pad_waveform(waveform, max_samples=5)

    def test_truncate_then_pad_matches_cap(self):
        long = np.arange(12, dtype=np.float32)
        capped = audio.truncate_waveform(long, max_samples=7)
        padded = audio.pad_waveform(capped, max_samples=7)
        np.testing.assert_array_equal(padded, np.arange(7, dtype=np.float32))

    def test_load_truncate_pad_pipeline(self):
        sr = constants.TARGET_SR
        n = 200
        samples = (np.sin(np.linspace(0, 4 * np.pi, n)) * 0.8).astype(np.float32)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = pathlib.Path(tmpdir) / "clip.wav"
            _write_wav(path, samples, sr)
            loaded = audio.load_waveform(path)
        truncated = audio.truncate_waveform(loaded, max_samples=100)
        padded = audio.pad_waveform(truncated, max_samples=100)
        self.assertEqual(padded.shape, (100,))
        self.assertAlmostEqual(float(np.max(np.abs(padded))), 1.0, places=5)
