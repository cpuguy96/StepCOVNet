"""Tests for DDC PRE features (`donahue2017ddc`)."""

from __future__ import annotations

import pathlib
import tempfile
import unittest

import numpy as np
import soundfile

from stepcovnet.ddc import constants, features


class DdcFeaturesTest(unittest.TestCase):
    def test_difficulty_one_hot_challenge(self):
        vec = features.difficulty_one_hot("Challenge")
        self.assertEqual(vec.shape, (5,))
        self.assertEqual(int(np.argmax(vec)), 4)
        self.assertEqual(features.difficulty_index("hard"), 3)

    def test_difficulty_rejects_edit(self):
        with self.assertRaises(ValueError):
            features.difficulty_one_hot("edit")
        with self.assertRaises(ValueError):
            features.difficulty_index("edit")

    def test_audio_to_ddc_logmel_shape(self):
        audio = np.zeros(constants.SAMPLE_RATE, dtype=np.float32)
        spec = features.audio_to_ddc_logmel(audio)
        self.assertEqual(spec.shape[1], constants.N_MELS)
        self.assertEqual(spec.shape[2], constants.N_CHANNELS)
        self.assertGreaterEqual(spec.shape[0], constants.FRAME_RATE - 2)
        self.assertLessEqual(spec.shape[0], constants.FRAME_RATE + 4)

    def test_zscore_and_context_windows(self):
        rng = np.random.default_rng(0)
        spec = rng.normal(size=(20, constants.N_MELS, constants.N_CHANNELS)).astype(
            np.float32
        )
        scored = features.zscore_bands(spec)
        self.assertAlmostEqual(float(np.mean(scored[:, 0, 0])), 0.0, places=5)
        windows = features.context_windows(scored, radius=2)
        self.assertEqual(windows.shape, (20, 5, constants.N_MELS, constants.N_CHANNELS))
        np.testing.assert_array_equal(windows[2, 2], scored[2])
        span = features.context_windows_span(scored, 2, 5, radius=2)
        self.assertEqual(span.shape, (5, 5, constants.N_MELS, constants.N_CHANNELS))
        np.testing.assert_array_equal(span[0], windows[2])
        with self.assertRaises(ValueError):
            features.context_windows_span(scored, 18, 5, radius=2)

    def test_context_windows_rejects_bad_rank(self):
        with self.assertRaises(ValueError):
            features.context_windows(np.zeros((10, 80), dtype=np.float32))
        with self.assertRaises(ValueError):
            features.context_windows(
                np.zeros((4, 80, 3), dtype=np.float32),
                radius=-1,
            )
        with self.assertRaises(ValueError):
            features.context_windows_span(
                np.zeros((10, 80), dtype=np.float32),
                0,
                1,
            )
        with self.assertRaises(ValueError):
            features.context_windows_span(
                np.zeros((4, 80, 3), dtype=np.float32),
                0,
                1,
                radius=-1,
            )

    def test_times_to_frame_target_and_cache_round_trip(self):
        target = features.times_to_frame_target(np.array([0.0, 0.05, 9.0]), 10)
        self.assertEqual(target[0], 1.0)
        self.assertEqual(target[5], 1.0)
        self.assertEqual(float(target.sum()), 2.0)
        empty = features.times_to_frame_target(np.array([0.1]), 0)
        self.assertEqual(empty.shape, (0,))
        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path = pathlib.Path(tmpdir) / "tone.wav"
            tone = np.zeros(constants.HOP_LENGTH * 12, dtype=np.float32)
            soundfile.write(str(audio_path), tone, constants.SAMPLE_RATE)
            first = features.load_or_compute_ddc_logmel(audio_path, cache=True)
            cache_path = features.feature_cache_path(audio_path)
            self.assertTrue(cache_path.is_file())
            second = features.load_or_compute_ddc_logmel(audio_path, cache=True)
            np.testing.assert_allclose(first, second)
            uncached = features.load_or_compute_ddc_logmel(audio_path, cache=False)
            self.assertEqual(uncached.shape, first.shape)


if __name__ == "__main__":
    unittest.main()
