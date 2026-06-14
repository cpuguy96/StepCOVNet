import unittest

import numpy as np

from stepcovnet.onset_events import targets


class TargetsTest(unittest.TestCase):
    def test_n_max_onsets_constant(self):
        self.assertEqual(targets.N_MAX_ONSETS, 1024)

    def test_pad_onset_times_empty(self):
        times_padded, mask = targets.pad_onset_times(np.array([], dtype=np.float64))

        self.assertEqual(times_padded.shape, (1024,))
        self.assertEqual(mask.shape, (1024,))
        self.assertEqual(times_padded.dtype, np.float32)
        self.assertEqual(mask.dtype, np.float32)
        np.testing.assert_array_equal(times_padded, 0.0)
        np.testing.assert_array_equal(mask, 0.0)

    def test_pad_onset_times_partial(self):
        times = np.array([0.5, 1.0, 2.0], dtype=np.float64)
        times_padded, mask = targets.pad_onset_times(times, n_max=5)

        np.testing.assert_allclose(times_padded[:3], [0.5, 1.0, 2.0])
        np.testing.assert_allclose(times_padded[3:], 0.0)
        np.testing.assert_allclose(mask, [1.0, 1.0, 1.0, 0.0, 0.0])

    def test_pad_onset_times_full(self):
        times = np.arange(4, dtype=np.float32)
        times_padded, mask = targets.pad_onset_times(times, n_max=4)

        np.testing.assert_allclose(times_padded, times)
        np.testing.assert_allclose(mask, [1.0, 1.0, 1.0, 1.0])

    def test_pad_onset_times_default_n_max(self):
        times = np.array([1.0], dtype=np.float32)
        times_padded, mask = targets.pad_onset_times(times)

        self.assertEqual(times_padded.shape[0], targets.N_MAX_ONSETS)
        self.assertEqual(mask.shape[0], targets.N_MAX_ONSETS)
        self.assertEqual(float(mask[0]), 1.0)
        self.assertEqual(float(mask.sum()), 1.0)

    def test_pad_onset_times_raises_when_over_cap(self):
        times = np.arange(3, dtype=np.float32)
        with self.assertRaises(ValueError):
            targets.pad_onset_times(times, n_max=2)

    def test_clip_times_to_duration_empty(self):
        clipped = targets.clip_times_to_duration(np.array([]), duration_sec=10.0)
        self.assertEqual(clipped.shape, (0,))

    def test_clip_times_to_duration_keeps_all_within_cap(self):
        times = np.array([0.0, 1.5, 10.0], dtype=np.float64)
        clipped = targets.clip_times_to_duration(times, duration_sec=10.0)
        np.testing.assert_array_equal(clipped, times)

    def test_clip_times_to_duration_drops_beyond_cap(self):
        times = np.array([0.5, 10.0, 10.01, 300.0], dtype=np.float64)
        clipped = targets.clip_times_to_duration(times, duration_sec=10.0)
        np.testing.assert_allclose(clipped, [0.5, 10.0])

    def test_clip_times_to_duration_preserves_dtype(self):
        times = np.array([1.0, 2.0], dtype=np.float32)
        clipped = targets.clip_times_to_duration(times, duration_sec=1.5)
        self.assertEqual(clipped.dtype, np.float32)
        np.testing.assert_allclose(clipped, [1.0])

    def test_clip_then_pad_workflow(self):
        times = np.array([0.0, 299.0, 300.0, 301.0, 400.0], dtype=np.float64)
        clipped = targets.clip_times_to_duration(times, duration_sec=300.0)
        times_padded, mask = targets.pad_onset_times(clipped, n_max=4)

        np.testing.assert_allclose(clipped, [0.0, 299.0, 300.0])
        np.testing.assert_allclose(times_padded[:3], [0.0, 299.0, 300.0])
        np.testing.assert_allclose(mask, [1.0, 1.0, 1.0, 0.0])
