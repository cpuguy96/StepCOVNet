"""Tests for DDCL 48-slot PRE (`omalley2025ddcl`)."""

from __future__ import annotations

import unittest

import numpy as np

from stepcovnet.dataset_prep import models as prep_models
from stepcovnet.ddcl import features, slots


def _constant_bpm(bpm: float = 120.0) -> list[prep_models.BpmSegment]:
    return [prep_models.BpmSegment(start_beat=0.0, bpm=bpm)]


class DdclSlotsTest(unittest.TestCase):
    def test_beat_time_round_trip_120bpm(self):
        segments = _constant_bpm(120.0)
        for beat in (0.0, 0.5, 1.0, 8.0):
            time_sec = slots.beat_to_time_sec(beat, 0.0, segments)
            recovered = slots.time_to_beat(time_sec, 0.0, segments)
            self.assertAlmostEqual(recovered, beat, places=6)
        self.assertAlmostEqual(slots.beat_to_time_sec(1.0, 0.0, segments), 0.5)

    def test_offset_shifts_time(self):
        segments = _constant_bpm(60.0)
        self.assertAlmostEqual(slots.beat_to_time_sec(0.0, 0.25, segments), -0.25)
        self.assertAlmostEqual(slots.time_to_beat(-0.25, 0.25, segments), 0.0)

    def test_slot_index_and_matrix(self):
        self.assertEqual(slots.slot_index(0.0), 0)
        self.assertEqual(slots.slot_index(0.5), 24)
        self.assertEqual(slots.slot_index(1.0), 0)
        segments = _constant_bpm(120.0)
        # 120 BPM: beat 0 at t=0, beat 0.5 at t=0.25, beat 1 at t=0.5.
        matrix = slots.times_to_slot_matrix(
            np.array([0.0, 0.25, 0.5]),
            0.0,
            segments,
        )
        self.assertEqual(matrix.shape, (2, 48))
        self.assertEqual(float(matrix[0, 0]), 1.0)
        self.assertEqual(float(matrix[0, 24]), 1.0)
        self.assertEqual(float(matrix[1, 0]), 1.0)

    def test_upsample_rhythm_bits_matches_ddcl_util(self):
        # label_to_vect_dict("1010", force_max_len=48) places 1s at 0,24.
        pattern = slots.upsample_rhythm_bits([1, 0, 1, 0], n_slots=48)
        self.assertEqual(int(pattern[0]), 1)
        self.assertEqual(int(pattern[12]), 0)
        self.assertEqual(int(pattern[24]), 1)
        self.assertEqual(int(pattern[36]), 0)
        empty = slots.upsample_rhythm_bits([], n_slots=48)
        self.assertEqual(int(empty.sum()), 0)

    def test_stream_features_meter_and_bpm(self):
        segments = [
            prep_models.BpmSegment(start_beat=0.0, bpm=120.0),
            prep_models.BpmSegment(start_beat=2.0, bpm=180.0),
        ]
        stream = slots.stream_features(4, meter=9, segments=segments)
        self.assertEqual(stream.shape, (4, 2))
        self.assertTrue(np.all(stream[:, 0] == 9.0))
        self.assertAlmostEqual(float(stream[0, 1]), 120.0)
        self.assertAlmostEqual(float(stream[2, 1]), 180.0)

    def test_rejects_empty_times(self):
        with self.assertRaises(ValueError):
            slots.times_to_slot_matrix(np.array([]), 0.0, _constant_bpm())
        with self.assertRaises(ValueError):
            slots.beat_to_time_sec(0.0, 0.0, [])


class DdclFeaturesTest(unittest.TestCase):
    def test_resample_and_windows(self):
        spec = np.arange(20 * 4 * 3, dtype=np.float32).reshape(20, 4, 3)
        window = features.resample_beat_audio(
            spec, 0.0, 0.1, n_frames=8, frame_rate=100
        )
        self.assertEqual(window.shape, (8, 4, 3))
        np.testing.assert_array_equal(window[0], spec[0])
        beat_times = np.array([0.0, 0.05, 0.10], dtype=np.float64)
        stacked = features.beats_to_audio_tensor(spec, beat_times, n_frames=4)
        self.assertEqual(stacked.shape, (2, 4, 4, 3))
        scored = features.zscore_beats(stacked)
        self.assertEqual(scored.shape, stacked.shape)
        windows = features.causal_windows(stacked, memlen=1, reverse=False)
        self.assertEqual(windows.shape, (2, 2, 4, 4, 3))
        backward = features.causal_windows(stacked, memlen=1, reverse=True)
        self.assertEqual(backward.shape, windows.shape)
        for beat_idx in range(stacked.shape[0]):
            np.testing.assert_array_equal(
                features.window_at_beat(stacked, 1, beat_idx, reverse=False),
                windows[beat_idx],
            )
            np.testing.assert_array_equal(
                features.window_at_beat(stacked, 1, beat_idx, reverse=True),
                backward[beat_idx],
            )

    def test_resample_rejects_bad_rank(self):
        with self.assertRaises(ValueError):
            features.resample_beat_audio(np.zeros((10, 80)), 0.0, 0.1)


if __name__ == "__main__":
    unittest.main()
