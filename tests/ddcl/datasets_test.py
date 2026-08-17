"""Tests for DDCL dataset batching."""

from __future__ import annotations

import unittest

import numpy as np

from stepcovnet.ddcl import constants, datasets


def _chart() -> datasets.DdclChart:
    n_beats = 4
    slots = np.zeros((n_beats, constants.N_SLOTS), dtype=np.float32)
    slots[1, 3] = 1.0
    return datasets.DdclChart(
        song_key="b/s",
        difficulty="hard",
        meter=8,
        beat_audio=np.zeros(
            (n_beats, 8, constants.N_MELS, constants.N_CHANNELS),
            dtype=np.float32,
        ),
        stream=np.zeros((n_beats, constants.STREAM_DIM), dtype=np.float32),
        slots=slots,
        memlen=1,
    )


class DdclDatasetsTest(unittest.TestCase):
    def test_sample_train_batch_shapes(self):
        rng = np.random.default_rng(0)
        inputs, labels = datasets.sample_train_batch(
            [_chart()],
            batch_size=3,
            rng=rng,
        )
        self.assertEqual(inputs["audio_fwd"].shape[0], 3)
        self.assertEqual(labels.shape, (3, constants.N_SLOTS))
        predict_inputs = datasets.chart_model_inputs(_chart())
        self.assertEqual(predict_inputs["audio_fwd"].shape[0], 4)
        batches = list(datasets.iter_window_batches(_chart(), batch_size=3))
        self.assertEqual([int(batch.shape[0]) for batch in batches], [3, 1])

    def test_windows_for_indices_rejects_empty(self):
        with self.assertRaises(ValueError):
            datasets.windows_for_indices(_chart(), np.array([], dtype=np.int64))

    def test_sample_rejects_empty(self):
        rng = np.random.default_rng(0)
        with self.assertRaises(ValueError):
            datasets.sample_train_batch([], batch_size=1, rng=rng)


if __name__ == "__main__":
    unittest.main()
