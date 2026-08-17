"""Tests for the DDCL ConvLSTM placement model."""

from __future__ import annotations

import unittest

import numpy as np

from stepcovnet.ddcl import constants, models


class DdclModelsTest(unittest.TestCase):
    def test_convlstm_output_shape(self):
        memlen = 2
        n_frames = 32
        model = models.build_convlstm_placement_model(
            memlen=memlen,
            n_frames=n_frames,
            lstm_units=8,
            dropout_rate=0.0,
            dense_sizes=(16, 8),
            model_name="ddcl_test",
        )
        batch = 2
        context = memlen + 1
        audio = np.zeros(
            (batch, context, n_frames, constants.N_MELS, constants.N_CHANNELS),
            dtype=np.float32,
        )
        stream = np.zeros((batch, context, constants.STREAM_DIM), dtype=np.float32)
        prediction = model.predict(
            {
                "audio_fwd": audio,
                "audio_bwd": audio,
                "stream_fwd": stream,
                "stream_bwd": stream,
            },
            verbose=0,
        )
        self.assertEqual(prediction.shape, (batch, constants.N_SLOTS))

    def test_rejects_bad_memlen(self):
        with self.assertRaises(ValueError):
            models.build_convlstm_placement_model(memlen=-1)


if __name__ == "__main__":
    unittest.main()
