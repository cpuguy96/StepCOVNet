"""Tests for the DDCL ConvLSTM placement model."""

from __future__ import annotations

import pathlib
import tempfile
import unittest

import keras
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

    def test_mert_model_shape_and_keras_roundtrip(self):
        memlen = 2
        n_frames = 8
        feature_dim = 16
        model = models.build_mert_placement_model(
            memlen=memlen,
            n_frames=n_frames,
            feature_dim=feature_dim,
            proj_dim=12,
            conv_filters=8,
            lstm_units=8,
            dropout_rate=0.0,
            dense_sizes=(16, 8),
            model_name="ddcl_mert_test",
        )
        batch = 2
        context = memlen + 1
        audio = np.ones((batch, context, n_frames, feature_dim), dtype=np.float32)
        stream = np.zeros((batch, context, constants.STREAM_DIM), dtype=np.float32)
        inputs = {
            "audio_fwd": audio,
            "audio_bwd": audio,
            "stream_fwd": stream,
            "stream_bwd": stream,
        }
        pred_before = model.predict(inputs, verbose=0)
        self.assertEqual(pred_before.shape, (batch, constants.N_SLOTS))
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = pathlib.Path(tmpdir) / "mert.keras"
            model.save(save_path)
            reloaded = keras.models.load_model(save_path, compile=False)
            pred_after = reloaded.predict(inputs, verbose=0)
        np.testing.assert_allclose(pred_before, pred_after, atol=1e-5)

        with self.assertRaises(ValueError):
            models.build_mert_placement_model(memlen=-1)
        with self.assertRaises(ValueError):
            models.build_mert_placement_model(feature_dim=0)


if __name__ == "__main__":
    unittest.main()
