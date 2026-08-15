"""Tests for the DDC C-LSTM placement model."""

from __future__ import annotations

import unittest

import numpy as np

from stepcovnet.ddc import constants, models


class DdcModelsTest(unittest.TestCase):
    def test_clstm_output_shape(self):
        model = models.build_clstm_placement_model(
            lstm_units=16,
            lstm_layers=1,
            dropout_rate=0.0,
            dnn_sizes=(8,),
            model_name="ddc_test",
        )
        n_time = 6
        audio = np.zeros(
            (
                1,
                n_time,
                constants.CONTEXT_FRAMES,
                constants.N_MELS,
                constants.N_CHANNELS,
            ),
            dtype=np.float32,
        )
        difficulty = np.zeros((1, n_time, constants.N_DIFFICULTIES), dtype=np.float32)
        difficulty[..., 0] = 1.0
        prediction = model.predict(
            {"audio": audio, "difficulty": difficulty},
            verbose=0,
        )
        self.assertEqual(prediction.shape, (1, n_time, 1))
        trained = models.build_clstm_placement_model(
            lstm_units=8,
            lstm_layers=1,
            dropout_rate=0.0,
            dnn_sizes=(8,),
            model_name="ddc_varlen",
        )
        trained.compile(optimizer="sgd", loss="binary_crossentropy")
        train_t = 32
        train_audio = np.zeros(
            (
                2,
                train_t,
                constants.CONTEXT_FRAMES,
                constants.N_MELS,
                constants.N_CHANNELS,
            ),
            dtype=np.float32,
        )
        train_diff = np.zeros((2, train_t, constants.N_DIFFICULTIES), dtype=np.float32)
        train_diff[..., 0] = 1.0
        trained.train_on_batch(
            {"audio": train_audio, "difficulty": train_diff},
            np.zeros((2, train_t, 1), dtype=np.float32),
        )
        eval_t = 60
        eval_audio = np.zeros(
            (
                1,
                eval_t,
                constants.CONTEXT_FRAMES,
                constants.N_MELS,
                constants.N_CHANNELS,
            ),
            dtype=np.float32,
        )
        eval_diff = np.zeros((1, eval_t, constants.N_DIFFICULTIES), dtype=np.float32)
        eval_diff[..., 0] = 1.0
        eval_pred = trained.predict(
            {"audio": eval_audio, "difficulty": eval_diff},
            verbose=0,
        )
        self.assertEqual(eval_pred.shape, (1, eval_t, 1))

    def test_clstm_rejects_empty_stack(self):
        with self.assertRaises(ValueError):
            models.build_clstm_placement_model(lstm_layers=0)
        with self.assertRaises(ValueError):
            models.build_clstm_placement_model(lstm_units=0)


if __name__ == "__main__":
    unittest.main()
