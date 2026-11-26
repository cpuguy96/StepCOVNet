"""Tests for the PyTorch DDC C-LSTM placement model."""

from __future__ import annotations

import unittest

import torch
from torch.nn import functional as nn_func

from stepcovnet.ddc import constants, models


class DdcModelsTest(unittest.TestCase):
    def test_clstm_output_shape_and_variable_length(self) -> None:
        model = models.build_clstm_placement_model(
            lstm_units=16,
            lstm_layers=1,
            dropout_rate=0.0,
            dnn_sizes=(8,),
        )
        model.eval()
        n_time = 6
        audio = torch.zeros(
            (
                1,
                n_time,
                constants.CONTEXT_FRAMES,
                constants.N_MELS,
                constants.N_CHANNELS,
            ),
            dtype=torch.float32,
        )
        difficulty = torch.zeros(
            (1, n_time, constants.N_DIFFICULTIES), dtype=torch.float32
        )
        difficulty[..., 0] = 1.0
        with torch.no_grad():
            prediction = model({"audio": audio, "difficulty": difficulty})
        self.assertEqual(tuple(prediction.shape), (1, n_time, 1))
        self.assertTrue(bool(torch.all((prediction >= 0.0) & (prediction <= 1.0))))

        trained = models.build_clstm_placement_model(
            lstm_units=8,
            lstm_layers=1,
            dropout_rate=0.0,
            dnn_sizes=(8,),
        )
        optimizer = torch.optim.SGD(trained.parameters(), lr=0.1)
        train_t = 32
        train_audio = torch.zeros(
            (
                2,
                train_t,
                constants.CONTEXT_FRAMES,
                constants.N_MELS,
                constants.N_CHANNELS,
            ),
            dtype=torch.float32,
        )
        train_diff = torch.zeros(
            (2, train_t, constants.N_DIFFICULTIES), dtype=torch.float32
        )
        train_diff[..., 0] = 1.0
        optimizer.zero_grad()
        logits = trained.forward_logits(
            {"audio": train_audio, "difficulty": train_diff}
        )
        loss = nn_func.binary_cross_entropy_with_logits(
            logits, torch.zeros((2, train_t, 1), dtype=torch.float32)
        )
        loss.backward()
        optimizer.step()
        self.assertTrue(bool(torch.isfinite(loss)))

        trained.eval()
        eval_t = 60
        eval_audio = torch.zeros(
            (
                1,
                eval_t,
                constants.CONTEXT_FRAMES,
                constants.N_MELS,
                constants.N_CHANNELS,
            ),
            dtype=torch.float32,
        )
        eval_diff = torch.zeros(
            (1, eval_t, constants.N_DIFFICULTIES), dtype=torch.float32
        )
        eval_diff[..., 0] = 1.0
        with torch.no_grad():
            eval_pred = trained({"audio": eval_audio, "difficulty": eval_diff})
        self.assertEqual(tuple(eval_pred.shape), (1, eval_t, 1))

    def test_clstm_rejects_empty_stack(self) -> None:
        with self.assertRaises(ValueError):
            models.build_clstm_placement_model(lstm_layers=0)
        with self.assertRaises(ValueError):
            models.build_clstm_placement_model(lstm_units=0)


if __name__ == "__main__":
    unittest.main()
