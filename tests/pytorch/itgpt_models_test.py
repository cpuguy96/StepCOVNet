"""Tests for PyTorch ITGPT placement model and masked slot loss."""

from __future__ import annotations

import subprocess
import sys
import unittest

import torch

from stepcovnet.itgpt import constants
from stepcovnet.pytorch.itgpt import models


class PyTorchItgptModelsTest(unittest.TestCase):
    def test_no_tensorflow_imported(self):
        proc = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import sys; "
                    "from stepcovnet.pytorch.itgpt import datasets, evaluation, models; "
                    "assert 'tensorflow' not in sys.modules, 'tensorflow imported'; "
                    "assert 'keras' not in sys.modules, 'keras imported'"
                ),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stderr or proc.stdout)

    def test_logmel_forward_dynamic_beats_and_backward(self):
        model = models.build_itgpt_placement_model(
            d_model=32,
            n_heads=4,
            n_enc_layers=1,
            cnn_hidden=8,
            dropout_rate=0.0,
            max_beats=128,
            audio_feature_type="logmel",
        )
        for n_beats in (64, 128):
            inputs = {
                "audio": torch.randn(1, n_beats, 32, 80, 3),
                "bpm": torch.tensor([[140.0]]),
                "difficulty": torch.tensor([[9.0]]),
            }
            logits = model.forward_logits(inputs)
            probs = model(inputs)
            self.assertEqual(tuple(logits.shape), (1, n_beats, constants.N_SLOTS))
            self.assertEqual(tuple(probs.shape), (1, n_beats, constants.N_SLOTS))
            self.assertTrue(torch.all((probs >= 0.0) & (probs <= 1.0)))
        loss = logits.mean()
        loss.backward()
        self.assertIsNotNone(model.onset_head.weight.grad)

    def test_mert_2d_and_multilayer_mixer_and_knobs(self):
        # 2D pre-averaged MERT + early_film + cyclic_measure_pos + continuous_beat_conv
        model = models.build_itgpt_placement_model(
            d_model=32,
            n_heads=4,
            n_enc_layers=1,
            dropout_rate=0.1,
            max_beats=64,
            audio_feature_type="mert",
            n_frames_per_beat=24,
            feature_dim=16,
            n_mert_layers=1,
            early_film=True,
            cyclic_measure_pos=True,
            continuous_beat_conv=True,
        )
        model.eval()
        inputs = {
            "audio": torch.randn(1, 64, 24, 16),
            "bpm": torch.tensor([[150.0]]),
            "difficulty": torch.tensor([[11.0]]),
        }
        out = model(inputs)
        self.assertEqual(tuple(out.shape), (1, 64, constants.N_SLOTS))

        # Verify early_film & cyclic_measure_pos start as exact zero/identity.
        assert model.early_film_subbeat is not None
        assert model.early_film_beat is not None
        assert model.bar_pos_emb is not None
        assert model.phrase_pos_emb is not None
        self.assertTrue(torch.all(model.early_film_subbeat.gamma_beta.weight == 0))
        self.assertTrue(torch.all(model.early_film_beat.gamma_beta.weight == 0))
        self.assertTrue(torch.all(model.bar_pos_emb.weight == 0))
        self.assertTrue(torch.all(model.phrase_pos_emb.weight == 0))

        # Stacked multi-layer MERT (SUPERB softmax mixer)
        multi = models.build_itgpt_placement_model(
            d_model=32,
            n_heads=4,
            n_enc_layers=1,
            dropout_rate=0.0,
            max_beats=64,
            audio_feature_type="mert",
            n_frames_per_beat=24,
            feature_dim=16,
            n_mert_layers=4,
        )
        m_inputs = {
            "audio": torch.randn(1, 64, 24, 4, 16),
            "bpm": torch.tensor([[130.0]]),
            "difficulty": torch.tensor([[7.0]]),
        }
        logits = multi.forward_logits(m_inputs)
        self.assertEqual(tuple(logits.shape), (1, 64, constants.N_SLOTS))
        logits.mean().backward()
        assert multi.beat_mert is not None
        assert multi.beat_mert.layer_logits is not None
        self.assertIsNotNone(multi.beat_mert.layer_logits.grad)

    def test_validation_errors(self):
        with self.assertRaises(ValueError):
            models.build_itgpt_placement_model(d_model=30, n_heads=4)
        with self.assertRaises(ValueError):
            models.build_itgpt_placement_model(n_enc_layers=0)
        with self.assertRaises(ValueError):
            models.build_itgpt_placement_model(audio_feature_type="invalid")
        with self.assertRaises(ValueError):
            models.build_itgpt_placement_model(
                audio_feature_type="mert", n_frames_per_beat=32
            )

    def test_placement_loss_masking_focal_and_grid_weights(self):
        loss_fn = models.ItgptPlacementLoss()
        focal_fn = models.ItgptPlacementLoss(focal_gamma=1.0, grid_weight_24th=2.0)
        norm_fn = models.ItgptPlacementLoss(normalize_by_valid_beats=True)

        logits = torch.zeros(1, 8, constants.N_SLOTS)
        targets = torch.zeros(1, 8, constants.N_SLOTS)
        targets[0, 0, 0] = 1.0
        mask = torch.tensor([[1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])

        # Perturbing padded beats must not change the masked loss.
        logits_perturbed = logits.clone()
        logits_perturbed[0, 2:, :] = 50.0
        l1 = loss_fn(logits, targets, mask)
        l2 = loss_fn(logits_perturbed, targets, mask)
        self.assertAlmostEqual(float(l1), float(l2), places=6)

        # Normalized-by-valid-beats should equal 4x mean-over-all-8 beats.
        l_norm = norm_fn(logits, targets, mask)
        self.assertAlmostEqual(float(l_norm), float(l1) * 4.0, places=6)

        # Focal loss is strictly smaller than plain BCE when p_t = 0.5 and same weights.
        focal_plain_w = models.ItgptPlacementLoss(focal_gamma=1.0)
        self.assertAlmostEqual(
            float(focal_plain_w(logits, targets, mask)),
            float(l1) * 0.5,
            places=5,
        )
        self.assertGreater(float(focal_fn(logits, targets, mask)), 0.0)
