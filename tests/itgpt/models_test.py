"""Tests for the PyTorch ITGPT placement model."""

from __future__ import annotations

import pathlib
import tempfile
import unittest

import numpy as np
import torch

from stepcovnet.itgpt import config, constants, models, trainers
from stepcovnet.pytorch import runtime


def _one_train_step(
    model: models.ItgptPlacementModel,
    run_cfg: config.ItgptRunConfig,
    inputs: dict[str, torch.Tensor],
    labels: torch.Tensor,
    mask: torch.Tensor,
) -> float:
    """Run a single PyTorch forward/backward/step and return scalar loss."""
    optimizer = trainers.build_optimizer(model, run_cfg)
    criterion = models.ItgptPlacementLoss()
    model.train()
    optimizer.zero_grad()
    logits = model.forward_logits(inputs)
    loss = criterion(logits, labels, mask)
    loss.backward()
    optimizer.step()
    model.eval()
    return float(loss.detach().item())


class ItgptModelsTest(unittest.TestCase):
    def test_forward_shape_and_finite_train_step(self) -> None:
        runtime.set_seed(0)
        n_beats = constants.CHUNK_ALIGN
        model = models.build_itgpt_placement_model(
            d_model=32,
            n_heads=4,
            n_enc_layers=1,
            cnn_hidden=8,
            dropout_rate=0.0,
            max_beats=n_beats,
        )
        inputs = {
            "audio": torch.zeros(
                (
                    1,
                    n_beats,
                    constants.N_FRAMES_PER_BEAT,
                    constants.N_MELS,
                    constants.N_CHANNELS,
                ),
                dtype=torch.float32,
            ),
            "bpm": torch.tensor([[140.0]], dtype=torch.float32),
            "difficulty": torch.tensor([[8.0]], dtype=torch.float32),
        }
        labels = torch.zeros((1, n_beats, constants.N_SLOTS), dtype=torch.float32)
        labels[0, 0, 0] = 1.0
        mask = torch.ones((1, n_beats), dtype=torch.float32)

        model.eval()
        with torch.no_grad():
            pred = model(inputs)
        self.assertEqual(tuple(pred.shape), (1, n_beats, constants.N_SLOTS))

        loss = _one_train_step(
            model, config.ItgptRunConfig(learning_rate=1e-4), inputs, labels, mask
        )
        self.assertTrue(np.isfinite(loss))

        weights = models.grid_importance_weights()
        self.assertEqual(tuple(weights.shape), (constants.N_SLOTS,))
        self.assertEqual(float(weights[0]), constants.GRID_WEIGHT_16TH)
        with self.assertRaises(ValueError):
            models.build_itgpt_placement_model(d_model=32, n_heads=3)
        with self.assertRaises(ValueError):
            models.build_itgpt_placement_model(n_enc_layers=0)

    def test_mert_forward_train_and_pt_save_reload_round_trip(self) -> None:
        runtime.set_seed(0)
        n_beats = constants.CHUNK_ALIGN
        feature_dim = 16
        model = models.build_itgpt_placement_model(
            d_model=32,
            n_heads=4,
            n_enc_layers=1,
            dropout_rate=0.1,
            max_beats=n_beats,
            audio_feature_type="mert",
            n_frames_per_beat=constants.CNN_FRAMES_OUT,
            feature_dim=feature_dim,
        )
        rng = np.random.default_rng(7)
        inputs = {
            "audio": torch.as_tensor(
                rng.standard_normal(
                    (1, n_beats, constants.CNN_FRAMES_OUT, feature_dim)
                ).astype(np.float32)
            ),
            "bpm": torch.tensor([[140.0]], dtype=torch.float32),
            "difficulty": torch.tensor([[8.0]], dtype=torch.float32),
        }
        labels = torch.zeros((1, n_beats, constants.N_SLOTS), dtype=torch.float32)
        labels[0, 0, 0] = 1.0
        mask = torch.ones((1, n_beats), dtype=torch.float32)

        model.eval()
        with torch.no_grad():
            pred_before = model(inputs).numpy()
        self.assertEqual(pred_before.shape, (1, n_beats, constants.N_SLOTS))
        loss = _one_train_step(
            model,
            config.ItgptRunConfig(learning_rate=1e-4, weight_decay=0.0),
            inputs,
            labels,
            mask,
        )
        self.assertTrue(np.isfinite(loss))
        with torch.no_grad():
            pred_trained = model(inputs).numpy()
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt = trainers.save_checkpoint(
                model, pathlib.Path(tmpdir) / "mert_roundtrip.pt"
            )
            reloaded = trainers.load_placement_model(ckpt, device="cpu")
            with torch.no_grad():
                pred_reloaded = reloaded(inputs).numpy()
        np.testing.assert_allclose(pred_trained, pred_reloaded, atol=1e-5)
        with self.assertRaises(ValueError):
            models.build_itgpt_placement_model(
                d_model=32,
                n_heads=4,
                audio_feature_type="mert",
                n_frames_per_beat=16,
            )

    def test_early_film_identity_init_train_and_pt_reload(self) -> None:
        runtime.set_seed(0)
        n_beats = constants.CHUNK_ALIGN
        feature_dim = 16
        film_layer = models.ItgptConditionFilm(d_model=32)
        rng = np.random.default_rng(11)
        subbeat = torch.as_tensor(
            rng.standard_normal((n_beats, constants.CNN_FRAMES_OUT, 32)).astype(
                np.float32
            )
        )
        cond = torch.as_tensor(rng.standard_normal((1, 64)).astype(np.float32))
        with torch.no_grad():
            subbeat_out = film_layer(
                subbeat,
                cond,
                batch_size=1,
                n_beats=n_beats,
                frames_per_beat=constants.CNN_FRAMES_OUT,
            ).numpy()
        np.testing.assert_allclose(subbeat_out, subbeat.numpy(), atol=1e-6)

        model = models.build_itgpt_placement_model(
            d_model=32,
            n_heads=4,
            n_enc_layers=1,
            dropout_rate=0.0,
            max_beats=n_beats,
            audio_feature_type="mert",
            n_frames_per_beat=constants.CNN_FRAMES_OUT,
            feature_dim=feature_dim,
            early_film=True,
            cyclic_measure_pos=True,
        )
        inputs = {
            "audio": torch.as_tensor(
                rng.standard_normal(
                    (1, n_beats, constants.CNN_FRAMES_OUT, feature_dim)
                ).astype(np.float32)
            ),
            "bpm": torch.tensor([[140.0]], dtype=torch.float32),
            "difficulty": torch.tensor([[8.0]], dtype=torch.float32),
        }
        labels = torch.zeros((1, n_beats, constants.N_SLOTS), dtype=torch.float32)
        labels[0, 0, 0] = 1.0
        mask = torch.ones((1, n_beats), dtype=torch.float32)
        loss = _one_train_step(
            model,
            config.ItgptRunConfig(learning_rate=1e-3, weight_decay=0.0),
            inputs,
            labels,
            mask,
        )
        self.assertTrue(np.isfinite(loss))
        with torch.no_grad():
            pred_trained = model(inputs).numpy()
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt = trainers.save_checkpoint(
                model, pathlib.Path(tmpdir) / "mert_film_roundtrip.pt"
            )
            reloaded = trainers.load_placement_model(ckpt, device="cpu")
            self.assertTrue(reloaded.early_film)
            self.assertTrue(reloaded.cyclic_measure_pos)
            with torch.no_grad():
                pred_reloaded = reloaded(inputs).numpy()
        np.testing.assert_allclose(pred_trained, pred_reloaded, atol=1e-5)

    def test_continuous_beat_conv_spans_boundaries_and_reloads(self) -> None:
        runtime.set_seed(0)
        n_beats = constants.CHUNK_ALIGN
        feature_dim = 16
        model_sep = models.build_itgpt_placement_model(
            d_model=32,
            n_heads=4,
            n_enc_layers=1,
            dropout_rate=0.0,
            max_beats=n_beats,
            audio_feature_type="mert",
            n_frames_per_beat=constants.CNN_FRAMES_OUT,
            feature_dim=feature_dim,
            continuous_beat_conv=False,
        )
        model_cont = models.build_itgpt_placement_model(
            d_model=32,
            n_heads=4,
            n_enc_layers=1,
            dropout_rate=0.0,
            max_beats=n_beats,
            audio_feature_type="mert",
            n_frames_per_beat=constants.CNN_FRAMES_OUT,
            feature_dim=feature_dim,
            continuous_beat_conv=True,
        )
        model_cont.load_state_dict(model_sep.state_dict())
        model_sep.eval()
        model_cont.eval()

        rng = np.random.default_rng(19)
        inputs = {
            "audio": torch.as_tensor(
                rng.standard_normal(
                    (1, n_beats, constants.CNN_FRAMES_OUT, feature_dim)
                ).astype(np.float32)
            ),
            "bpm": torch.tensor([[140.0]], dtype=torch.float32),
            "difficulty": torch.tensor([[8.0]], dtype=torch.float32),
        }
        with torch.no_grad():
            pred_sep = model_sep(inputs).numpy()
            pred_cont = model_cont(inputs).numpy()
        self.assertGreater(float(np.max(np.abs(pred_cont - pred_sep))), 1e-4)

        labels = torch.zeros((1, n_beats, constants.N_SLOTS), dtype=torch.float32)
        labels[0, 1, 0] = 1.0
        mask = torch.ones((1, n_beats), dtype=torch.float32)
        loss = _one_train_step(
            model_cont,
            config.ItgptRunConfig(learning_rate=1e-3, weight_decay=0.0),
            inputs,
            labels,
            mask,
        )
        self.assertTrue(np.isfinite(loss))
        with torch.no_grad():
            pred_trained = model_cont(inputs).numpy()
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt = trainers.save_checkpoint(
                model_cont, pathlib.Path(tmpdir) / "mert_cont_roundtrip.pt"
            )
            reloaded = trainers.load_placement_model(ckpt, device="cpu")
            self.assertTrue(reloaded.continuous_beat_conv)
            with torch.no_grad():
                pred_reloaded = reloaded(inputs).numpy()
        np.testing.assert_allclose(pred_trained, pred_reloaded, atol=1e-5)

    def test_mert_layer_mixer_trains_weights_and_reloads(self) -> None:
        runtime.set_seed(0)
        n_beats = constants.CHUNK_ALIGN
        n_layers = 4
        feature_dim = 16
        model = models.build_itgpt_placement_model(
            d_model=32,
            n_heads=4,
            n_enc_layers=1,
            dropout_rate=0.0,
            max_beats=n_beats,
            audio_feature_type="mert",
            n_frames_per_beat=constants.CNN_FRAMES_OUT,
            feature_dim=feature_dim,
            n_mert_layers=n_layers,
        )
        rng = np.random.default_rng(23)
        inputs = {
            "audio": torch.as_tensor(
                rng.standard_normal(
                    (1, n_beats, constants.CNN_FRAMES_OUT, n_layers, feature_dim)
                ).astype(np.float32)
            ),
            "bpm": torch.tensor([[140.0]], dtype=torch.float32),
            "difficulty": torch.tensor([[8.0]], dtype=torch.float32),
        }
        labels = torch.zeros((1, n_beats, constants.N_SLOTS), dtype=torch.float32)
        labels[0, 0, 0] = 1.0
        mask = torch.ones((1, n_beats), dtype=torch.float32)

        assert model.beat_mert is not None
        assert model.beat_mert.layer_logits is not None
        np.testing.assert_allclose(
            model.beat_mert.layer_logits.detach().numpy(),
            np.zeros(n_layers),
            atol=1e-6,
        )
        loss = _one_train_step(
            model,
            config.ItgptRunConfig(learning_rate=1e-2, weight_decay=0.0),
            inputs,
            labels,
            mask,
        )
        self.assertTrue(np.isfinite(loss))
        self.assertGreater(
            float(np.max(np.abs(model.beat_mert.layer_logits.detach().numpy()))),
            0.0,
        )
        with torch.no_grad():
            pred_trained = model(inputs).numpy()
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt = trainers.save_checkpoint(
                model, pathlib.Path(tmpdir) / "mert_mix_roundtrip.pt"
            )
            reloaded = trainers.load_placement_model(ckpt, device="cpu")
            self.assertEqual(reloaded.n_mert_layers, n_layers)
            with torch.no_grad():
                pred_reloaded = reloaded(inputs).numpy()
        np.testing.assert_allclose(pred_trained, pred_reloaded, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
