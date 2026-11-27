"""Tests for the PyTorch ITGPT placement model."""

from __future__ import annotations

import pathlib
import tempfile
import unittest

import numpy as np
import torch

from stepcovnet.ddcl import constants as ddcl_constants
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

    def test_mert_specaug_train_only_and_checkpoint_roundtrip(self):
        n_beats = constants.CHUNK_ALIGN
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
            mert_time_mask_max_frames=4,
            mert_channel_dropout=0.25,
        )
        rng = np.random.default_rng(7)
        inputs = {
            "audio": torch.as_tensor(
                rng.standard_normal(
                    (1, n_beats, constants.CNN_FRAMES_OUT, feature_dim)
                ).astype(np.float32)
            ),
            "bpm": torch.tensor([[130.0]], dtype=torch.float32),
            "difficulty": torch.tensor([[9.0]], dtype=torch.float32),
        }
        model.eval()
        with torch.no_grad():
            eval_a = model(inputs).numpy()
            eval_b = model(inputs).numpy()
        np.testing.assert_array_equal(eval_a, eval_b)

        model.train()
        torch.manual_seed(1)
        train_a = model(inputs).detach().numpy()
        torch.manual_seed(2)
        train_b = model(inputs).detach().numpy()
        self.assertGreater(float(np.max(np.abs(train_a - train_b))), 0.0)

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt = trainers.save_checkpoint(model, pathlib.Path(tmpdir) / "specaug.pt")
            reloaded = trainers.load_placement_model(ckpt, device="cpu")
            self.assertEqual(reloaded.mert_time_mask_max_frames, 4)
            self.assertAlmostEqual(reloaded.mert_channel_dropout, 0.25)
            with torch.no_grad():
                eval_reloaded = reloaded(inputs).numpy()
        np.testing.assert_allclose(eval_a, eval_reloaded, atol=1e-6)

    def test_density_difficulty_source_bounds_and_checkpoint_roundtrip(self):
        model = models.build_itgpt_placement_model(
            d_model=16,
            n_heads=4,
            n_enc_layers=1,
            dropout_rate=0.0,
            max_beats=constants.CHUNK_ALIGN,
            audio_feature_type=ddcl_constants.AUDIO_FEATURE_MERT,
            n_frames_per_beat=constants.CNN_FRAMES_OUT,
            feature_dim=8,
            difficulty_source=constants.DIFFICULTY_SOURCE_DENSITY,
        )
        self.assertEqual(model.difficulty_source, constants.DIFFICULTY_SOURCE_DENSITY)
        self.assertAlmostEqual(model.diff_proj.min_val, constants.MIN_DENSITY)
        self.assertAlmostEqual(model.diff_proj.max_val, constants.MAX_DENSITY)
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt = trainers.save_checkpoint(model, pathlib.Path(tmpdir) / "density.pt")
            reloaded = trainers.load_placement_model(ckpt, device="cpu")
            self.assertEqual(
                reloaded.difficulty_source, constants.DIFFICULTY_SOURCE_DENSITY
            )
            self.assertAlmostEqual(reloaded.diff_proj.min_val, constants.MIN_DENSITY)
            self.assertAlmostEqual(reloaded.diff_proj.max_val, constants.MAX_DENSITY)
        with self.assertRaises(ValueError):
            models.build_itgpt_placement_model(
                d_model=16,
                n_heads=4,
                n_enc_layers=1,
                difficulty_source="invalid",
            )

    def test_mert_specaug_preserve_downbeat_roundtrip(self):
        model = models.build_itgpt_placement_model(
            d_model=16,
            n_heads=4,
            n_enc_layers=1,
            dropout_rate=0.0,
            max_beats=constants.CHUNK_ALIGN,
            audio_feature_type=ddcl_constants.AUDIO_FEATURE_MERT,
            n_frames_per_beat=constants.CNN_FRAMES_OUT,
            feature_dim=8,
            mert_time_mask_max_frames=4,
            mert_channel_dropout=0.15,
            mert_time_mask_preserve_downbeat=True,
        )
        self.assertTrue(model.mert_time_mask_preserve_downbeat)
        assert model.beat_mert is not None
        self.assertTrue(model.beat_mert.mert_time_mask_preserve_downbeat)
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt = trainers.save_checkpoint(model, pathlib.Path(tmpdir) / "keep0.pt")
            reloaded = trainers.load_placement_model(ckpt, device="cpu")
            self.assertTrue(reloaded.mert_time_mask_preserve_downbeat)
            assert reloaded.beat_mert is not None
            self.assertTrue(reloaded.beat_mert.mert_time_mask_preserve_downbeat)

    def test_nps_condition_forward_validation_and_roundtrip(self):
        with self.assertRaises(ValueError):
            models.build_itgpt_placement_model(
                d_model=16,
                n_heads=4,
                n_enc_layers=1,
                difficulty_source=constants.DIFFICULTY_SOURCE_METER,
                use_nps_condition=True,
            )
        model = models.build_itgpt_placement_model(
            d_model=16,
            n_heads=4,
            n_enc_layers=1,
            dropout_rate=0.0,
            max_beats=constants.CHUNK_ALIGN,
            audio_feature_type=ddcl_constants.AUDIO_FEATURE_MERT,
            n_frames_per_beat=constants.CNN_FRAMES_OUT,
            feature_dim=8,
            difficulty_source=constants.DIFFICULTY_SOURCE_DENSITY,
            use_nps_condition=True,
        )
        self.assertTrue(model.use_nps_condition)
        self.assertIsNotNone(model.nps_proj)
        inputs = {
            "audio": torch.zeros(
                (1, constants.CHUNK_ALIGN, constants.CNN_FRAMES_OUT, 8),
                dtype=torch.float32,
            ),
            "bpm": torch.tensor([[150.0]], dtype=torch.float32),
            "difficulty": torch.tensor([[2.0]], dtype=torch.float32),
        }
        model.eval()
        with torch.no_grad():
            out_a = model(inputs).numpy()
        self.assertEqual(out_a.shape, (1, constants.CHUNK_ALIGN, constants.N_SLOTS))
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt = trainers.save_checkpoint(model, pathlib.Path(tmpdir) / "nps.pt")
            reloaded = trainers.load_placement_model(ckpt, device="cpu")
            self.assertTrue(reloaded.use_nps_condition)
            with torch.no_grad():
                out_b = reloaded(inputs).numpy()
        np.testing.assert_allclose(out_a, out_b, atol=1e-6)

    def test_beat_aux_head_zero_init_aux_loss_and_roundtrip(self) -> None:
        n_beats = constants.CHUNK_ALIGN
        feature_dim = 16
        runtime.set_seed(0)
        base_model = models.build_itgpt_placement_model(
            d_model=32,
            n_heads=4,
            n_enc_layers=1,
            dropout_rate=0.0,
            max_beats=n_beats,
            audio_feature_type=ddcl_constants.AUDIO_FEATURE_MERT,
            n_frames_per_beat=constants.CNN_FRAMES_OUT,
            feature_dim=feature_dim,
            beat_aux_head=False,
        )
        runtime.set_seed(0)
        aux_model = models.build_itgpt_placement_model(
            d_model=32,
            n_heads=4,
            n_enc_layers=1,
            dropout_rate=0.0,
            max_beats=n_beats,
            audio_feature_type=ddcl_constants.AUDIO_FEATURE_MERT,
            n_frames_per_beat=constants.CNN_FRAMES_OUT,
            feature_dim=feature_dim,
            beat_aux_head=True,
        )
        self.assertFalse(base_model.beat_aux_head)
        self.assertIsNone(base_model.beat_aux)
        self.assertTrue(aux_model.beat_aux_head)
        self.assertIsNotNone(aux_model.beat_aux)

        aux_state = aux_model.state_dict()
        for key, value in base_model.state_dict().items():
            aux_state[key] = value.clone()
        aux_model.load_state_dict(aux_state)

        rng = np.random.default_rng(73)
        inputs = {
            "audio": torch.as_tensor(
                rng.standard_normal(
                    (1, n_beats, constants.CNN_FRAMES_OUT, feature_dim)
                ).astype(np.float32)
            ),
            "bpm": torch.tensor([[145.0]], dtype=torch.float32),
            "difficulty": torch.tensor([[2.0]], dtype=torch.float32),
        }
        labels = torch.zeros((1, n_beats, constants.N_SLOTS), dtype=torch.float32)
        labels[0, 0, 0] = 1.0
        labels[0, 0, 24] = 1.0
        labels[0, 3, 12] = 1.0
        mask = torch.ones((1, n_beats), dtype=torch.float32)
        mask[0, -8:] = 0.0

        base_model.eval()
        aux_model.eval()
        with torch.no_grad():
            base_logits_init = base_model.forward_logits(inputs).numpy()
            aux_logits_init, aux_preds = aux_model.forward_with_aux(inputs)
            aux_probs_init = aux_model(inputs).numpy()
        self.assertEqual(aux_logits_init.shape, (1, n_beats, constants.N_SLOTS))
        self.assertEqual(aux_probs_init.shape, (1, n_beats, constants.N_SLOTS))
        self.assertIsNotNone(aux_preds)
        assert aux_preds is not None
        self.assertEqual(tuple(aux_preds["beat_active_logit"].shape), (1, n_beats))
        self.assertEqual(tuple(aux_preds["beat_count_pred"].shape), (1, n_beats))
        np.testing.assert_allclose(aux_logits_init.numpy(), base_logits_init, atol=1e-6)

        # Verify masked aux loss ignores masked-out beats and trains all aux layers.
        run_cfg = config.ItgptRunConfig(learning_rate=1e-2, weight_decay=0.0)
        optimizer_aux = trainers.build_optimizer(aux_model, run_cfg)
        criterion = models.ItgptPlacementLoss()
        aux_model.train()
        optimizer_aux.zero_grad()
        logits, aux_out = aux_model.forward_with_aux(inputs)
        assert aux_out is not None
        aux_loss = models.compute_beat_aux_loss(
            aux_out["beat_active_logit"],
            aux_out["beat_count_pred"],
            labels,
            mask,
        )
        self.assertTrue(np.isfinite(float(aux_loss.detach().item())))
        self.assertGreater(float(aux_loss.detach().item()), 0.0)
        total_loss = criterion(logits, labels, mask) + 0.15 * aux_loss
        total_loss.backward()
        assert aux_model.beat_aux is not None
        assert aux_model.beat_aux.aux_proj.weight.grad is not None
        assert aux_model.beat_aux.gate_proj.weight.grad is not None
        assert aux_model.beat_aux.hidden.weight.grad is not None
        self.assertGreater(
            float(aux_model.beat_aux.aux_proj.weight.grad.abs().max().item()), 0.0
        )
        self.assertGreater(
            float(aux_model.beat_aux.gate_proj.weight.grad.abs().max().item()), 0.0
        )
        self.assertGreater(
            float(aux_model.beat_aux.hidden.weight.grad.abs().max().item()), 0.0
        )
        optimizer_aux.step()

        aux_model.eval()
        with torch.no_grad():
            aux_out_after = aux_model.forward_logits(inputs).numpy()
        self.assertGreater(
            float(np.max(np.abs(aux_out_after - base_logits_init))),
            1e-4,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt = trainers.save_checkpoint(
                aux_model, pathlib.Path(tmpdir) / "beat_aux.pt"
            )
            reloaded = trainers.load_placement_model(ckpt, device="cpu")
            self.assertTrue(reloaded.beat_aux_head)
            self.assertIsNotNone(reloaded.beat_aux)
            with torch.no_grad():
                reloaded_out = reloaded.forward_logits(inputs).numpy()
        np.testing.assert_allclose(aux_out_after, reloaded_out, atol=1e-5)

    def test_placement_loss_dice_term_masking_gradients_and_validation(self) -> None:
        with self.assertRaises(ValueError):
            models.ItgptPlacementLoss(dice_loss_weight=-0.1)

        rng = np.random.default_rng(123)
        logits_data = rng.standard_normal((1, 4, constants.N_SLOTS)).astype(np.float32)
        targets = torch.zeros((1, 4, constants.N_SLOTS), dtype=torch.float32)
        targets[0, 0, 0] = 1.0
        targets[0, 1, 24] = 1.0
        mask = torch.tensor([[1.0, 1.0, 0.0, 0.0]], dtype=torch.float32)

        # dice_loss_weight=0.0 matches existing loss bit-for-bit.
        base_loss_fn = models.ItgptPlacementLoss(label_smoothing=0.01)
        zero_dice_fn = models.ItgptPlacementLoss(
            label_smoothing=0.01, dice_loss_weight=0.0
        )
        logits_a = torch.tensor(logits_data, requires_grad=True)
        logits_b = torch.tensor(logits_data, requires_grad=True)
        loss_a = base_loss_fn(logits_a, targets, mask)
        loss_b = zero_dice_fn(logits_b, targets, mask)
        self.assertTrue(torch.equal(loss_a, loss_b))
        loss_a.backward()
        loss_b.backward()
        assert logits_a.grad is not None and logits_b.grad is not None
        self.assertTrue(torch.equal(logits_a.grad, logits_b.grad))

        # dice_loss_weight > 0 changes loss and gradients cleanly using raw targets.
        dice_weight = 0.25
        dice_loss_fn = models.ItgptPlacementLoss(
            label_smoothing=0.01, dice_loss_weight=dice_weight
        )
        logits_c = torch.tensor(logits_data, requires_grad=True)
        loss_c = dice_loss_fn(logits_c, targets, mask)
        with torch.no_grad():
            probs = torch.sigmoid(torch.tensor(logits_data))
            slot_mask = mask.unsqueeze(-1)
            eps = 1e-6
            expected_dice = 1.0 - (
                2.0 * torch.sum(probs * targets * slot_mask) + eps
            ) / (torch.sum(probs * slot_mask) + torch.sum(targets * slot_mask) + eps)
            expected_total = loss_a.detach() + dice_weight * expected_dice
        self.assertAlmostEqual(
            float(loss_c.detach().item()), float(expected_total.item()), places=6
        )
        self.assertGreater(float(loss_c.detach().item()), float(loss_a.detach().item()))
        loss_c.backward()
        assert logits_c.grad is not None
        self.assertTrue(torch.isfinite(logits_c.grad).all())
        self.assertGreater(
            float(torch.max(torch.abs(logits_c.grad - logits_a.grad)).item()), 0.0
        )
        # Padded beats must have exact zero gradients.
        self.assertEqual(
            float(torch.max(torch.abs(logits_c.grad[:, 2:, :])).item()), 0.0
        )

        # Mutating logits and targets on padded beats must not affect loss or real grads.
        logits_pad_mut = torch.tensor(logits_data, requires_grad=True)
        with torch.no_grad():
            logits_pad_mut[:, 2:, :] = 10.0
        targets_pad_mut = targets.clone()
        targets_pad_mut[:, 2:, :] = 1.0
        loss_pad_mut = dice_loss_fn(logits_pad_mut, targets_pad_mut, mask)
        self.assertAlmostEqual(
            float(loss_pad_mut.detach().item()), float(loss_c.detach().item()), places=6
        )
        loss_pad_mut.backward()
        assert logits_pad_mut.grad is not None
        np.testing.assert_allclose(
            logits_pad_mut.grad[:, :2, :].numpy(),
            logits_c.grad[:, :2, :].numpy(),
            atol=1e-6,
        )
        self.assertEqual(
            float(torch.max(torch.abs(logits_pad_mut.grad[:, 2:, :])).item()), 0.0
        )

    def test_placement_loss_bce_pos_weight_baseline_shift_and_validation(self) -> None:
        with self.assertRaises(ValueError):
            models.ItgptPlacementLoss(bce_pos_weight=0.0)
        with self.assertRaises(ValueError):
            models.ItgptPlacementLoss(bce_pos_weight=-1.0)

        rng = np.random.default_rng(77)
        logits_data = rng.standard_normal((1, 4, constants.N_SLOTS)).astype(np.float32)
        targets = torch.zeros((1, 4, constants.N_SLOTS), dtype=torch.float32)
        targets[0, 0, 0] = 1.0
        targets[0, 1, 24] = 1.0
        mask = torch.tensor([[1.0, 1.0, 0.0, 0.0]], dtype=torch.float32)

        base_fn = models.ItgptPlacementLoss(label_smoothing=0.01, dice_loss_weight=0.05)
        unit_pw_fn = models.ItgptPlacementLoss(
            label_smoothing=0.01, dice_loss_weight=0.05, bce_pos_weight=1.0
        )
        logits_a = torch.tensor(logits_data, requires_grad=True)
        logits_b = torch.tensor(logits_data, requires_grad=True)
        loss_a = base_fn(logits_a, targets, mask)
        loss_b = unit_pw_fn(logits_b, targets, mask)
        self.assertTrue(torch.equal(loss_a, loss_b))
        loss_a.backward()
        loss_b.backward()
        assert logits_a.grad is not None and logits_b.grad is not None
        self.assertTrue(torch.equal(logits_a.grad, logits_b.grad))

        # bce_pos_weight > 1.0 increases positive-slot loss and pulls logits upward.
        pw_fn = models.ItgptPlacementLoss(
            label_smoothing=0.01, dice_loss_weight=0.05, bce_pos_weight=1.35
        )
        logits_c = torch.tensor(logits_data, requires_grad=True)
        loss_c = pw_fn(logits_c, targets, mask)
        self.assertGreater(float(loss_c.detach().item()), float(loss_a.detach().item()))
        loss_c.backward()
        assert logits_c.grad is not None
        self.assertLess(float(logits_c.grad[0, 0, 0]), float(logits_a.grad[0, 0, 0]))
        # Negative slots have identical BCE gradient (only Dice denom/target term is unchanged).
        # Compare gradient-descent updated logits on positive slots.
        updated_a = logits_data - 0.1 * logits_a.grad.numpy()
        updated_c = logits_data - 0.1 * logits_c.grad.numpy()
        self.assertGreater(float(updated_c[0, 0, 0]), float(updated_a[0, 0, 0]))
        self.assertGreater(float(updated_c[0, 1, 24]), float(updated_a[0, 1, 24]))

        # On all-zero targets without label smoothing, pos_weight > 1.0 leaves loss unchanged.
        neg_base = models.ItgptPlacementLoss(label_smoothing=0.0, dice_loss_weight=0.0)
        neg_pw = models.ItgptPlacementLoss(
            label_smoothing=0.0, dice_loss_weight=0.0, bce_pos_weight=1.35
        )
        zero_targets = torch.zeros_like(targets)
        self.assertTrue(
            torch.equal(
                neg_base(torch.tensor(logits_data), zero_targets, mask),
                neg_pw(torch.tensor(logits_data), zero_targets, mask),
            )
        )

    def test_bernoulli_rdrop_kl_symmetry_stability_and_masking(self) -> None:
        torch.manual_seed(41)
        z1 = torch.randn(2, 8, constants.N_SLOTS, requires_grad=True)
        z2 = torch.randn(2, 8, constants.N_SLOTS, requires_grad=True)
        mask = torch.zeros((2, 8), dtype=torch.float32)
        mask[0, :5] = 1.0
        mask[1, :3] = 1.0

        kl_zero = models.compute_bernoulli_rdrop_kl(z1, z1.detach(), mask)
        self.assertAlmostEqual(float(kl_zero.item()), 0.0, places=6)

        kl_12 = models.compute_bernoulli_rdrop_kl(z1, z2, mask)
        kl_21 = models.compute_bernoulli_rdrop_kl(z2, z1, mask)
        self.assertGreater(float(kl_12.item()), 0.0)
        torch.testing.assert_close(kl_12, kl_21, atol=1e-6, rtol=1e-6)
        kl_12.backward()
        assert z1.grad is not None and z2.grad is not None
        self.assertTrue(torch.isfinite(z1.grad).all())
        self.assertTrue(torch.isfinite(z2.grad).all())

        # Masked tail beats do not contribute.
        z1_tail = z1.detach().clone()
        z2_tail = z2.detach().clone()
        z1_tail[0, 5:] += 20.0
        z2_tail[1, 3:] -= 20.0
        kl_tail = models.compute_bernoulli_rdrop_kl(z1_tail, z2_tail, mask)
        torch.testing.assert_close(kl_12.detach(), kl_tail, atol=1e-6, rtol=1e-6)

        # Extreme logits stay finite.
        ext1 = torch.full((1, 4, constants.N_SLOTS), 30.0, requires_grad=True)
        ext2 = torch.full((1, 4, constants.N_SLOTS), -30.0, requires_grad=True)
        kl_ext = models.compute_bernoulli_rdrop_kl(
            ext1, ext2, torch.ones((1, 4)), normalize_by_valid_beats=True
        )
        self.assertTrue(torch.isfinite(kl_ext).item())
        kl_ext.backward()
        assert ext1.grad is not None and ext2.grad is not None
        self.assertTrue(torch.isfinite(ext1.grad).all())
        self.assertTrue(torch.isfinite(ext2.grad).all())


if __name__ == "__main__":
    unittest.main()
