"""Tests for PyTorch ITGPT placement trainers."""

from __future__ import annotations

import json
import pathlib
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

from stepcovnet.ddcl import constants as ddcl_constants
from stepcovnet.ddcl import datasets as ddcl_datasets
from stepcovnet.itgpt import config, constants, trainers


def _synthetic_chart(*, n_beats: int = 6) -> ddcl_datasets.DdclChart:
    """Return a short in-memory chart for trainer tests."""
    slots = np.zeros((n_beats, constants.N_SLOTS), dtype=np.float32)
    slots[0, 0] = 1.0
    stream = np.zeros((n_beats, ddcl_constants.STREAM_DIM), dtype=np.float32)
    stream[:, 1] = 140.0
    return ddcl_datasets.DdclChart(
        song_key="bundle/song",
        difficulty="easy",
        meter=5,
        beat_audio=np.zeros(
            (
                n_beats,
                constants.N_FRAMES_PER_BEAT,
                constants.N_MELS,
                constants.N_CHANNELS,
            ),
            dtype=np.float32,
        ),
        stream=stream,
        slots=slots,
        memlen=0,
    )


def _mock_report(f1_at_05: float = 0.55, f1_max: float = 0.60) -> mock.Mock:
    report = mock.Mock()
    report.f1_at_05 = f1_at_05
    report.f1_max = f1_max
    report.best_threshold = 0.45
    report.as_dict.return_value = {"f1_at_05": f1_at_05, "f1_max": f1_max}
    return report


class ItgptTrainersTest(unittest.TestCase):
    def test_tensorboard_run_log_dir_is_timestamped_child(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = trainers.tensorboard_run_log_dir(tmpdir, "smoke")
            self.assertEqual(log_dir.parent, pathlib.Path(tmpdir) / "logs")
            self.assertTrue(log_dir.name.endswith("-smoke"))
            self.assertTrue(log_dir.is_dir())

    def test_train_placement_writes_artifacts(self) -> None:
        chart = _synthetic_chart()
        report = _mock_report(0.1, 0.2)
        with tempfile.TemporaryDirectory() as tmpdir:
            experiment = config.ItgptExperimentConfig(
                dataset=config.ItgptDatasetConfig(
                    training_index_path=str(pathlib.Path(tmpdir) / "missing.json"),
                    max_beats=constants.CHUNK_ALIGN,
                ),
                model=config.ItgptModelConfig(
                    d_model=32,
                    n_heads=4,
                    n_enc_layers=1,
                    cnn_hidden=8,
                    dropout_rate=0.0,
                ),
                run=config.ItgptRunConfig(
                    epoch=1,
                    steps_per_epoch=1,
                    validation_steps=1,
                    model_output_dir=str(pathlib.Path(tmpdir) / "models"),
                    callback_root_dir=str(pathlib.Path(tmpdir) / "callbacks"),
                    model_name="smoke",
                    resume=False,
                ),
            )
            with (
                mock.patch(
                    "stepcovnet.pytorch.itgpt.trainers.datasets.load_split_charts",
                    return_value=[chart],
                ),
                mock.patch(
                    "stepcovnet.pytorch.itgpt.trainers.evaluation.evaluate_slot48",
                    return_value=report,
                ) as mock_eval,
                mock.patch(
                    "stepcovnet.pytorch.itgpt.trainers.evaluation.report_as_dict",
                    side_effect=lambda r, *, weights: {
                        **r.as_dict(),
                        "weights": weights,
                    },
                ),
            ):
                trainers.train_placement(experiment, device="cpu")
            saved = pathlib.Path(tmpdir) / "models" / "smoke.pt"
            best = pathlib.Path(tmpdir) / "models" / "best.pt"
            last = pathlib.Path(tmpdir) / "models" / "last.pt"
            eval_path = pathlib.Path(tmpdir) / "models" / "eval_val_slot48.json"
            best_eval = pathlib.Path(tmpdir) / "models" / "eval_val_slot48_best.json"
            self.assertTrue(saved.is_file())
            self.assertTrue(best.is_file())
            self.assertTrue(last.is_file())
            payload = json.loads(eval_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["f1_max"], 0.2)
            self.assertEqual(payload["weights"], "last")
            best_payload = json.loads(best_eval.read_text(encoding="utf-8"))
            self.assertEqual(best_payload["weights"], "best")
            self.assertEqual(mock_eval.call_count, 2)
            log_root = pathlib.Path(tmpdir) / "callbacks" / "logs"
            run_dirs = [path for path in log_root.iterdir() if path.is_dir()]
            self.assertEqual(len(run_dirs), 1)
            self.assertTrue(run_dirs[0].name.endswith("-smoke"))

    def test_train_placement_rejects_empty_split(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            experiment = config.ItgptExperimentConfig(
                dataset=config.ItgptDatasetConfig(
                    training_index_path=str(pathlib.Path(tmpdir) / "missing.json"),
                    max_beats=constants.CHUNK_ALIGN,
                ),
            )
            with (
                mock.patch(
                    "stepcovnet.pytorch.itgpt.trainers.datasets.load_split_charts",
                    return_value=[],
                ),
                self.assertRaises(ValueError),
            ):
                trainers.train_placement(experiment, device="cpu")
            with (
                mock.patch(
                    "stepcovnet.pytorch.itgpt.trainers.datasets.load_split_charts",
                    side_effect=[[_synthetic_chart()], []],
                ),
                self.assertRaises(ValueError),
            ):
                trainers.train_placement(experiment, device="cpu")

    def test_train_placement_mert_and_periodic_slot_f1_checkpoint(self) -> None:
        n_beats = 6
        slots = np.zeros((n_beats, constants.N_SLOTS), dtype=np.float32)
        slots[0, 0] = 1.0
        stream = np.zeros((n_beats, ddcl_constants.STREAM_DIM), dtype=np.float32)
        stream[:, 1] = 140.0
        mert_chart = ddcl_datasets.DdclChart(
            song_key="bundle/mert",
            difficulty="easy",
            meter=5,
            beat_audio=np.zeros((n_beats, 24, 16), dtype=np.float32),
            stream=stream,
            slots=slots,
            memlen=0,
        )
        report = _mock_report(0.55, 0.60)
        with tempfile.TemporaryDirectory() as tmpdir:
            experiment = config.ItgptExperimentConfig(
                dataset=config.ItgptDatasetConfig(
                    training_index_path=str(pathlib.Path(tmpdir) / "missing.json"),
                    max_beats=constants.CHUNK_ALIGN,
                    audio_feature_type="mert",
                    n_frames_per_beat=24,
                ),
                model=config.ItgptModelConfig(
                    d_model=32,
                    n_heads=4,
                    n_enc_layers=1,
                    dropout_rate=0.1,
                ),
                run=config.ItgptRunConfig(
                    epoch=1,
                    steps_per_epoch=1,
                    validation_steps=1,
                    weight_decay=0.0,
                    eval_slot_f1_every=1,
                    model_output_dir=str(pathlib.Path(tmpdir) / "models"),
                    callback_root_dir="",
                    model_name="mert_smoke",
                    resume=False,
                ),
            )
            with (
                mock.patch(
                    "stepcovnet.pytorch.itgpt.trainers.datasets.load_split_charts",
                    return_value=[mert_chart],
                ),
                mock.patch(
                    "stepcovnet.pytorch.itgpt.trainers.evaluation.evaluate_slot48",
                    return_value=report,
                ) as mock_eval,
                mock.patch(
                    "stepcovnet.pytorch.itgpt.trainers.evaluation.report_as_dict",
                    side_effect=lambda r, *, weights: {
                        **r.as_dict(),
                        "weights": weights,
                    },
                ),
            ):
                model = trainers.train_placement(experiment, device="cpu")
            self.assertEqual(model.audio_feature_type, "mert")
            best_f1 = pathlib.Path(tmpdir) / "models" / "best_slot_f1.pt"
            best_f1_eval = (
                pathlib.Path(tmpdir) / "models" / "eval_val_slot48_best_f1.json"
            )
            self.assertTrue(best_f1.is_file())
            self.assertTrue(best_f1_eval.is_file())
            payload = json.loads(best_f1_eval.read_text(encoding="utf-8"))
            self.assertEqual(payload["weights"], "best_slot_f1")
            # 1 periodic eval + 3 final evals (last, best, best_slot_f1)
            self.assertEqual(mock_eval.call_count, 4)

    def test_learning_rate_for_step_constant_and_cosine(self) -> None:
        const_cfg = config.ItgptRunConfig(learning_rate=1e-4, lr_schedule="constant")
        self.assertEqual(trainers.learning_rate_for_step(const_cfg, 0), 1e-4)
        self.assertEqual(trainers.learning_rate_for_step(const_cfg, 100), 1e-4)

        no_warmup_cfg = config.ItgptRunConfig(
            epoch=10,
            steps_per_epoch=4,
            learning_rate=1e-4,
            min_learning_rate=1e-6,
            lr_schedule="cosine",
            warmup_epochs=0,
        )
        self.assertAlmostEqual(
            trainers.learning_rate_for_step(no_warmup_cfg, 0), 1e-4, places=8
        )
        self.assertAlmostEqual(
            trainers.learning_rate_for_step(no_warmup_cfg, 20), 5.05e-5, places=8
        )
        self.assertAlmostEqual(
            trainers.learning_rate_for_step(no_warmup_cfg, 40), 1e-6, places=8
        )

        with_warmup_cfg = config.ItgptRunConfig(
            epoch=10,
            steps_per_epoch=4,
            learning_rate=1e-4,
            min_learning_rate=1e-6,
            lr_schedule="cosine",
            warmup_epochs=2,
        )
        self.assertAlmostEqual(
            trainers.learning_rate_for_step(with_warmup_cfg, 0), 1e-6, places=8
        )
        self.assertAlmostEqual(
            trainers.learning_rate_for_step(with_warmup_cfg, 4), 5.05e-5, places=8
        )
        self.assertAlmostEqual(
            trainers.learning_rate_for_step(with_warmup_cfg, 8), 1e-4, places=8
        )
        self.assertAlmostEqual(
            trainers.learning_rate_for_step(with_warmup_cfg, 24), 5.05e-5, places=8
        )
        self.assertAlmostEqual(
            trainers.learning_rate_for_step(with_warmup_cfg, 40), 1e-6, places=8
        )

    def test_focal_and_label_smoothing_loss_math(self) -> None:
        y_true = torch.zeros((1, 4, constants.N_SLOTS), dtype=torch.float32)
        y_true[0, 0, 0] = 1.0
        y_true[0, 1, 12] = 1.0
        probs = torch.full((1, 4, constants.N_SLOTS), 0.1, dtype=torch.float32)
        probs[0, 0, 0] = 0.9
        probs[0, 1, 12] = 0.4
        logits = torch.logit(probs)
        mask = torch.tensor([[1.0, 1.0, 0.0, 0.0]], dtype=torch.float32)

        plain_loss = trainers.ItgptPlacementLoss(focal_gamma=0.0, label_smoothing=0.0)
        focal_loss = trainers.ItgptPlacementLoss(focal_gamma=1.0, label_smoothing=0.0)
        plain_val = float(plain_loss(logits, y_true, mask).item())
        focal_val = float(focal_loss(logits, y_true, mask).item())
        self.assertGreater(plain_val, 0.0)
        self.assertLess(focal_val, plain_val)

        # Under label_smoothing=0.02, calibrated 0.99/0.01 incurs lower loss
        # than over-confident 0.9999/1e-4.
        y2 = torch.zeros((1, 2, constants.N_SLOTS), dtype=torch.float32)
        y2[0, 0, 0] = 1.0
        m2 = torch.ones((1, 2), dtype=torch.float32)
        p_cal = torch.full((1, 2, constants.N_SLOTS), 0.01, dtype=torch.float32)
        p_cal[0, 0, 0] = 0.99
        p_over = torch.full((1, 2, constants.N_SLOTS), 1e-4, dtype=torch.float32)
        p_over[0, 0, 0] = 0.9999
        ls_loss = trainers.ItgptPlacementLoss(label_smoothing=0.02)
        cal_val = float(ls_loss(torch.logit(p_cal), y2, m2).item())
        over_val = float(ls_loss(torch.logit(p_over), y2, m2).item())
        self.assertGreater(over_val, cal_val)

    def test_train_placement_with_cosine_grad_accum_ema_and_label_smoothing(
        self,
    ) -> None:
        chart = _synthetic_chart()
        report = _mock_report(0.5, 0.55)
        with tempfile.TemporaryDirectory() as tmpdir:
            experiment = config.ItgptExperimentConfig(
                dataset=config.ItgptDatasetConfig(
                    training_index_path=str(pathlib.Path(tmpdir) / "missing.json"),
                    max_beats=constants.CHUNK_ALIGN,
                ),
                model=config.ItgptModelConfig(
                    d_model=32,
                    n_heads=4,
                    n_enc_layers=1,
                    cnn_hidden=8,
                    dropout_rate=0.0,
                ),
                run=config.ItgptRunConfig(
                    epoch=2,
                    steps_per_epoch=4,
                    validation_steps=1,
                    learning_rate=1e-4,
                    min_learning_rate=1e-6,
                    lr_schedule="cosine",
                    warmup_epochs=1,
                    focal_gamma=1.0,
                    grid_weight_24th=2.0,
                    label_smoothing=0.02,
                    gradient_accumulation_steps=2,
                    use_ema=True,
                    ema_momentum=0.995,
                    eval_slot_f1_every=1,
                    model_output_dir=str(pathlib.Path(tmpdir) / "models"),
                    callback_root_dir="",
                    model_name="full_smoke",
                    resume=False,
                ),
            )
            with (
                mock.patch(
                    "stepcovnet.pytorch.itgpt.trainers.datasets.load_split_charts",
                    return_value=[chart],
                ),
                mock.patch(
                    "stepcovnet.pytorch.itgpt.trainers.evaluation.evaluate_slot48",
                    return_value=report,
                ),
                mock.patch(
                    "stepcovnet.pytorch.itgpt.trainers.evaluation.report_as_dict",
                    side_effect=lambda r, *, weights: {
                        **r.as_dict(),
                        "weights": weights,
                    },
                ),
            ):
                trainers.train_placement(experiment, device="cpu")
            for name in ("full_smoke.pt", "best.pt", "best_slot_f1.pt"):
                ckpt = pathlib.Path(tmpdir) / "models" / name
                self.assertTrue(ckpt.is_file())
                reloaded = trainers.load_placement_model(ckpt, device="cpu")
                self.assertEqual(reloaded.d_model, 32)


if __name__ == "__main__":
    unittest.main()
