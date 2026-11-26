"""Tests for ITGPT placement trainers."""

from __future__ import annotations

import json
import pathlib
import tempfile
import unittest
from unittest import mock

import numpy as np

from stepcovnet.ddcl import constants as ddcl_constants
from stepcovnet.ddcl import datasets as ddcl_datasets
from stepcovnet.itgpt import config, constants, trainers


def _synthetic_chart(*, n_beats: int = 6) -> ddcl_datasets.DdclChart:
    """Return a short in-memory chart for trainer tests.

    Args:
        n_beats: Integer-beat length.

    Returns:
        DdclChart with BPM in stream column 1.
    """
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


class ItgptTrainersTest(unittest.TestCase):
    def test_tensorboard_run_log_dir_is_timestamped_child(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = trainers.tensorboard_run_log_dir(tmpdir, "smoke")
            self.assertEqual(log_dir.parent, pathlib.Path(tmpdir) / "logs")
            self.assertTrue(log_dir.name.endswith("-smoke"))
            self.assertTrue(log_dir.is_dir())

    def test_train_placement_writes_artifacts(self):
        chart = _synthetic_chart()
        report = mock.Mock()
        report.as_dict.return_value = {"f1_at_05": 0.1, "f1_max": 0.2}
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
                    "stepcovnet.itgpt.trainers.datasets.load_split_charts",
                    return_value=[chart],
                ),
                mock.patch(
                    "stepcovnet.itgpt.trainers.evaluation.evaluate_slot48",
                    return_value=report,
                ) as mock_eval,
            ):
                trainers.train_placement(experiment)
            saved = pathlib.Path(tmpdir) / "models" / "smoke.keras"
            best = pathlib.Path(tmpdir) / "models" / "best.keras"
            last = pathlib.Path(tmpdir) / "models" / "last.keras"
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

    def test_train_placement_rejects_empty_split(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            experiment = config.ItgptExperimentConfig(
                dataset=config.ItgptDatasetConfig(
                    training_index_path=str(pathlib.Path(tmpdir) / "missing.json"),
                    max_beats=constants.CHUNK_ALIGN,
                ),
            )
            with (
                mock.patch(
                    "stepcovnet.itgpt.trainers.datasets.load_split_charts",
                    return_value=[],
                ),
                self.assertRaises(ValueError),
            ):
                trainers.train_placement(experiment)
            with (
                mock.patch(
                    "stepcovnet.itgpt.trainers.datasets.load_split_charts",
                    side_effect=[[_synthetic_chart()], []],
                ),
                self.assertRaises(ValueError),
            ):
                trainers.train_placement(experiment)

    def test_train_placement_mert_and_periodic_slot_f1_checkpoint(self):
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
        report = mock.Mock()
        report.f1_at_05 = 0.55
        report.f1_max = 0.60
        report.best_threshold = 0.45
        report.as_dict.return_value = {"f1_at_05": 0.55, "f1_max": 0.60}
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
                    "stepcovnet.itgpt.trainers.datasets.load_split_charts",
                    return_value=[mert_chart],
                ),
                mock.patch(
                    "stepcovnet.itgpt.trainers.evaluation.evaluate_slot48",
                    return_value=report,
                ) as mock_eval,
            ):
                model = trainers.train_placement(experiment)
            self.assertEqual(model.optimizer.__class__.__name__, "Adam")
            best_f1 = pathlib.Path(tmpdir) / "models" / "best_slot_f1.keras"
            best_f1_eval = (
                pathlib.Path(tmpdir) / "models" / "eval_val_slot48_best_f1.json"
            )
            self.assertTrue(best_f1.is_file())
            self.assertTrue(best_f1_eval.is_file())
            payload = json.loads(best_f1_eval.read_text(encoding="utf-8"))
            self.assertEqual(payload["weights"], "best_slot_f1")
            # 1 periodic callback eval + 3 final evals (last, best, best_slot_f1)
            self.assertEqual(mock_eval.call_count, 4)

    def test_build_learning_rate_constant_and_cosine(self):
        const_lr = trainers.build_learning_rate(
            config.ItgptRunConfig(learning_rate=1e-4, lr_schedule="constant")
        )
        self.assertEqual(const_lr, 1e-4)

        no_warmup = trainers.build_learning_rate(
            config.ItgptRunConfig(
                epoch=10,
                steps_per_epoch=4,
                learning_rate=1e-4,
                min_learning_rate=1e-6,
                lr_schedule="cosine",
                warmup_epochs=0,
            )
        )
        self.assertAlmostEqual(float(no_warmup(0)), 1e-4, places=8)
        self.assertAlmostEqual(float(no_warmup(20)), 5.05e-5, places=8)
        self.assertAlmostEqual(float(no_warmup(40)), 1e-6, places=8)

        with_warmup = trainers.build_learning_rate(
            config.ItgptRunConfig(
                epoch=10,
                steps_per_epoch=4,
                learning_rate=1e-4,
                min_learning_rate=1e-6,
                lr_schedule="cosine",
                warmup_epochs=2,
            )
        )
        self.assertAlmostEqual(float(with_warmup(0)), 1e-6, places=8)
        self.assertAlmostEqual(float(with_warmup(4)), 5.05e-5, places=8)
        self.assertAlmostEqual(float(with_warmup(8)), 1e-4, places=8)
        self.assertAlmostEqual(float(with_warmup(24)), 5.05e-5, places=8)
        self.assertAlmostEqual(float(with_warmup(40)), 1e-6, places=8)

    def test_train_placement_cosine_schedule_saves_and_reloads(self):
        chart = _synthetic_chart()
        report = mock.Mock()
        report.as_dict.return_value = {"f1_at_05": 0.1, "f1_max": 0.2}
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
                    steps_per_epoch=2,
                    validation_steps=1,
                    learning_rate=1e-4,
                    min_learning_rate=1e-6,
                    lr_schedule="cosine",
                    warmup_epochs=1,
                    model_output_dir=str(pathlib.Path(tmpdir) / "models"),
                    callback_root_dir="",
                    model_name="cosine_smoke",
                    resume=False,
                ),
            )
            with (
                mock.patch(
                    "stepcovnet.itgpt.trainers.datasets.load_split_charts",
                    return_value=[chart],
                ),
                mock.patch(
                    "stepcovnet.itgpt.trainers.evaluation.evaluate_slot48",
                    return_value=report,
                ),
            ):
                model = trainers.train_placement(experiment)
            self.assertEqual(
                model.optimizer.get_config()["learning_rate"]["class_name"],
                "CosineDecay",
            )
            self.assertAlmostEqual(float(model.optimizer.learning_rate), 1e-6, places=7)
            reloaded = trainers.load_placement_model(
                pathlib.Path(tmpdir) / "models" / "cosine_smoke.keras"
            )
            self.assertEqual(
                reloaded.optimizer.get_config()["learning_rate"]["class_name"],
                "CosineDecay",
            )

    def test_focal_loss_equivalence_modulation_and_reload(self):
        y_true = np.zeros((1, 4, constants.N_SLOTS), dtype=np.float32)
        y_true[0, 0, 0] = 1.0
        y_true[0, 1, 12] = 1.0
        y_pred = np.full((1, 4, constants.N_SLOTS), 0.1, dtype=np.float32)
        y_pred[0, 0, 0] = 0.9
        y_pred[0, 1, 12] = 0.4
        mask = np.array([[1.0, 1.0, 0.0, 0.0]], dtype=np.float32)

        plain_per_beat = np.asarray(
            trainers.placement_binary_crossentropy(y_true, y_pred)
        )
        zero_focal = trainers.ItgptPlacementLoss(focal_gamma=0.0)
        zero_per_beat = np.asarray(zero_focal.call(y_true, y_pred))
        np.testing.assert_allclose(zero_per_beat, plain_per_beat, rtol=1e-6, atol=1e-6)

        focal_1 = trainers.ItgptPlacementLoss(focal_gamma=1.0)
        focal_per_beat = np.asarray(focal_1.call(y_true, y_pred))
        self.assertTrue(np.all(focal_per_beat < plain_per_beat))
        # Harder beat (slot 12 at p=0.4) retains a larger fraction of its loss
        # than the easier beat (slot 0 at p=0.9).
        self.assertGreater(
            focal_per_beat[0, 1] / plain_per_beat[0, 1],
            focal_per_beat[0, 0] / plain_per_beat[0, 0],
        )
        # Masked reduction matches Keras sum_over_batch_size used by the plain loss.
        masked_loss = float(focal_1(y_true, y_pred, sample_weight=mask))
        self.assertAlmostEqual(
            masked_loss,
            float(np.mean(focal_per_beat * mask)),
            places=6,
        )
        self.assertAlmostEqual(
            float(zero_focal(y_true, y_pred, sample_weight=mask)),
            float(np.mean(plain_per_beat * mask)),
            places=6,
        )

        chart = _synthetic_chart()
        report = mock.Mock()
        report.as_dict.return_value = {"f1_at_05": 0.1, "f1_max": 0.2}
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
                    focal_gamma=1.0,
                    grid_weight_24th=2.0,
                    model_output_dir=str(pathlib.Path(tmpdir) / "models"),
                    callback_root_dir="",
                    model_name="focal_smoke",
                    resume=False,
                ),
            )
            with (
                mock.patch(
                    "stepcovnet.itgpt.trainers.datasets.load_split_charts",
                    return_value=[chart],
                ),
                mock.patch(
                    "stepcovnet.itgpt.trainers.evaluation.evaluate_slot48",
                    return_value=report,
                ),
            ):
                trainers.train_placement(experiment)
            reloaded = trainers.load_placement_model(
                pathlib.Path(tmpdir) / "models" / "focal_smoke.keras"
            )
            self.assertIsInstance(reloaded.loss, trainers.ItgptPlacementLoss)
            self.assertAlmostEqual(reloaded.loss.focal_gamma, 1.0)
            self.assertAlmostEqual(reloaded.loss.grid_weight_24th, 2.0)

    def test_train_placement_gradient_accumulation_saves_and_reloads(self):
        chart = _synthetic_chart()
        report = mock.Mock()
        report.as_dict.return_value = {"f1_at_05": 0.1, "f1_max": 0.2}
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
                    steps_per_epoch=4,
                    validation_steps=1,
                    learning_rate=1e-4,
                    min_learning_rate=1e-6,
                    lr_schedule="cosine",
                    warmup_epochs=0,
                    gradient_accumulation_steps=4,
                    model_output_dir=str(pathlib.Path(tmpdir) / "models"),
                    callback_root_dir="",
                    model_name="gradaccum_smoke",
                    resume=False,
                ),
            )
            with (
                mock.patch(
                    "stepcovnet.itgpt.trainers.datasets.load_split_charts",
                    return_value=[chart],
                ),
                mock.patch(
                    "stepcovnet.itgpt.trainers.evaluation.evaluate_slot48",
                    return_value=report,
                ),
            ):
                model = trainers.train_placement(experiment)
            self.assertEqual(model.optimizer.gradient_accumulation_steps, 4)
            self.assertEqual(int(model.optimizer.iterations), 1)
            reloaded = trainers.load_placement_model(
                pathlib.Path(tmpdir) / "models" / "gradaccum_smoke.keras"
            )
            self.assertEqual(reloaded.optimizer.gradient_accumulation_steps, 4)

    def test_train_placement_ema_swaps_and_reloads(self):
        chart = _synthetic_chart()
        report = mock.Mock()
        report.f1_at_05 = 0.5
        report.f1_max = 0.55
        report.best_threshold = 0.45
        report.as_dict.return_value = {"f1_at_05": 0.5, "f1_max": 0.55}
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
                    steps_per_epoch=2,
                    validation_steps=1,
                    learning_rate=1e-4,
                    min_learning_rate=1e-6,
                    lr_schedule="cosine",
                    warmup_epochs=0,
                    use_ema=True,
                    ema_momentum=0.995,
                    eval_slot_f1_every=1,
                    model_output_dir=str(pathlib.Path(tmpdir) / "models"),
                    callback_root_dir="",
                    model_name="ema_smoke",
                    resume=False,
                ),
            )
            with (
                mock.patch(
                    "stepcovnet.itgpt.trainers.datasets.load_split_charts",
                    return_value=[chart],
                ),
                mock.patch(
                    "stepcovnet.itgpt.trainers.evaluation.evaluate_slot48",
                    return_value=report,
                ),
            ):
                model = trainers.train_placement(experiment)
            self.assertTrue(model.optimizer.use_ema)
            self.assertAlmostEqual(float(model.optimizer.ema_momentum), 0.995)
            reloaded = trainers.load_placement_model(
                pathlib.Path(tmpdir) / "models" / "best_slot_f1.keras"
            )
            self.assertTrue(reloaded.optimizer.use_ema)


if __name__ == "__main__":
    unittest.main()
