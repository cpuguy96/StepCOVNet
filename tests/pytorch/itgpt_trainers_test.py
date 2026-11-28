"""Tests for PyTorch ITGPT trainers, evaluation, and CLI entrypoints."""

from __future__ import annotations

import json
import pathlib
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

from stepcovnet.dataset_prep import models as prep_models
from stepcovnet.ddcl import constants as ddcl_constants
from stepcovnet.ddcl import datasets as ddcl_datasets
from stepcovnet.itgpt import config, constants, evaluation, models, trainers
from stepcovnet.itgpt import eval as eval_cli
from stepcovnet.itgpt import train as train_cli


def _synthetic_chart(
    *,
    n_beats: int = 8,
    mert: bool = False,
    feature_dim: int = 16,
) -> ddcl_datasets.DdclChart:
    """Return a short in-memory chart with BPM segments and onset times."""
    slots = np.zeros((n_beats, constants.N_SLOTS), dtype=np.float32)
    slots[0, 0] = 1.0
    slots[2, 24] = 1.0
    stream = np.zeros((n_beats, ddcl_constants.STREAM_DIM), dtype=np.float32)
    stream[:, 1] = 120.0
    rng = np.random.default_rng(7)
    audio_shape = (
        (n_beats, 24, feature_dim)
        if mert
        else (
            n_beats,
            constants.N_FRAMES_PER_BEAT,
            constants.N_MELS,
            constants.N_CHANNELS,
        )
    )
    beat_audio = rng.standard_normal(audio_shape).astype(np.float32)
    return ddcl_datasets.DdclChart(
        song_key="bundle/song",
        difficulty="easy",
        meter=5,
        beat_audio=beat_audio,
        stream=stream,
        slots=slots,
        memlen=0,
        offset_sec=0.0,
        bpm_segments=(prep_models.BpmSegment(start_beat=0.0, bpm=120.0),),
        gt_times_sec=np.array([0.0, 1.25], dtype=np.float64),
    )


class PyTorchItgptTrainersTest(unittest.TestCase):
    def test_tensorboard_run_log_dir_is_timestamped_child(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = trainers.tensorboard_run_log_dir(tmpdir, "smoke")
            self.assertEqual(log_dir.parent, pathlib.Path(tmpdir) / "logs")
            self.assertTrue(log_dir.name.endswith("-smoke"))
            self.assertTrue(log_dir.is_dir())

    def test_train_placement_logmel_writes_artifacts_and_roundtrips(self):
        chart = _synthetic_chart(mert=False)
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
            with mock.patch(
                "stepcovnet.itgpt.trainers.datasets.load_split_charts",
                return_value=[chart],
            ):
                trained = trainers.train_placement(experiment, device="cpu")

            model_dir = pathlib.Path(tmpdir) / "models"
            saved = model_dir / "smoke.pt"
            best = model_dir / "best.pt"
            last = model_dir / "last.pt"
            eval_path = model_dir / "eval_val_slot48.json"
            best_eval = model_dir / "eval_val_slot48_best.json"
            self.assertTrue(saved.is_file())
            self.assertTrue(best.is_file())
            self.assertTrue(last.is_file())
            self.assertFalse((model_dir / trainers.BACKUP_DIRNAME).exists())

            payload = json.loads(eval_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["weights"], "last")
            self.assertEqual(payload["framework"], "pytorch")
            best_payload = json.loads(best_eval.read_text(encoding="utf-8"))
            self.assertEqual(best_payload["weights"], "best")

            log_root = pathlib.Path(tmpdir) / "callbacks" / "logs"
            run_dirs = [path for path in log_root.iterdir() if path.is_dir()]
            self.assertEqual(len(run_dirs), 1)
            self.assertTrue(run_dirs[0].name.endswith("-smoke"))

            # Verify save/load checkpoint reproduces exact same slot predictions.
            reloaded = trainers.load_placement_model(saved, device="cpu")
            p1 = evaluation.predict_chart_slots(trained, chart, max_beats=64)
            p2 = evaluation.predict_chart_slots(reloaded, chart, max_beats=64)
            self.assertTrue(np.allclose(p1, p2, atol=1e-6))

    def test_train_placement_mert_ema_cosine_grad_accum_and_best_f1(self):
        chart = _synthetic_chart(mert=True, feature_dim=16)
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
                    epoch=2,
                    steps_per_epoch=2,
                    validation_steps=1,
                    weight_decay=1e-2,
                    lr_schedule="cosine",
                    warmup_epochs=1,
                    min_learning_rate=1e-6,
                    gradient_accumulation_steps=2,
                    use_ema=True,
                    ema_momentum=0.995,
                    eval_slot_f1_every=1,
                    model_output_dir=str(pathlib.Path(tmpdir) / "models"),
                    callback_root_dir="",
                    model_name="mert_ema",
                    resume=False,
                ),
            )
            with mock.patch(
                "stepcovnet.itgpt.trainers.datasets.load_split_charts",
                return_value=[chart],
            ):
                trainers.train_placement(experiment, device="cpu")

            model_dir = pathlib.Path(tmpdir) / "models"
            self.assertTrue((model_dir / "best_slot_f1.pt").is_file())
            best_f1_eval = model_dir / "eval_val_slot48_best_f1.json"
            self.assertTrue(best_f1_eval.is_file())
            payload = json.loads(best_f1_eval.read_text(encoding="utf-8"))
            self.assertEqual(payload["weights"], "best_slot_f1")

    def test_train_placement_resumes_from_backup(self):
        chart = _synthetic_chart(mert=True, feature_dim=16)
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = pathlib.Path(tmpdir) / "models"
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
                    dropout_rate=0.0,
                ),
                run=config.ItgptRunConfig(
                    epoch=2,
                    steps_per_epoch=1,
                    validation_steps=1,
                    use_ema=True,
                    model_output_dir=str(model_dir),
                    callback_root_dir="",
                    model_name="resume_test",
                    resume=True,
                ),
            )
            # Pre-seed a backup after epoch 1.
            seed_model = trainers.build_model(experiment, feature_dim=16)
            seed_opt = trainers.build_optimizer(seed_model, experiment.run)
            backup_path = model_dir / trainers.BACKUP_DIRNAME / trainers.BACKUP_FILENAME
            trainers.save_checkpoint(
                seed_model,
                backup_path,
                extra={
                    "completed_epoch": 1,
                    "global_step": 1,
                    "optimizer_state_dict": seed_opt.state_dict(),
                    "best_val_loss": 999.0,
                },
            )
            with mock.patch(
                "stepcovnet.itgpt.trainers.datasets.load_split_charts",
                return_value=[chart],
            ):
                trainers.train_placement(experiment, device="cpu")
            self.assertTrue((model_dir / "resume_test.pt").is_file())
            self.assertFalse((model_dir / trainers.BACKUP_DIRNAME).exists())

    def test_train_placement_rejects_empty_splits(self):
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
                trainers.train_placement(experiment, device="cpu")
            with (
                mock.patch(
                    "stepcovnet.itgpt.trainers.datasets.load_split_charts",
                    side_effect=[[_synthetic_chart()], []],
                ),
                self.assertRaises(ValueError),
            ):
                trainers.train_placement(experiment, device="cpu")

    def test_grounding_evaluation_and_cli_entrypoints(self):
        chart = _synthetic_chart(mert=True, feature_dim=16)
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = pathlib.Path(tmpdir) / "models"
            cfg_path = pathlib.Path(tmpdir) / "cfg.json"
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
                    dropout_rate=0.0,
                ),
                run=config.ItgptRunConfig(
                    epoch=1,
                    steps_per_epoch=1,
                    validation_steps=1,
                    model_output_dir=str(model_dir),
                    callback_root_dir="",
                    model_name="cli_smoke",
                    resume=False,
                ),
            )
            cfg_path.write_text(
                json.dumps(experiment.as_dict(), indent=2),
                encoding="utf-8",
            )
            with mock.patch(
                "stepcovnet.itgpt.trainers.datasets.load_split_charts",
                return_value=[chart],
            ):
                rc = train_cli.main(["--config", str(cfg_path), "--device", "cpu"])
            self.assertEqual(rc, 0)

            with mock.patch(
                "stepcovnet.itgpt.eval.datasets.load_split_charts",
                return_value=[chart],
            ):
                eval_out = pathlib.Path(tmpdir) / "eval_out.json"
                rc_eval = eval_cli.main(
                    [
                        "--config",
                        str(cfg_path),
                        "--device",
                        "cpu",
                        "--output",
                        str(eval_out),
                    ]
                )
                self.assertEqual(rc_eval, 0)
                eval_payload = json.loads(eval_out.read_text(encoding="utf-8"))
                self.assertEqual(eval_payload["framework"], "pytorch")

                grounding_out = pathlib.Path(tmpdir) / "grounding_out.json"
                rc_ground = eval_cli.main(
                    [
                        "--config",
                        str(cfg_path),
                        "--device",
                        "cpu",
                        "--diagnose_grounding",
                        "--output",
                        str(grounding_out),
                    ]
                )
                self.assertEqual(rc_ground, 0)
                grounding_payload = json.loads(
                    grounding_out.read_text(encoding="utf-8")
                )
                self.assertEqual(grounding_payload["framework"], "pytorch")
                self.assertIn("conditions", grounding_payload)
                self.assertIn("shuffle", grounding_payload["conditions"])

    def test_optimizer_uses_adam_when_weight_decay_zero(self):
        model = models.build_itgpt_placement_model(
            d_model=32,
            n_heads=4,
            n_enc_layers=1,
            audio_feature_type="mert",
            n_frames_per_beat=24,
            feature_dim=16,
        )
        run_cfg = config.ItgptRunConfig(weight_decay=0.0)
        opt = trainers.build_optimizer(model, run_cfg)
        self.assertIsInstance(opt, torch.optim.Adam)
        self.assertNotIsInstance(opt, torch.optim.AdamW)
