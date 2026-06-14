import os
import tempfile
import unittest

import numpy as np
import pytest
import scipy.io.wavfile
import tensorflow as tf

from stepcovnet import constants
from stepcovnet.onset_events import config
from stepcovnet.onset_events import trainers


def _write_chart(path: str, step_lines: list[str]) -> None:
    with open(path, "w") as chart_file:
        chart_file.write("TITLE Test\nBPM 120.0\nNOTES\nDIFFICULTY Challenge\n")
        chart_file.write("".join(step_lines))


def _write_wav(path: str, samples: np.ndarray, sample_rate: int) -> None:
    int_samples = np.clip(samples * 32767.0, -32768, 32767).astype(np.int16)
    scipy.io.wavfile.write(path, sample_rate, int_samples)


def _write_valid_pair(
    directory: str,
    stem: str,
    *,
    step_lines: list[str] | None = None,
    duration_sec: float = 0.2,
) -> tuple[str, str]:
    if step_lines is None:
        step_lines = ["1000 0.05\n", "0100 0.10\n", "0010 0.15\n"]
    audio_path = os.path.join(directory, f"{stem}.wav")
    chart_path = os.path.join(directory, f"{stem}.txt")
    sr = constants.TARGET_SR
    n = max(1, int(duration_sec * sr))
    t = np.linspace(0, duration_sec, n, endpoint=False, dtype=np.float64)
    samples = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    _write_wav(audio_path, samples, sr)
    _write_chart(chart_path, step_lines)
    return audio_path, chart_path


def _tiny_experiment(
    data_dir: str,
    val_data_dir: str,
    model_output_dir: str,
) -> config.OnsetEventExperimentConfig:
    return config.OnsetEventExperimentConfig(
        dataset=config.OnsetEventDatasetConfig(
            data_dir=data_dir,
            val_data_dir=val_data_dir,
            test_data_dir="",
            batch_size=1,
            max_audio_seconds=0.25,
            n_max_onsets=8,
            max_steps_per_chart=1024,
            target_sample_rate=constants.TARGET_SR,
        ),
        model=config.OnsetEventModelConfig(
            max_audio_seconds=0.25,
            num_queries=3,
            embed_dim=16,
            decoder_layers=1,
            base_filters=16,
            encoder=config.OnsetEventEncoderConfig(initial_filters=8, depth=1),
        ),
        run=config.OnsetEventRunConfig(
            epochs=1,
            model_output_dir=model_output_dir,
            callback_root_dir="",
            seed=7,
        ),
    )


class TrainersTest(unittest.TestCase):
    def test_train_onset_event_writes_model_and_returns_history(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = os.path.join(tmpdir, "train")
            val_dir = os.path.join(tmpdir, "val")
            model_dir = os.path.join(tmpdir, "models")
            os.makedirs(data_dir)
            os.makedirs(val_dir)
            _write_valid_pair(data_dir, "song")
            _write_valid_pair(val_dir, "val_song")

            experiment = _tiny_experiment(data_dir, val_dir, model_dir)
            base_model, history = trainers.train_onset_event(
                experiment,
                take_count=1,
                val_take_count=1,
            )

            self.assertTrue(
                os.path.isfile(os.path.join(model_dir, "onset_event_model.keras"))
            )
            self.assertEqual(base_model.name, "onset_event_model")
            self.assertIn("loss", history.history)
            self.assertIn("event_onset_f1", history.history)
            self.assertIn("event_onset_f1_mingap", history.history)
            self.assertEqual(len(history.history["loss"]), 1)

    def test_train_onset_event_requires_output_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = os.path.join(tmpdir, "train")
            val_dir = os.path.join(tmpdir, "val")
            os.makedirs(data_dir)
            os.makedirs(val_dir)
            _write_valid_pair(data_dir, "song")
            _write_valid_pair(val_dir, "val_song")

            experiment = _tiny_experiment(data_dir, val_dir, model_output_dir="")
            with self.assertRaises(ValueError):
                trainers.train_onset_event(experiment, take_count=1, val_take_count=1)

    def test_event_onset_f1_metric_accumulates_counts(self):
        metric = trainers.EventOnsetF1Metric(
            tolerance_sec=0.02,
            confidence_threshold=0.5,
        )
        pred_times = np.array([[0.05, 0.10, 0.50]], dtype=np.float32)
        pred_confidence = np.array([[0.9, 0.9, 0.9]], dtype=np.float32)
        gt_times = np.array([[0.05, 0.10, 0.0, 0.0]], dtype=np.float32)
        gt_mask = np.array([[1.0, 1.0, 0.0, 0.0]], dtype=np.float32)

        metric.update_state(pred_times, pred_confidence, gt_times, gt_mask)
        f1 = float(metric.result().numpy())
        self.assertAlmostEqual(f1, 0.8, places=4)

    def test_event_onset_f1_metric_near_perfect_batch_k(self):
        metric = trainers.EventOnsetF1Metric(
            tolerance_sec=0.02,
            confidence_threshold=0.5,
        )
        k = 8
        pred_times = np.array(
            [[0.05, 0.10, 0.15, 0.20, 0.99, 0.98, 0.97, 0.96]],
            dtype=np.float32,
        )
        pred_confidence = np.array(
            [[0.95, 0.95, 0.95, 0.95, 0.1, 0.1, 0.1, 0.1]],
            dtype=np.float32,
        )
        gt_times = np.array(
            [[0.05, 0.10, 0.15, 0.20, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32
        )
        gt_mask = np.array([[1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32)

        metric.update_state(
            tf.constant(pred_times),
            tf.constant(pred_confidence),
            tf.constant(gt_times),
            tf.constant(gt_mask),
        )
        f1 = float(metric.result().numpy())
        self.assertAlmostEqual(f1, 1.0, places=4)

    def test_build_experiment_callbacks_empty_without_root(self):
        experiment = _tiny_experiment("train", "val", "out")
        callbacks = trainers._build_experiment_callbacks(
            run_config=experiment.run,
            experiment_name="test",
            monitor_metric="val_event_onset_f1",
            monitor_mode="max",
            experiment_config=experiment,
        )
        self.assertEqual(callbacks, [])

    def test_query_ref_normalized_from_batch_matches_sorted_gt(self):
        gt_times = np.array([0.3, 0.1, 0.0, 0.0], dtype=np.float64)
        gt_mask = np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float64)
        ref = trainers._query_ref_normalized_from_batch(
            gt_times,
            gt_mask,
            duration=1.0,
            num_queries=3,
        )
        self.assertEqual(len(ref), 3)
        np.testing.assert_allclose(ref[:2], [0.1, 0.3], rtol=0.0, atol=1e-6)
        self.assertGreater(ref[2], ref[1])

    def test_resolve_overfit_query_options(self):
        normal = config.OnsetEventRunConfig()
        self.assertEqual(
            trainers.resolve_overfit_query_options(normal),
            (False, True),
        )
        gt_learn = config.OnsetEventRunConfig(
            init_query_refs_from_gt=True,
            learn_time_delta=True,
        )
        self.assertEqual(
            trainers.resolve_overfit_query_options(gt_learn),
            (True, True),
        )
        frozen = config.OnsetEventRunConfig(
            init_query_refs_from_gt=False,
            learn_time_delta=False,
        )
        self.assertEqual(
            trainers.resolve_overfit_query_options(frozen),
            (False, False),
        )
        shortcuts = config.OnsetEventRunConfig(pipeline_check_shortcuts=True)
        self.assertEqual(
            trainers.resolve_overfit_query_options(shortcuts),
            (True, False),
        )

    def test_overfit_paths_require_both(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = os.path.join(tmpdir, "train")
            os.makedirs(data_dir)
            audio_path, _chart_path = _write_valid_pair(data_dir, "song")
            experiment = _tiny_experiment(
                data_dir, data_dir, os.path.join(tmpdir, "out")
            )
            experiment.dataset.overfit_audio_path = audio_path
            with self.assertRaises(ValueError):
                trainers._resolve_single_song_pair(
                    experiment.dataset,
                    experiment.run,
                )

    def test_overfit_explicit_pair_uses_same_train_and_val(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = os.path.join(tmpdir, "train")
            os.makedirs(data_dir)
            audio_path, chart_path = _write_valid_pair(data_dir, "song")
            model_dir = os.path.join(tmpdir, "models")

            experiment = _tiny_experiment(data_dir, data_dir, model_dir)
            experiment.dataset.overfit_audio_path = audio_path
            experiment.dataset.overfit_chart_path = chart_path

            train_ds, val_ds = trainers._create_datasets(experiment)
            train_batch = next(iter(train_ds.take(1)))
            val_batch = next(iter(val_ds.take(1)))
            np.testing.assert_array_equal(
                train_batch["gt_times"], val_batch["gt_times"]
            )

    def test_overfit_one_song_loss_decreases(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = os.path.join(tmpdir, "train")
            os.makedirs(data_dir)
            _write_valid_pair(data_dir, "song")
            model_dir = os.path.join(tmpdir, "models")

            experiment = _tiny_experiment(data_dir, data_dir, model_dir)
            experiment.run.overfit_one_song = True
            experiment.run.epochs = 6

            _base_model, history = trainers.train_onset_event(
                experiment,
                take_count=-1,
                val_take_count=-1,
            )

            losses = history.history["loss"]
            self.assertEqual(len(losses), 6)
            self.assertLess(min(losses), losses[0])

    @pytest.mark.slow
    def test_overfit_one_song_caps_epochs_at_300(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = os.path.join(tmpdir, "train")
            os.makedirs(data_dir)
            _write_valid_pair(data_dir, "song")
            model_dir = os.path.join(tmpdir, "models")

            experiment = _tiny_experiment(data_dir, data_dir, model_dir)
            experiment.run.overfit_one_song = True
            experiment.run.epochs = 400

            _base_model, history = trainers.train_onset_event(
                experiment,
                take_count=-1,
                val_take_count=-1,
            )

            self.assertEqual(len(history.history["loss"]), 300)

    @pytest.mark.slow
    def test_pipeline_check_shortcuts_reaches_perfect_f1(self):
        step_lines = [f"1000 {t:.6f}\n" for t in np.arange(0.1, 2.5, 0.2)]
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = os.path.join(tmpdir, "train")
            os.makedirs(data_dir)
            _write_valid_pair(
                data_dir,
                "song",
                step_lines=step_lines,
                duration_sec=2.4,
            )
            model_dir = os.path.join(tmpdir, "models")

            experiment = _tiny_experiment(data_dir, data_dir, model_dir)
            experiment.dataset.max_audio_seconds = 2.5
            experiment.dataset.n_max_onsets = 16
            experiment.model.max_audio_seconds = 2.5
            experiment.model.num_queries = 12
            experiment.model.embed_dim = 32
            experiment.model.decoder_layers = 2
            experiment.model.base_filters = 32
            experiment.model.encoder = config.OnsetEventEncoderConfig(
                initial_filters=16,
                depth=1,
            )
            experiment.run.pipeline_check_shortcuts = True
            experiment.run.epochs = 100
            experiment.run.lambda_cls = 5.0
            experiment.run.lambda_time = 20.0

            _base_model, history = trainers.train_onset_event(
                experiment,
                take_count=-1,
                val_take_count=-1,
            )

            val_f1 = history.history["val_event_onset_f1"]
            self.assertGreaterEqual(max(val_f1), 0.99)

    @pytest.mark.slow
    def test_overfit_one_song_event_f1_reaches_one(self):
        # 12 onsets every 0.2 s from 0.1–2.3 s with num_queries=12 (1:1). Audio
        # duration 2.4 s matches the chart extent so query-grid times align with
        # GT; the model must still learn per-slot confidence above threshold.
        step_lines = [f"1000 {t:.6f}\n" for t in np.arange(0.1, 2.5, 0.2)]
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = os.path.join(tmpdir, "train")
            os.makedirs(data_dir)
            _write_valid_pair(
                data_dir,
                "song",
                step_lines=step_lines,
                duration_sec=2.4,
            )
            model_dir = os.path.join(tmpdir, "models")

            experiment = _tiny_experiment(data_dir, data_dir, model_dir)
            experiment.dataset.max_audio_seconds = 2.5
            experiment.dataset.n_max_onsets = 16
            experiment.model.max_audio_seconds = 2.5
            experiment.model.num_queries = 12
            experiment.model.embed_dim = 32
            experiment.model.decoder_layers = 2
            experiment.model.base_filters = 32
            experiment.model.encoder = config.OnsetEventEncoderConfig(
                initial_filters=16,
                depth=1,
            )
            experiment.run.overfit_one_song = True
            experiment.run.epochs = 100
            experiment.run.seed = 42
            experiment.run.lambda_cls = 5.0
            experiment.run.lambda_time = 20.0

            _base_model, history = trainers.train_onset_event(
                experiment,
                take_count=-1,
                val_take_count=-1,
            )

            val_f1 = history.history["val_event_onset_f1"]
            self.assertGreaterEqual(max(val_f1), 0.99)

    def test_build_experiment_callbacks_saves_config(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            callback_root = os.path.join(tmpdir, "callbacks")
            experiment = _tiny_experiment("train", "val", "out")
            experiment.run.callback_root_dir = callback_root
            callbacks = trainers._build_experiment_callbacks(
                run_config=experiment.run,
                experiment_name="test-run",
                monitor_metric="val_event_onset_f1",
                monitor_mode="max",
                experiment_config=experiment,
            )
            self.assertEqual(len(callbacks), 2)
            logs_dir = os.path.join(callback_root, "logs")
            run_dirs = os.listdir(logs_dir)
            self.assertEqual(len(run_dirs), 1)
            config_path = os.path.join(logs_dir, run_dirs[0], "config.json")
            self.assertTrue(os.path.isfile(config_path))
