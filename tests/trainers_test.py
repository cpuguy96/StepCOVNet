import contextlib
import json
import pathlib
import tempfile
import typing
import unittest
from types import SimpleNamespace
from unittest import mock

import keras
import numpy as np
import tensorflow as tf

from stepcovnet import config, datasets, dense_overfit_eval, losses, models, trainers
from stepcovnet.dataset_prep import config as prep_config
from stepcovnet.dataset_prep import pipeline, training_index

TEST_DATA_DIR = pathlib.Path(__file__).resolve().parent / "testdata"


def _keras_model_stub(*, predict_return_value=None):
    model = mock.create_autospec(keras.Model, instance=True)
    if predict_return_value is not None:
        model.predict.return_value = predict_return_value
    model.fit.return_value = SimpleNamespace(
        history={"val_loss": [1.0], "loss": [1.0]},
    )
    return model


@contextlib.contextmanager
def _temp_model_and_callback_dirs(with_callbacks: bool = False):
    """Yield (model_output_dir, callback_root_dir) inside a temporary directory."""
    with tempfile.TemporaryDirectory() as temp_dir:
        model_output_dir = pathlib.Path(temp_dir) / "models"
        callback_root_dir = (
            pathlib.Path(temp_dir) / "callbacks" if with_callbacks else ""
        )
        yield model_output_dir, callback_root_dir


def _make_onset_configs(
    model_output_dir: str,
    *,
    dataset_kwargs: dict | None = None,
    model_kwargs: dict | None = None,
    run_kwargs: dict | None = None,
) -> tuple[config.OnsetDatasetConfig, config.OnsetModelConfig, config.RunConfig]:
    """Factory for a minimal onset (dataset, model, run) config triple."""
    dataset_kwargs = dict(dataset_kwargs or {})
    model_kwargs = dict(model_kwargs or {})
    run_kwargs = dict(run_kwargs or {})

    dataset_config = config.OnsetDatasetConfig(
        data_dir=dataset_kwargs.pop("data_dir", TEST_DATA_DIR),
        val_data_dir=dataset_kwargs.pop("val_data_dir", TEST_DATA_DIR),
        **dataset_kwargs,
    )
    model_config = config.OnsetModelConfig(**model_kwargs)

    base_run_kwargs: dict = {
        "epoch": 1,
        "take_count": 1,
        "model_output_dir": model_output_dir,
    }
    base_run_kwargs.update(run_kwargs)
    run_config = config.RunConfig(**base_run_kwargs)
    return dataset_config, model_config, run_config


def _make_arrow_configs(
    model_output_dir: str,
    *,
    dataset_kwargs: dict | None = None,
    model_kwargs: dict | None = None,
    run_kwargs: dict | None = None,
) -> tuple[config.ArrowDatasetConfig, config.ArrowModelConfig, config.ArrowRunConfig]:
    """Factory for a minimal arrow (dataset, model, run) config triple."""
    dataset_kwargs = dict(dataset_kwargs or {})
    model_kwargs = dict(model_kwargs or {})
    run_kwargs = dict(run_kwargs or {})

    dataset_config = config.ArrowDatasetConfig(
        data_dir=dataset_kwargs.pop("data_dir", TEST_DATA_DIR),
        val_data_dir=dataset_kwargs.pop("val_data_dir", TEST_DATA_DIR),
        **dataset_kwargs,
    )
    model_config = config.ArrowModelConfig.from_dict(model_kwargs or {})

    base_run_kwargs: dict = {
        "epoch": 1,
        "take_count": 1,
        "model_output_dir": model_output_dir,
    }
    base_run_kwargs.update(run_kwargs)
    run_config = config.ArrowRunConfig(**base_run_kwargs)
    return dataset_config, model_config, run_config


def _make_arrow_experiment_config(
    model_output_dir: str,
    *,
    dataset_kwargs: dict | None = None,
    model_kwargs: dict | None = None,
    run_kwargs: dict | None = None,
) -> config.ArrowExperimentConfig:
    """Build an ArrowExperimentConfig for tests that call run_arrow_train_from_config."""
    dataset_config, model_config, run_config = _make_arrow_configs(
        model_output_dir,
        dataset_kwargs=dataset_kwargs,
        model_kwargs=model_kwargs,
        run_kwargs=run_kwargs,
    )
    return config.ArrowExperimentConfig(
        dataset=dataset_config, model=model_config, run=run_config
    )


class TrainersTest(unittest.TestCase):
    def test_run_train(self):
        with _temp_model_and_callback_dirs(with_callbacks=True) as (
            model_output_dir,
            callback_root_dir,
        ):
            model, history = trainers.run_train(
                data_dir=TEST_DATA_DIR,
                val_data_dir=TEST_DATA_DIR,
                batch_size=1,
                apply_temporal_augment=False,
                should_apply_spec_augment=False,
                use_gaussian_target=False,
                gaussian_sigma=0.0,
                model_params={
                    "initial_filters": 8,
                    "depth": 1,
                    "dilation_rates": [1, 2],
                    "dropout_rate": 0.0,
                },
                take_count=1,
                epoch=1,
                callback_root_dir=callback_root_dir,
                model_output_dir=model_output_dir,
            )
        self.assertIsNotNone(model)
        self.assertIsNotNone(history)

    def test_run_arrow_train(self):
        with _temp_model_and_callback_dirs(with_callbacks=True) as (
            model_output_dir,
            callback_root_dir,
        ):
            model, history = trainers.run_arrow_train(
                data_dir=TEST_DATA_DIR,
                val_data_dir=TEST_DATA_DIR,
                batch_size=1,
                model_params={},
                take_count=1,
                epoch=1,
                callback_root_dir=callback_root_dir,
                model_output_dir=model_output_dir,
            )

        self.assertIsNotNone(model)
        self.assertIsNotNone(history)

    def test_run_train_from_config(self):
        with _temp_model_and_callback_dirs(with_callbacks=True) as (
            model_output_dir,
            callback_root_dir,
        ):
            dataset_config, model_config, run_config = _make_onset_configs(
                model_output_dir,
                dataset_kwargs={
                    "batch_size": 1,
                    "apply_temporal_augment": False,
                    "should_apply_spec_augment": False,
                    "use_gaussian_target": False,
                    "gaussian_sigma": 0.0,
                },
                model_kwargs={
                    "initial_filters": 8,
                    "depth": 1,
                    "dilation_rates": [1, 2],
                    "dropout_rate": 0.0,
                },
                run_kwargs={
                    "epoch": 1,
                    "take_count": 1,
                    "callback_root_dir": callback_root_dir,
                },
            )
            model, history = trainers.run_train_from_config(
                dataset_config, model_config, run_config
            )
        self.assertIsNotNone(model)
        self.assertIsNotNone(history)

    def test_run_train_from_config_with_training_index_path(self):
        fixtures_root = (
            pathlib.Path(__file__).resolve().parent / "fixtures" / "dataset_prep"
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = pathlib.Path(tmpdir) / "out"
            pipeline.run_preprocess(
                prep_config.PrepConfig(
                    input_dir=str(fixtures_root / "vocaloid_multi_sm"),
                    output_dir=str(out_dir),
                    overwrite=True,
                )
            )
            index_path = training_index.save_training_index(
                training_index.build_training_index(
                    out_dir,
                    val_fraction=0.0,
                    seed=1,
                )
            )
            stub_ds = datasets.create_dataset(TEST_DATA_DIR, batch_size=1)

            def _fake_create_dataset(data_dir, **kwargs):
                return stub_ds

            with _temp_model_and_callback_dirs() as (
                model_output_dir,
                callback_root_dir,
            ):
                dataset_config, model_config, run_config = _make_onset_configs(
                    model_output_dir,
                    dataset_kwargs={
                        "data_dir": "",
                        "val_data_dir": "",
                        "training_index_path": str(index_path),
                        "batch_size": 1,
                    },
                    model_kwargs={
                        "initial_filters": 8,
                        "depth": 1,
                        "dilation_rates": [1, 2],
                        "dropout_rate": 0.0,
                    },
                    run_kwargs={
                        "epoch": 1,
                        "take_count": 1,
                        "val_take_count": 1,
                        "callback_root_dir": callback_root_dir,
                    },
                )
                with mock.patch.object(
                    datasets,
                    "create_dataset",
                    side_effect=_fake_create_dataset,
                ) as mock_create:
                    model, history = trainers.run_train_from_config(
                        dataset_config,
                        model_config,
                        run_config,
                    )
        self.assertEqual(mock_create.call_count, 2)
        train_call = mock_create.call_args_list[0]
        val_call = mock_create.call_args_list[1]
        self.assertEqual(train_call.kwargs["split"], training_index.SPLIT_TRAIN)
        self.assertEqual(val_call.kwargs["split"], training_index.SPLIT_VAL)
        self.assertEqual(train_call.kwargs["data_dir"], str(index_path))
        self.assertEqual(val_call.kwargs["data_dir"], str(index_path))
        self.assertEqual(
            pathlib.Path(train_call.kwargs["data_root"]),
            out_dir.resolve(),
        )
        self.assertIsNotNone(model)
        self.assertIsNotNone(history)

    def test_run_arrow_train_from_config(self):
        with _temp_model_and_callback_dirs(with_callbacks=True) as (
            model_output_dir,
            callback_root_dir,
        ):
            exp = _make_arrow_experiment_config(
                model_output_dir,
                run_kwargs={
                    "epoch": 1,
                    "take_count": 1,
                    "callback_root_dir": callback_root_dir,
                },
            )
            model, history = trainers.run_arrow_train_from_config(exp)
        self.assertIsNotNone(model)
        self.assertIsNotNone(history)

    def test_latest_monitored_checkpoint_picks_highest_onset_f1_score(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = pathlib.Path(tmp) / "models" / "run1"
            pathlib.Path(run_dir).mkdir(parents=True, exist_ok=True)
            low_path = pathlib.Path(run_dir) / "VAL_ONSET_F1_SCORE-0.10000.keras"
            high_path = pathlib.Path(run_dir) / "VAL_ONSET_F1_SCORE-0.99000.keras"
            for path in (low_path, high_path):
                with pathlib.Path(path).open("wb") as checkpoint_file:
                    checkpoint_file.write(b"")
            selected = trainers._latest_monitored_checkpoint(
                tmp, trainers.ONSET_CHECKPOINT_MONITOR
            )
            self.assertEqual(selected, str(high_path))

    def test_list_monitored_checkpoints_returns_all_sorted(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = pathlib.Path(tmp) / "models" / "run1"
            pathlib.Path(run_dir).mkdir(parents=True, exist_ok=True)
            paths = [
                pathlib.Path(run_dir) / "VAL_ONSET_F1_SCORE-0.20000.keras",
                pathlib.Path(run_dir) / "VAL_ONSET_F1_SCORE-0.10000.keras",
                pathlib.Path(run_dir) / "VAL_ONSET_F1_SCORE-notanumber.keras",
                pathlib.Path(run_dir) / "not_a_checkpoint.keras",
            ]
            for path in paths:
                with pathlib.Path(path).open("wb") as checkpoint_file:
                    checkpoint_file.write(b"")
            listed = trainers._list_monitored_checkpoints(
                tmp, trainers.ONSET_CHECKPOINT_MONITOR
            )
            self.assertEqual(
                listed,
                [
                    str(pathlib.Path(run_dir) / "VAL_ONSET_F1_SCORE-0.10000.keras"),
                    str(pathlib.Path(run_dir) / "VAL_ONSET_F1_SCORE-0.20000.keras"),
                ],
            )

    def test_list_monitored_checkpoints_empty_when_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(
                trainers._list_monitored_checkpoints(
                    tmp, trainers.ONSET_CHECKPOINT_MONITOR
                ),
                [],
            )

    def test_select_best_event_f1_checkpoint_uses_event_f1_not_frame_f1(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = pathlib.Path(tmp) / "callbacks" / "models" / "run1"
            pathlib.Path(run_dir).mkdir(parents=True, exist_ok=True)
            low_frame = pathlib.Path(run_dir) / "VAL_ONSET_F1_SCORE-0.10000.keras"
            high_frame = pathlib.Path(run_dir) / "VAL_ONSET_F1_SCORE-0.20000.keras"
            for path in (low_frame, high_frame):
                with pathlib.Path(path).open("wb") as checkpoint_file:
                    checkpoint_file.write(b"")
            dataset_config, model_config, run_config = _make_onset_configs(
                pathlib.Path(tmp) / "models",
                run_kwargs={
                    "callback_root_dir": pathlib.Path(tmp) / "callbacks",
                    "post_hoc_event_f1_thresholds": [0.2, 0.35],
                },
            )

            # Sorted checkpoint order: [0.10000, 0.20000]; give the lower-frame
            # checkpoint the higher event F1 to prove event F1 drives selection.
            sweeps = [
                {"best_threshold": 0.35, "best_micro_event_f1": 0.8},
                {"best_threshold": 0.2, "best_micro_event_f1": 0.5},
            ]
            with (
                mock.patch.object(
                    trainers.keras.models,
                    "load_model",
                    return_value=_keras_model_stub(),
                    autospec=True,
                ),
                mock.patch.object(
                    trainers.dense_overfit_eval,
                    "sweep_thresholds_dense_val_event_f1",
                    side_effect=sweeps,
                    autospec=True,
                ),
            ):
                report = trainers._select_best_event_f1_checkpoint(
                    dataset_config, model_config, run_config
                )
            self.assertEqual(report["best_checkpoint"], str(low_frame))
            self.assertEqual(report["best_threshold"], 0.35)
            self.assertEqual(report["best_micro_event_f1"], 0.8)
            self.assertEqual(len(report["per_checkpoint"]), 2)

    def test_select_best_event_f1_checkpoint_none_without_checkpoints(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset_config, model_config, run_config = _make_onset_configs(
                pathlib.Path(tmp) / "models",
                run_kwargs={"callback_root_dir": pathlib.Path(tmp) / "callbacks"},
            )
            self.assertIsNone(
                trainers._select_best_event_f1_checkpoint(
                    dataset_config, model_config, run_config
                )
            )

    def test_export_best_event_f1_checkpoint_writes_model_and_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = pathlib.Path(tmp) / "callbacks" / "models" / "run1"
            pathlib.Path(run_dir).mkdir(parents=True, exist_ok=True)
            best_ckpt = pathlib.Path(run_dir) / "VAL_ONSET_F1_SCORE-0.15000.keras"
            with pathlib.Path(best_ckpt).open("wb") as checkpoint_file:
                checkpoint_file.write(b"")
            model_output_dir = pathlib.Path(tmp) / "models"
            dataset_config, model_config, run_config = _make_onset_configs(
                model_output_dir,
                run_kwargs={
                    "callback_root_dir": pathlib.Path(tmp) / "callbacks",
                    "post_hoc_event_f1_thresholds": [0.2],
                },
            )
            stub_model = _keras_model_stub()
            stub_model.name = "dense_model"
            best_model = _keras_model_stub()

            def _save(filepath):
                with pathlib.Path(filepath).open("wb") as out_file:
                    out_file.write(b"saved")

            best_model.save.side_effect = _save
            with (
                mock.patch.object(
                    trainers.keras.models,
                    "load_model",
                    return_value=best_model,
                    autospec=True,
                ),
                mock.patch.object(
                    trainers.dense_overfit_eval,
                    "sweep_thresholds_dense_val_event_f1",
                    return_value={
                        "best_threshold": 0.2,
                        "best_micro_event_f1": 0.7,
                    },
                    autospec=True,
                ),
            ):
                report = trainers._export_best_event_f1_checkpoint(
                    stub_model, dataset_config, model_config, run_config
                )
            self.assertIsNotNone(report)
            exported = pathlib.Path(model_output_dir) / "dense_model.keras"
            self.assertTrue(pathlib.Path(exported).is_file())
            report_path = (
                pathlib.Path(model_output_dir) / trainers.POST_HOC_EVENT_F1_REPORT_NAME
            )
            self.assertTrue(report_path.is_file())
            with report_path.open(encoding="utf-8") as report_file:
                written = json.load(report_file)
            self.assertEqual(written["best_checkpoint"], str(best_ckpt))
            self.assertEqual(written["best_threshold"], 0.2)

    def test_export_best_event_f1_checkpoint_none_without_checkpoints(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset_config, model_config, run_config = _make_onset_configs(
                pathlib.Path(tmp) / "models",
                run_kwargs={"callback_root_dir": pathlib.Path(tmp) / "callbacks"},
            )
            stub_model = _keras_model_stub()
            stub_model.name = "dense_model"
            self.assertIsNone(
                trainers._export_best_event_f1_checkpoint(
                    stub_model, dataset_config, model_config, run_config
                )
            )

    def test_run_train_from_config_invokes_post_hoc_export_when_enabled(self):
        with _temp_model_and_callback_dirs(with_callbacks=True) as (
            model_output_dir,
            callback_root_dir,
        ):
            dataset_config, model_config, run_config = _make_onset_configs(
                model_output_dir,
                dataset_kwargs={
                    "batch_size": 1,
                    "use_gaussian_target": True,
                    "gaussian_sigma": 1.0,
                },
                model_kwargs={
                    "initial_filters": 8,
                    "depth": 1,
                    "dilation_rates": [1, 2],
                },
                run_kwargs={
                    "epoch": 1,
                    "take_count": 1,
                    "callback_root_dir": callback_root_dir,
                    "post_hoc_event_f1_export": True,
                },
            )
            with mock.patch.object(
                trainers,
                "_export_best_event_f1_checkpoint",
                autospec=True,
            ) as export_mock:
                trainers.run_train_from_config(dataset_config, model_config, run_config)
            export_mock.assert_called_once()

    def test_run_train_from_config_skips_post_hoc_export_when_disabled(self):
        with _temp_model_and_callback_dirs(with_callbacks=True) as (
            model_output_dir,
            callback_root_dir,
        ):
            dataset_config, model_config, run_config = _make_onset_configs(
                model_output_dir,
                dataset_kwargs={"batch_size": 1},
                model_kwargs={
                    "initial_filters": 8,
                    "depth": 1,
                    "dilation_rates": [1, 2],
                },
                run_kwargs={
                    "epoch": 1,
                    "take_count": 1,
                    "callback_root_dir": callback_root_dir,
                },
            )
            with mock.patch.object(
                trainers,
                "_export_best_event_f1_checkpoint",
                autospec=True,
            ) as export_mock:
                trainers.run_train_from_config(dataset_config, model_config, run_config)
            export_mock.assert_not_called()

    def test_onset_checkpoint_callback_monitors_frame_f1(self):
        callback = trainers._get_ckpt_callback(
            "/tmp/cb",
            "run1",
            trainers.ONSET_CHECKPOINT_MONITOR,
            "max",
        )
        self.assertEqual(callback.monitor, trainers.ONSET_CHECKPOINT_MONITOR)
        self.assertIn("VAL_ONSET_F1_SCORE", callback.filepath)

    def test_build_onset_dense_compile_metrics_keeps_frame_f1(self):
        run_config = config.RunConfig(
            epoch=1,
            take_count=1,
            model_output_dir="out",
            confidence_threshold=0.05,
        )
        metric_names = [
            metric.name
            for metric in trainers.build_onset_dense_compile_metrics(run_config)
        ]
        self.assertIn("onset_f1_score", metric_names)
        self.assertNotIn("dense_event_onset_f1", metric_names)

    def test_dense_val_event_f1_callback_logs_monitor_metric(self):
        features = np.zeros((1, 50, 128), dtype=np.float32)
        y_true = np.zeros((1, 50, 1), dtype=np.float32)
        y_true[0, 10, 0] = 1.0
        y_pred = np.zeros((1, 50, 1), dtype=np.float32)
        y_pred[0, 10, 0] = 0.9
        stub_model = _keras_model_stub(predict_return_value=y_pred)
        val_ds = tf.data.Dataset.from_tensor_slices((features, y_true)).batch(1)
        callback = dense_overfit_eval.DenseValEventF1Callback(
            val_ds,
            confidence_threshold=0.5,
        )
        callback.set_model(stub_model)
        logs: dict[str, float] = {}
        callback.on_epoch_end(0, logs)
        self.assertIn("val_dense_event_onset_f1", logs)
        self.assertEqual(logs["val_dense_event_onset_f1"], 1.0)

    def test_dense_val_event_f1_callback_sweeps_past_flooded_low_threshold(self):
        """A low threshold floods peaks; the sweep must find the clean operating point.

        Weak spurious peaks that clear a 0.05 threshold become false positives
        and sink event F1 even when every real onset is predicted exactly.
        """
        n_frames = 1200
        features = np.zeros((1, n_frames, 128), dtype=np.float32)
        y_true = np.zeros((1, n_frames, 1), dtype=np.float32)
        y_pred = np.zeros((1, n_frames, 1), dtype=np.float32)
        # Irregular gaps: evenly spaced onsets are reproduced exactly by the
        # audio-blind interval-shuffling null, which saturates the floor at 1.0.
        rng = np.random.default_rng(0)
        gaps = rng.integers(14, 30, size=80)
        onset_frames = 20 + np.cumsum(gaps)
        frames = onset_frames[onset_frames < n_frames - 20]
        for frame in frames:
            y_true[0, frame, 0] = 1.0
            y_pred[0, frame, 0] = 0.9
        # 10 ms hop, 50 ms min gap: spurious peaks must stay 5+ frames clear.
        for start, end in zip(frames[:-1], frames[1:], strict=True):
            mid = (start + end) // 2
            if mid - start >= 6 and end - mid >= 6:
                y_pred[0, mid, 0] = 0.06

        stub_model = _keras_model_stub(predict_return_value=y_pred)
        val_ds = tf.data.Dataset.from_tensor_slices((features, y_true)).batch(1)
        callback = dense_overfit_eval.DenseValEventF1Callback(
            val_ds,
            confidence_threshold=0.05,
        )
        callback.set_model(stub_model)
        logs: dict[str, float] = {}
        callback.on_epoch_end(0, logs)

        flooded = callback._summary_at([(y_true, y_pred)], 0.05)
        clean = callback._summary_at([(y_true, y_pred)], 0.3)
        self.assertLess(flooded["event_f1"], clean["event_f1"])
        self.assertLess(flooded["skill_event_f1"], clean["skill_event_f1"])

        self.assertEqual(logs["val_dense_event_onset_f1"], 1.0)
        self.assertGreater(logs["val_timing_match_threshold"], 0.05)
        self.assertGreater(logs["val_skill_event_f1"], 0.0)

    def test_dense_val_event_f1_callback_selects_on_skill_not_raw_f1(self):
        """Selection must discount the audio-blind floor, not just maximize F1."""
        n_frames = 400
        features = np.zeros((1, n_frames, 128), dtype=np.float32)
        y_true = np.zeros((1, n_frames, 1), dtype=np.float32)
        y_pred = np.zeros((1, n_frames, 1), dtype=np.float32)
        for frame in range(20, n_frames - 20, 8):
            y_true[0, frame, 0] = 1.0
            y_pred[0, frame, 0] = 0.9

        stub_model = _keras_model_stub(predict_return_value=y_pred)
        val_ds = tf.data.Dataset.from_tensor_slices((features, y_true)).batch(1)
        callback = dense_overfit_eval.DenseValEventF1Callback(
            val_ds,
            confidence_threshold=0.05,
        )
        callback.set_model(stub_model)
        logs: dict[str, float] = {}
        callback.on_epoch_end(0, logs)

        self.assertIn("val_skill_event_f1", logs)
        self.assertIn("val_skill_event_f1_floor", logs)
        self.assertLessEqual(logs["val_skill_event_f1"], 1.0)
        self.assertLessEqual(
            logs["val_skill_event_f1"],
            logs["val_dense_event_onset_f1"],
        )

    def test_get_callbacks_adds_early_stopping_when_patience_set(self):
        callbacks, _ = trainers._get_callbacks(
            "/tmp/cb",
            trainers.ONSET_CHECKPOINT_MONITOR,
            "max",
            early_stopping_patience=10,
        )
        self.assertEqual(len(callbacks), 3)
        self.assertIsInstance(callbacks[2], keras.callbacks.EarlyStopping)
        self.assertEqual(callbacks[2].patience, 10)

    def test_run_train_from_config_saves_model_with_explicit_model_name(self):
        """Saved model file and model.name use run_config.model_name when set."""
        with _temp_model_and_callback_dirs() as (model_output_dir, _):
            dataset_config, model_config, run_config = _make_onset_configs(
                model_output_dir,
                model_kwargs={"initial_filters": 8, "depth": 1},
                run_kwargs={"model_name": "my_onset_model"},
            )
            model, _ = trainers.run_train_from_config(
                dataset_config, model_config, run_config
            )
            expected_name = "stepcovnet_ONSET-my_onset_model"
            self.assertEqual(model.name, expected_name)
            saved_path = pathlib.Path(model_output_dir) / (expected_name + ".keras")
            self.assertTrue(
                saved_path.is_file(),
                f"Expected saved model at {saved_path}",
            )

    def test_run_train_from_config_saves_model_with_derived_name_when_model_name_empty(
        self,
    ):
        """When model_name is empty, model name uses experiment-derived name."""
        with _temp_model_and_callback_dirs() as (model_output_dir, _):
            dataset_config, model_config, run_config = _make_onset_configs(
                model_output_dir,
                model_kwargs={"initial_filters": 8, "depth": 1},
                run_kwargs={"model_name": ""},  # empty: use experiment name
            )
            model, _ = trainers.run_train_from_config(
                dataset_config, model_config, run_config
            )
            self.assertTrue(
                model.name.startswith("stepcovnet_ONSET-"),
                f"Expected model.name to start with 'stepcovnet_ONSET-', got {model.name!r}",
            )
            saved_path = pathlib.Path(model_output_dir) / (model.name + ".keras")
            self.assertTrue(
                saved_path.is_file(),
                f"Expected saved model at {saved_path}",
            )

    def test_run_arrow_train_from_config_saves_model_with_explicit_model_name(self):
        """Arrow saved model file and model.name use run_config.model_name when set."""
        with _temp_model_and_callback_dirs() as (model_output_dir, _):
            exp = _make_arrow_experiment_config(
                model_output_dir,
                run_kwargs={"model_name": "my_arrow_model"},
            )
            model, _ = trainers.run_arrow_train_from_config(exp)
            expected_name = "stepcovnet_ARROW-my_arrow_model"
            self.assertEqual(model.name, expected_name)
            saved_path = pathlib.Path(model_output_dir) / (expected_name + ".keras")
            self.assertTrue(
                saved_path.is_file(),
                f"Expected saved model at {saved_path}",
            )

    def test_run_arrow_train_from_config_saves_model_with_derived_name_when_model_name_empty(
        self,
    ):
        """When model_name is empty, arrow model name uses experiment-derived name."""
        with _temp_model_and_callback_dirs() as (model_output_dir, _):
            exp = _make_arrow_experiment_config(
                model_output_dir,
                run_kwargs={"model_name": ""},
            )
            model, _ = trainers.run_arrow_train_from_config(exp)
            self.assertTrue(
                model.name.startswith("stepcovnet_ARROW-"),
                f"Expected model.name to start with 'stepcovnet_ARROW-', got {model.name!r}",
            )
            saved_path = pathlib.Path(model_output_dir) / (model.name + ".keras")
            self.assertTrue(
                saved_path.is_file(),
                f"Expected saved model at {saved_path}",
            )

    def test_run_arrow_train_from_config_with_verbosity_quiet(self):
        """Training with show_model_summary=False and fit_verbose=0 completes."""
        with _temp_model_and_callback_dirs(with_callbacks=True) as (
            model_output_dir,
            callback_root_dir,
        ):
            exp = _make_arrow_experiment_config(
                model_output_dir,
                run_kwargs={
                    "callback_root_dir": callback_root_dir,
                    "show_model_summary": False,
                    "fit_verbose": 0,
                },
            )
            model, history = trainers.run_arrow_train_from_config(exp)
        self.assertIsNotNone(model)
        self.assertIsNotNone(history)

    def test_run_arrow_train_from_config_fit_receives_verbose(self):
        """model.fit is called with verbose from run_config.fit_verbose."""
        mock_model = _keras_model_stub()

        with _temp_model_and_callback_dirs() as (model_output_dir, _):
            exp = _make_arrow_experiment_config(
                model_output_dir,
                run_kwargs={"fit_verbose": 2},
            )
            with (
                mock.patch.object(
                    models,
                    "build_arrow_model_from_config",
                    return_value=mock_model,
                    autospec=True,
                ),
                mock.patch.object(trainers, "_write_model", autospec=True),
            ):
                trainers.run_arrow_train_from_config(exp)
        mock_model.fit.assert_called_once()
        self.assertEqual(mock_model.fit.call_args.kwargs["verbose"], 2)

    def test_run_arrow_train_from_config_show_model_summary_false_skips_summary(
        self,
    ):
        """When show_model_summary is False, model.summary() is not called."""
        mock_model = _keras_model_stub()

        with _temp_model_and_callback_dirs() as (model_output_dir, _):
            exp = _make_arrow_experiment_config(
                model_output_dir,
                run_kwargs={"show_model_summary": False},
            )
            with (
                mock.patch.object(
                    models,
                    "build_arrow_model_from_config",
                    return_value=mock_model,
                    autospec=True,
                ),
                mock.patch.object(trainers, "_write_model", autospec=True),
            ):
                trainers.run_arrow_train_from_config(exp)
        mock_model.summary.assert_not_called()

    def test_run_arrow_train_from_config_show_model_summary_true_calls_summary(
        self,
    ):
        """When show_model_summary is True (default), model.summary() is called."""
        mock_model = _keras_model_stub()

        with _temp_model_and_callback_dirs() as (model_output_dir, _):
            exp = _make_arrow_experiment_config(
                model_output_dir,
                run_kwargs={"show_model_summary": True},
            )
            with (
                mock.patch.object(
                    models,
                    "build_arrow_model_from_config",
                    return_value=mock_model,
                    autospec=True,
                ),
                mock.patch.object(trainers, "_write_model", autospec=True),
            ):
                trainers.run_arrow_train_from_config(exp)
        mock_model.summary.assert_called_once()

    def test_config_serialization(self):
        """Test that configs can be serialized to/from JSON."""
        dataset_config = config.OnsetDatasetConfig(
            data_dir="data/train",
            val_data_dir="data/val",
            batch_size=4,
            apply_temporal_augment=True,
            should_apply_spec_augment=True,
            use_gaussian_target=True,
            gaussian_sigma=1.5,
        )
        model_config = config.OnsetModelConfig(
            initial_filters=16,
            depth=2,
            dilation_rates=[1, 2, 4, 8],
            kernel_size=3,
            dropout_rate=0.1,
        )
        run_config = config.RunConfig(
            epoch=20,
            take_count=-1,
            model_output_dir="out",
            callback_root_dir="callbacks",
            model_name="test_model",
            seed=42,
        )
        experiment_config = config.OnsetExperimentConfig(
            dataset=dataset_config, model=model_config, run=run_config
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = pathlib.Path(temp_dir) / "config.json"
            experiment_config.to_json(config_path)

            # Load it back
            loaded_config = config.OnsetExperimentConfig.from_json(config_path)

            self.assertEqual(loaded_config.dataset.data_dir, dataset_config.data_dir)
            self.assertEqual(
                loaded_config.dataset.batch_size, dataset_config.batch_size
            )
            self.assertEqual(
                loaded_config.model.initial_filters, model_config.initial_filters
            )
            self.assertEqual(
                loaded_config.model.dilation_rates, model_config.dilation_rates
            )
            self.assertEqual(loaded_config.run.epoch, run_config.epoch)
            self.assertEqual(loaded_config.run.seed, run_config.seed)

    def test_run_train_from_config_saves_config(self):
        """Test that config is saved when callback_root_dir is set."""
        with tempfile.TemporaryDirectory() as temp_dir:
            callback_root_dir = pathlib.Path(temp_dir) / "callbacks"
            model_output_dir = pathlib.Path(temp_dir) / "models"
            dataset_config = config.OnsetDatasetConfig(
                data_dir=TEST_DATA_DIR,
                val_data_dir=TEST_DATA_DIR,
                batch_size=1,
            )
            model_config = config.OnsetModelConfig(initial_filters=8, depth=1)
            run_config = config.RunConfig(
                epoch=1,
                take_count=1,
                model_output_dir=model_output_dir,
                callback_root_dir=callback_root_dir,
            )
            model, history = trainers.run_train_from_config(
                dataset_config, model_config, run_config
            )

            # Check that config file was created
            log_dirs = [
                d
                for d in (pathlib.Path(callback_root_dir) / "logs").iterdir()
                if d.is_dir()
            ]
            self.assertGreater(len(log_dirs), 0)
            config_path = (
                pathlib.Path(callback_root_dir)
                / "logs"
                / log_dirs[0].name
                / "config.json"
            )
            self.assertTrue(config_path.exists())

            # Verify config can be loaded
            loaded_config = config.OnsetExperimentConfig.from_json(config_path)
            self.assertEqual(loaded_config.dataset.batch_size, 1)
            self.assertEqual(loaded_config.model.initial_filters, 8)

    def test_run_train_from_config_no_callbacks(self):
        """Test that training works without callback_root_dir."""
        with tempfile.TemporaryDirectory() as temp_dir:
            model_output_dir = pathlib.Path(temp_dir) / "models"
            dataset_config = config.OnsetDatasetConfig(
                data_dir=TEST_DATA_DIR,
                val_data_dir=TEST_DATA_DIR,
                batch_size=1,
            )
            model_config = config.OnsetModelConfig(initial_filters=8, depth=1)
            run_config = config.RunConfig(
                epoch=1,
                take_count=1,
                model_output_dir=model_output_dir,
                callback_root_dir="",  # No callbacks
            )
            model, history = trainers.run_train_from_config(
                dataset_config, model_config, run_config
            )
            self.assertIsNotNone(model)
            self.assertIsNotNone(history)

    def test_run_train_with_gaussian_targets(self):
        """Test training with Gaussian targets."""
        with tempfile.TemporaryDirectory() as temp_dir:
            model_output_dir = pathlib.Path(temp_dir) / "models"
            dataset_config = config.OnsetDatasetConfig(
                data_dir=TEST_DATA_DIR,
                val_data_dir=TEST_DATA_DIR,
                batch_size=1,
                use_gaussian_target=True,
                gaussian_sigma=1.5,
            )
            model_config = config.OnsetModelConfig(initial_filters=8, depth=1)
            run_config = config.RunConfig(
                epoch=1,
                take_count=1,
                model_output_dir=model_output_dir,
            )
            model, history = trainers.run_train_from_config(
                dataset_config, model_config, run_config
            )
            self.assertIsNotNone(model)
            self.assertIsNotNone(history)

    def test_run_train_with_augmentations(self):
        """Test training with temporal and spectrogram augmentations."""
        with tempfile.TemporaryDirectory() as temp_dir:
            model_output_dir = pathlib.Path(temp_dir) / "models"
            dataset_config = config.OnsetDatasetConfig(
                data_dir=TEST_DATA_DIR,
                val_data_dir=TEST_DATA_DIR,
                batch_size=1,
                apply_temporal_augment=True,
                should_apply_spec_augment=True,
            )
            model_config = config.OnsetModelConfig(initial_filters=8, depth=1)
            run_config = config.RunConfig(
                epoch=1,
                take_count=1,
                model_output_dir=model_output_dir,
            )
            model, history = trainers.run_train_from_config(
                dataset_config, model_config, run_config
            )
            self.assertIsNotNone(model)
            self.assertIsNotNone(history)

    def test_run_train_with_seed(self):
        """Test that seed is set when provided in config."""
        with tempfile.TemporaryDirectory() as temp_dir:
            model_output_dir = pathlib.Path(temp_dir) / "models"
            dataset_config = config.OnsetDatasetConfig(
                data_dir=TEST_DATA_DIR,
                val_data_dir=TEST_DATA_DIR,
                batch_size=1,
            )
            model_config = config.OnsetModelConfig(initial_filters=8, depth=1)
            run_config = config.RunConfig(
                epoch=1,
                take_count=1,
                model_output_dir=model_output_dir,
                seed=42,
            )
            model, history = trainers.run_train_from_config(
                dataset_config, model_config, run_config
            )
            self.assertIsNotNone(model)
            self.assertIsNotNone(history)

    def test_run_arrow_train_with_take_count_minus_one(self):
        """Test arrow training with take_count=-1 (entire dataset)."""
        with tempfile.TemporaryDirectory() as temp_dir:
            model_output_dir = pathlib.Path(temp_dir) / "models"
            exp = config.ArrowExperimentConfig(
                dataset=config.ArrowDatasetConfig(
                    data_dir=TEST_DATA_DIR,
                    val_data_dir=TEST_DATA_DIR,
                    batch_size=1,
                ),
                model=config.ArrowModelConfig.from_dict({}),
                run=config.ArrowRunConfig(
                    epoch=1,
                    take_count=-1,  # Entire dataset
                    model_output_dir=model_output_dir,
                ),
            )
            model, history = trainers.run_arrow_train_from_config(exp)
            self.assertIsNotNone(model)
            self.assertIsNotNone(history)

    def test_run_arrow_train_saves_config(self):
        """Test that arrow config is saved when callback_root_dir is set."""
        with tempfile.TemporaryDirectory() as temp_dir:
            callback_root_dir = pathlib.Path(temp_dir) / "callbacks"
            model_output_dir = pathlib.Path(temp_dir) / "models"
            exp = config.ArrowExperimentConfig(
                dataset=config.ArrowDatasetConfig(
                    data_dir=TEST_DATA_DIR,
                    val_data_dir=TEST_DATA_DIR,
                    batch_size=1,
                ),
                model=config.ArrowModelConfig.from_dict(
                    {"transformer": {"num_layers": 2}}
                ),
                run=config.ArrowRunConfig(
                    epoch=1,
                    take_count=1,
                    model_output_dir=model_output_dir,
                    callback_root_dir=callback_root_dir,
                ),
            )
            model, history = trainers.run_arrow_train_from_config(exp)

            # Check that config file was created
            log_dirs = [
                d
                for d in (pathlib.Path(callback_root_dir) / "logs").iterdir()
                if d.is_dir()
            ]
            self.assertGreater(len(log_dirs), 0)
            config_path = (
                pathlib.Path(callback_root_dir)
                / "logs"
                / log_dirs[0].name
                / "config.json"
            )
            self.assertTrue(config_path.exists())

            # Verify config can be loaded (nested model config)
            loaded_config = config.ArrowExperimentConfig.from_json(config_path)
            assert loaded_config.model.transformer is not None
            self.assertEqual(loaded_config.model.transformer.num_layers, 2)

    def test_backward_compatibility_run_train(self):
        """Test that old run_train API still works (backward compatibility)."""
        with tempfile.TemporaryDirectory() as temp_dir:
            callback_root_dir = pathlib.Path(temp_dir) / "callbacks"
            model_output_dir = pathlib.Path(temp_dir) / "models"
            # Use old API with kwargs
            model, history = trainers.run_train(
                data_dir=TEST_DATA_DIR,
                val_data_dir=TEST_DATA_DIR,
                batch_size=1,
                apply_temporal_augment=False,
                should_apply_spec_augment=False,
                use_gaussian_target=False,
                gaussian_sigma=0.0,
                model_params={"initial_filters": 8, "depth": 1},
                take_count=1,
                epoch=1,
                callback_root_dir=callback_root_dir,
                model_output_dir=model_output_dir,
            )
            self.assertIsNotNone(model)
            self.assertIsNotNone(history)

    def test_backward_compatibility_run_arrow_train(self):
        """Test that run_arrow_train with model_params dict (nested format) still works."""
        with tempfile.TemporaryDirectory() as temp_dir:
            callback_root_dir = pathlib.Path(temp_dir) / "callbacks"
            model_output_dir = pathlib.Path(temp_dir) / "models"
            model, history = trainers.run_arrow_train(
                data_dir=TEST_DATA_DIR,
                val_data_dir=TEST_DATA_DIR,
                batch_size=1,
                model_params={
                    "model_type": "transformer",
                    "transformer": {"num_layers": 1},
                },
                take_count=1,
                epoch=1,
                callback_root_dir=callback_root_dir,
                model_output_dir=model_output_dir,
            )
            self.assertIsNotNone(model)
            self.assertIsNotNone(history)

    def test_run_train_without_model_params_uses_defaults(self):
        """run_train works without model_params (uses default OnsetModelConfig)."""
        with tempfile.TemporaryDirectory() as temp_dir:
            model_output_dir = pathlib.Path(temp_dir) / "models"
            model, history = trainers.run_train(
                data_dir=TEST_DATA_DIR,
                val_data_dir=TEST_DATA_DIR,
                batch_size=1,
                apply_temporal_augment=False,
                should_apply_spec_augment=False,
                use_gaussian_target=False,
                gaussian_sigma=0.0,
                take_count=1,
                epoch=1,
                model_output_dir=model_output_dir,
            )
            self.assertIsNotNone(model)
            self.assertIsNotNone(history)

    def test_run_arrow_train_without_model_params_uses_defaults(self):
        """run_arrow_train works without model_params (uses default ArrowModelConfig)."""
        with tempfile.TemporaryDirectory() as temp_dir:
            model_output_dir = pathlib.Path(temp_dir) / "models"
            model, history = trainers.run_arrow_train(
                data_dir=TEST_DATA_DIR,
                val_data_dir=TEST_DATA_DIR,
                batch_size=1,
                take_count=1,
                epoch=1,
                model_output_dir=model_output_dir,
            )
            self.assertIsNotNone(model)

    def test_run_arrow_train_from_config_with_audio_snippets(self):
        """Arrow model and dataset work with snippet_half_frames > 0 (forward pass)."""

        dataset_config = config.ArrowDatasetConfig(
            data_dir=TEST_DATA_DIR,
            val_data_dir=TEST_DATA_DIR,
            batch_size=1,
            snippet_half_frames=5,
        )
        model_config = config.ArrowModelConfig(
            model_type="transformer",
            transformer=config.TransformerArrowParams(),
        )
        ds = datasets.create_arrow_dataset(
            data_dir=dataset_config.data_dir,
            batch_size=dataset_config.batch_size,
            snippet_half_frames=dataset_config.snippet_half_frames,
        )
        model = models.build_arrow_model_from_config(
            model_config,
            input_options=models.ArrowInputOptions(
                snippet_half_frames=dataset_config.snippet_half_frames
            ),
            output_options=models.ArrowOutputOptions(),
        )
        batch = next(iter(ds.take(1)))
        x, y = batch  # type: ignore[reportGeneralTypeIssues]
        out = model(x)
        self.assertEqual(out.shape[0], y.shape[0])
        self.assertEqual(out.shape[1], y.shape[1])
        self.assertEqual(out.shape[2], 256)
        self.assertEqual(len(model.inputs), 2)


class ExperimentNameHelperTests(unittest.TestCase):
    def test_get_onset_experiment_name_includes_key_hyperparameters(self):
        model_config = config.OnsetModelConfig(
            initial_filters=32,
            depth=3,
            dilation_rates=[1, 2, 4],
            kernel_size=5,
            dropout_rate=0.1,
        )
        name = trainers._get_onset_experiment_name(
            take_count=-1,
            apply_temporal_augment=True,
            should_apply_spec_augment=True,
            use_gaussian_target=True,
            gaussian_sigma=1.5,
            model_params=model_config,
        )
        self.assertIn("ONSET-take_all", name)
        self.assertIn("sigma_1_5", name)
        self.assertIn("temporal_augment", name)
        self.assertIn("spec_augment", name)
        self.assertIn("unet_wavenet", name)
        self.assertIn("filters_32", name)
        self.assertIn("depth_3", name)
        self.assertIn("kernel_5", name)
        self.assertIn("dropout_0_1", name)
        self.assertIn("dilations_1_2_4", name)

    def test_get_onset_experiment_name_includes_mert_feature_source(self):
        name = trainers._get_onset_experiment_name(
            take_count=1,
            apply_temporal_augment=False,
            should_apply_spec_augment=False,
            use_gaussian_target=False,
            gaussian_sigma=1.0,
            model_params=config.OnsetModelConfig(),
            feature_source=config.FeatureSource.MERT,
        )
        self.assertIn("mert", name)

    def test_get_onset_experiment_name_includes_waveform_feature_source(self):
        name = trainers._get_onset_experiment_name(
            take_count=1,
            apply_temporal_augment=False,
            should_apply_spec_augment=False,
            use_gaussian_target=False,
            gaussian_sigma=1.0,
            model_params=config.OnsetModelConfig(),
            feature_source=config.FeatureSource.WAVEFORM,
        )
        self.assertIn("waveform", name)

    def test_get_onset_experiment_name_handles_dilation_rate_edge_cases(self):
        class _DummyOnsetModelParams:
            def __init__(self, dilation_rates):
                base = config.OnsetModelConfig()
                self.onset_architecture = base.onset_architecture
                self.initial_filters = base.initial_filters
                self.depth = base.depth
                self.kernel_size = base.kernel_size
                self.dropout_rate = base.dropout_rate
                self.dilation_rates = dilation_rates
                self.recurrent_units = base.recurrent_units
                self.transformer_layers = base.transformer_layers
                self.transformer_heads = base.transformer_heads
                self.tcn_blocks = base.tcn_blocks

        params_none = _DummyOnsetModelParams(dilation_rates=None)
        name_none = trainers._get_onset_experiment_name(
            take_count=1,
            apply_temporal_augment=False,
            should_apply_spec_augment=False,
            use_gaussian_target=False,
            gaussian_sigma=0.0,
            model_params=typing.cast(config.OnsetModelConfig, params_none),
        )
        self.assertIn("dilations_N_A", name_none)

        params_str = _DummyOnsetModelParams(dilation_rates="custom")
        name_str = trainers._get_onset_experiment_name(
            take_count=1,
            apply_temporal_augment=False,
            should_apply_spec_augment=False,
            use_gaussian_target=False,
            gaussian_sigma=0.0,
            model_params=typing.cast(config.OnsetModelConfig, params_str),
        )
        self.assertIn("dilations_custom", name_str)

    def test_get_arrow_experiment_name_includes_snippets_and_aux_weights(self):
        exp = config.ArrowExperimentConfig(
            dataset=config.ArrowDatasetConfig(data_dir="d", val_data_dir="v"),
            model=config.ArrowModelConfig.from_dict(
                {
                    "model_type": "transformer",
                    "transformer": {
                        "num_layers": 2,
                        "d_model": 256,
                        "num_heads": 8,
                        "ff_dim": 512,
                        "dropout_rate": 0.2,
                    },
                }
            ),
            run=config.ArrowRunConfig(
                epoch=1,
                take_count=-1,
                model_output_dir="out",
                chart_validity_aux_weight=0.3,
                diversity_aux_weight=0.4,
            ),
        )
        name = trainers._get_arrow_experiment_name(exp)
        self.assertIn("ARROW", name)
        self.assertIn("transformer", name)
        self.assertIn("take_all", name)
        self.assertIn("att_layers_2", name)
        self.assertIn("d_model_256", name)
        self.assertIn("num_heads_8", name)
        self.assertIn("ff_dim_512", name)
        self.assertIn("dropout_0_2", name)
        self.assertIn("chart_val_aux_0_3", name)
        self.assertIn("diversity_aux_0_4", name)

    def test_get_arrow_experiment_name_transformer_includes_architecture_parts(self):
        """_get_arrow_experiment_name with transformer config includes ARROW and architecture parts."""
        exp = config.ArrowExperimentConfig(
            dataset=config.ArrowDatasetConfig(data_dir="d", val_data_dir="v"),
            model=config.ArrowModelConfig.from_dict(
                {
                    "model_type": "transformer",
                    "transformer": {
                        "num_layers": 1,
                        "d_model": 128,
                        "num_heads": 4,
                        "ff_dim": 256,
                        "dropout_rate": 0.0,
                    },
                }
            ),
            run=config.ArrowRunConfig(
                epoch=1,
                take_count=5,
                model_output_dir="out",
            ),
        )
        name = trainers._get_arrow_experiment_name(exp)
        self.assertIn("ARROW", name)
        self.assertIn("transformer", name)
        self.assertIn("att_layers_1", name)

    def test_get_arrow_experiment_name_omits_use_interval(self):
        """_get_arrow_experiment_name does not include 'use_interval' (input options are on dataset config)."""
        exp = config.ArrowExperimentConfig(
            dataset=config.ArrowDatasetConfig(data_dir="d", val_data_dir="v"),
            model=config.ArrowModelConfig.from_dict(
                {
                    "model_type": "transformer",
                    "transformer": {
                        "num_layers": 1,
                        "d_model": 128,
                        "num_heads": 4,
                        "ff_dim": 256,
                        "dropout_rate": 0.0,
                    },
                }
            ),
            run=config.ArrowRunConfig(
                epoch=1,
                take_count=5,
                model_output_dir="out",
            ),
        )
        name = trainers._get_arrow_experiment_name(exp)
        self.assertNotIn("use_interval", name)

    def test_get_arrow_experiment_name_includes_timing_jitter_when_enabled(self):
        """_get_arrow_experiment_name includes timing_jitter part when dataset_config.timing_jitter_sigma > 0."""
        model_config = config.ArrowModelConfig.from_dict(
            {"model_type": "transformer", "transformer": {}}
        )
        run_config = config.ArrowRunConfig(
            epoch=1, take_count=5, model_output_dir="out"
        )
        exp = config.ArrowExperimentConfig(
            dataset=config.ArrowDatasetConfig(
                data_dir="d", val_data_dir="v", timing_jitter_sigma=0.02
            ),
            model=model_config,
            run=run_config,
        )
        name = trainers._get_arrow_experiment_name(exp)
        self.assertIn("timing_jitter", name)
        self.assertIn("0_02", name)
        exp_no_jitter = config.ArrowExperimentConfig(
            dataset=config.ArrowDatasetConfig(data_dir="d", val_data_dir="v"),
            model=model_config,
            run=run_config,
        )
        name_no_jitter = trainers._get_arrow_experiment_name(exp_no_jitter)
        self.assertNotIn("timing_jitter", name_no_jitter)

    def test_get_arrow_experiment_name_differing_only_in_interval_encoding_differ(self):
        """Dataset configs differing only in interval_encoding are distinct; round-trip preserves enum."""
        dataset_default = config.ArrowDatasetConfig(
            data_dir="d",
            val_data_dir="v",
            interval_encoding=config.IntervalEncoding.DEFAULT,
        )
        dataset_log = config.ArrowDatasetConfig(
            data_dir="d",
            val_data_dir="v",
            interval_encoding=config.IntervalEncoding.LOG,
        )
        self.assertNotEqual(
            dataset_default.interval_encoding, dataset_log.interval_encoding
        )
        d = dataset_log.as_dict()
        self.assertEqual(d["interval_encoding"], "log")
        loaded = config.ArrowDatasetConfig.from_dict(d)
        self.assertEqual(loaded.interval_encoding, config.IntervalEncoding.LOG)

    def test_get_arrow_experiment_name_transformer_minimal(self):
        """Minimal transformer config produces a valid experiment name with ARROW and architecture."""
        exp = config.ArrowExperimentConfig(
            dataset=config.ArrowDatasetConfig(data_dir="d", val_data_dir="v"),
            model=config.ArrowModelConfig.from_dict(
                {
                    "model_type": "transformer",
                    "transformer": {"num_layers": 1, "d_model": 64},
                }
            ),
            run=config.ArrowRunConfig(epoch=1, take_count=1, model_output_dir="out"),
        )
        name = trainers._get_arrow_experiment_name(exp)
        self.assertIn("ARROW", name)
        self.assertIn("transformer", name)
        self.assertIn("att_layers_1", name)
        self.assertIn("d_model_64", name)

    def test_get_arrow_experiment_name_mlp_includes_hidden_dims_and_dropout(self):
        """_get_arrow_experiment_name with model_type=mlp includes mlp_* and dropout (mlp branch)."""
        exp = config.ArrowExperimentConfig(
            dataset=config.ArrowDatasetConfig(data_dir="d", val_data_dir="v"),
            model=config.ArrowModelConfig.from_dict(
                {
                    "model_type": "mlp",
                    "mlp": {
                        "hidden_dims": [256, 128],
                        "dropout_rate": 0.1,
                    },
                }
            ),
            run=config.ArrowRunConfig(
                epoch=1,
                take_count=10,
                model_output_dir="out",
            ),
        )
        name = trainers._get_arrow_experiment_name(exp)
        self.assertIn("ARROW", name)
        self.assertIn("mlp", name)
        self.assertIn("take_10", name)
        self.assertIn("mlp_256_128", name)
        self.assertIn("dropout_0_1", name)

    def test_get_arrow_experiment_name_lstm_includes_units_and_dropout(self):
        """_get_arrow_experiment_name with model_type=lstm includes lstm_* and dropout."""
        exp = config.ArrowExperimentConfig(
            dataset=config.ArrowDatasetConfig(data_dir="d", val_data_dir="v"),
            model=config.ArrowModelConfig.from_dict(
                {
                    "model_type": "lstm",
                    "lstm": {"units": 64, "num_layers": 2, "dropout_rate": 0.1},
                }
            ),
            run=config.ArrowRunConfig(
                epoch=1,
                take_count=10,
                model_output_dir="out",
            ),
        )
        name = trainers._get_arrow_experiment_name(exp)
        self.assertIn("ARROW", name)
        self.assertIn("lstm", name)
        self.assertIn("take_10", name)
        self.assertIn("lstm_units_64", name)
        self.assertIn("lstm_layers_2", name)
        self.assertIn("dropout_0_1", name)
        self.assertNotIn("lstm_bidir", name)

    def test_get_arrow_experiment_name_lstm_bidirectional_includes_bidir(self):
        """_get_arrow_experiment_name with lstm.bidirectional=True includes lstm_bidir."""
        exp = config.ArrowExperimentConfig(
            dataset=config.ArrowDatasetConfig(data_dir="d", val_data_dir="v"),
            model=config.ArrowModelConfig.from_dict(
                {
                    "model_type": "lstm",
                    "lstm": {
                        "units": 64,
                        "num_layers": 2,
                        "dropout_rate": 0.1,
                        "bidirectional": True,
                    },
                }
            ),
            run=config.ArrowRunConfig(
                epoch=1,
                take_count=10,
                model_output_dir="out",
            ),
        )
        name = trainers._get_arrow_experiment_name(exp)
        self.assertIn("lstm_bidir", name)
        self.assertIn("lstm_units_64", name)

    def test_get_arrow_experiment_name_gru_includes_units_and_dropout(self):
        """_get_arrow_experiment_name with model_type=gru includes gru_* and dropout."""
        exp = config.ArrowExperimentConfig(
            dataset=config.ArrowDatasetConfig(data_dir="d", val_data_dir="v"),
            model=config.ArrowModelConfig.from_dict(
                {
                    "model_type": "gru",
                    "gru": {"units": 64, "num_layers": 2, "dropout_rate": 0.1},
                }
            ),
            run=config.ArrowRunConfig(
                epoch=1,
                take_count=10,
                model_output_dir="out",
            ),
        )
        name = trainers._get_arrow_experiment_name(exp)
        self.assertIn("ARROW", name)
        self.assertIn("gru", name)
        self.assertIn("take_10", name)
        self.assertIn("gru_units_64", name)
        self.assertIn("gru_layers_2", name)
        self.assertIn("dropout_0_1", name)
        self.assertNotIn("gru_bidir", name)

    def test_get_arrow_experiment_name_gru_bidirectional_includes_bidir(self):
        """_get_arrow_experiment_name with gru.bidirectional=True includes gru_bidir."""
        exp = config.ArrowExperimentConfig(
            dataset=config.ArrowDatasetConfig(data_dir="d", val_data_dir="v"),
            model=config.ArrowModelConfig.from_dict(
                {
                    "model_type": "gru",
                    "gru": {
                        "units": 64,
                        "num_layers": 2,
                        "dropout_rate": 0.1,
                        "bidirectional": True,
                    },
                }
            ),
            run=config.ArrowRunConfig(
                epoch=1,
                take_count=10,
                model_output_dir="out",
            ),
        )
        name = trainers._get_arrow_experiment_name(exp)
        self.assertIn("gru_bidir", name)
        self.assertIn("gru_units_64", name)

    def test_get_arrow_experiment_name_uses_only_active_model_type(self):
        """When both transformer and mlp blocks are set, name uses only active model_type (no duplicate/conflicting params)."""
        # Simulate e.g. loading transformer config then overriding --model_type mlp without clearing transformer
        exp = config.ArrowExperimentConfig(
            dataset=config.ArrowDatasetConfig(data_dir="d", val_data_dir="v"),
            model=config.ArrowModelConfig(
                model_type="mlp",
                transformer=config.TransformerArrowParams(
                    num_layers=2, d_model=128, num_heads=4, ff_dim=512, dropout_rate=0.0
                ),
                mlp=config.MLPArrowParams(hidden_dims=[64, 32], dropout_rate=0.1),
            ),
            run=config.ArrowRunConfig(
                epoch=1,
                take_count=5,
                model_output_dir="out",
            ),
        )
        name = trainers._get_arrow_experiment_name(exp)
        self.assertIn("ARROW-mlp", name)
        self.assertIn("mlp_64_32", name)
        self.assertIn("dropout_0_1", name)
        self.assertNotIn("att_layers", name)
        self.assertNotIn("d_model", name)
        self.assertNotIn("num_heads", name)
        self.assertNotIn("ff_dim", name)

    def test_run_arrow_train_from_config_lstm_succeeds(self):
        """run_arrow_train_from_config with model_type lstm builds and runs (minimal step)."""
        with _temp_model_and_callback_dirs() as (model_output_dir, _):
            dataset_config, _, _ = _make_arrow_configs(model_output_dir)
            model_config = config.ArrowModelConfig.from_dict(
                {
                    "model_type": "lstm",
                    "lstm": {"units": 32, "num_layers": 1, "dropout_rate": 0.0},
                }
            )
            run_config = config.ArrowRunConfig(
                epoch=1,
                take_count=1,
                val_take_count=1,
                model_output_dir=model_output_dir,
                callback_root_dir=model_output_dir,
                model_name="",
            )
            exp = config.ArrowExperimentConfig(
                dataset=dataset_config, model=model_config, run=run_config
            )
            trainers.run_arrow_train_from_config(exp)
            keras_files = [
                f
                for f in [p.name for p in pathlib.Path(model_output_dir).iterdir()]
                if f.endswith(".keras")
            ]
            self.assertGreater(
                len(keras_files), 0, "Expected at least one .keras file in output dir"
            )

    def test_run_arrow_train_from_config_lstm_bidirectional_succeeds(self):
        """run_arrow_train_from_config with model_type lstm and bidirectional=True runs."""
        with _temp_model_and_callback_dirs() as (model_output_dir, _):
            dataset_config, _, _ = _make_arrow_configs(model_output_dir)
            model_config = config.ArrowModelConfig.from_dict(
                {
                    "model_type": "lstm",
                    "lstm": {
                        "units": 32,
                        "num_layers": 1,
                        "dropout_rate": 0.0,
                        "bidirectional": True,
                    },
                }
            )
            run_config = config.ArrowRunConfig(
                epoch=1,
                take_count=1,
                val_take_count=1,
                model_output_dir=model_output_dir,
                callback_root_dir=model_output_dir,
                model_name="",
            )
            exp = config.ArrowExperimentConfig(
                dataset=dataset_config, model=model_config, run=run_config
            )
            trainers.run_arrow_train_from_config(exp)
            keras_files = [
                f
                for f in [p.name for p in pathlib.Path(model_output_dir).iterdir()]
                if f.endswith(".keras")
            ]
            self.assertGreater(
                len(keras_files), 0, "Expected at least one .keras file in output dir"
            )

    def test_run_arrow_train_from_config_gru_succeeds(self):
        """run_arrow_train_from_config with model_type gru builds and runs (minimal step)."""
        with _temp_model_and_callback_dirs() as (model_output_dir, _):
            dataset_config, _, _ = _make_arrow_configs(model_output_dir)
            model_config = config.ArrowModelConfig.from_dict(
                {
                    "model_type": "gru",
                    "gru": {"units": 32, "num_layers": 1, "dropout_rate": 0.0},
                }
            )
            run_config = config.ArrowRunConfig(
                epoch=1,
                take_count=1,
                val_take_count=1,
                model_output_dir=model_output_dir,
                callback_root_dir=model_output_dir,
                model_name="",
            )
            exp = config.ArrowExperimentConfig(
                dataset=dataset_config, model=model_config, run=run_config
            )
            trainers.run_arrow_train_from_config(exp)
            keras_files = [
                f
                for f in [p.name for p in pathlib.Path(model_output_dir).iterdir()]
                if f.endswith(".keras")
            ]
            self.assertGreater(
                len(keras_files), 0, "Expected at least one .keras file in output dir"
            )

    def test_run_arrow_train_from_config_gru_bidirectional_succeeds(self):
        """run_arrow_train_from_config with model_type gru and bidirectional=True runs."""
        with _temp_model_and_callback_dirs() as (model_output_dir, _):
            dataset_config, _, _ = _make_arrow_configs(model_output_dir)
            model_config = config.ArrowModelConfig.from_dict(
                {
                    "model_type": "gru",
                    "gru": {
                        "units": 32,
                        "num_layers": 1,
                        "dropout_rate": 0.0,
                        "bidirectional": True,
                    },
                }
            )
            run_config = config.ArrowRunConfig(
                epoch=1,
                take_count=1,
                val_take_count=1,
                model_output_dir=model_output_dir,
                callback_root_dir=model_output_dir,
                model_name="",
            )
            exp = config.ArrowExperimentConfig(
                dataset=dataset_config, model=model_config, run=run_config
            )
            trainers.run_arrow_train_from_config(exp)
            keras_files = [
                f
                for f in [p.name for p in pathlib.Path(model_output_dir).iterdir()]
                if f.endswith(".keras")
            ]
            self.assertGreater(
                len(keras_files), 0, "Expected at least one .keras file in output dir"
            )

    def test_run_arrow_train_from_config_unknown_model_type_raises(self):
        """run_arrow_train_from_config raises ValueError for unsupported model_type."""
        with _temp_model_and_callback_dirs() as (model_output_dir, _):
            dataset_config, _, run_config = _make_arrow_configs(model_output_dir)
            model_config = config.ArrowModelConfig(model_type="unknown_arch")
            exp = config.ArrowExperimentConfig(
                dataset=dataset_config, model=model_config, run=run_config
            )
            with self.assertRaises(ValueError) as ctx:
                trainers.run_arrow_train_from_config(exp)
            self.assertIn("unknown_arch", str(ctx.exception))

    def test_run_arrow_train_from_config_raises_when_snippet_half_frames_mismatch(self):
        """run_arrow_train_from_config uses configs; input options are synced via ArrowExperimentConfig.from_dict."""
        with _temp_model_and_callback_dirs() as (model_output_dir, _):
            dataset_config, _, run_config = _make_arrow_configs(
                model_output_dir,
                dataset_kwargs={"snippet_half_frames": 0},
            )
            model_config = config.ArrowModelConfig.from_dict(
                {"model_type": "gru", "gru": {"units": 32}}
            )
            exp = config.ArrowExperimentConfig(
                dataset=dataset_config, model=model_config, run=run_config
            )
            trainers.run_arrow_train_from_config(exp)

    def test_run_arrow_train_from_config_with_use_interval_from_dataset(self):
        """When configs are built from ArrowExperimentConfig.from_dict, model gets input options from dataset."""
        with _temp_model_and_callback_dirs() as (model_output_dir, _):
            data = {
                "dataset": {
                    "data_dir": TEST_DATA_DIR,
                    "val_data_dir": TEST_DATA_DIR,
                    "batch_size": 1,
                    "use_interval": True,
                },
                "model": {
                    "model_type": "gru",
                    "gru": {"units": 32, "num_layers": 1, "dropout_rate": 0.0},
                },
                "run": {
                    "epoch": 1,
                    "take_count": 1,
                    "model_output_dir": model_output_dir,
                },
            }
            exp = config.ArrowExperimentConfig.from_dict(data)
            self.assertTrue(exp.dataset.use_interval)
            trainers.run_arrow_train_from_config(exp)


class ArrowLossTests(unittest.TestCase):
    """Integration tests for arrow training with focal, label smoothing, aux_interval, rejection."""

    def test_run_arrow_train_from_config_with_focal_loss_completes(self):
        """run_arrow_train_from_config with loss_type=focal builds and runs one epoch."""
        with _temp_model_and_callback_dirs(with_callbacks=True) as (
            model_output_dir,
            callback_root_dir,
        ):
            exp = _make_arrow_experiment_config(
                model_output_dir,
                run_kwargs={
                    "epoch": 1,
                    "take_count": 1,
                    "callback_root_dir": callback_root_dir,
                    "loss_type": "focal",
                    "focal_gamma": 2.0,
                },
            )
            model, history = trainers.run_arrow_train_from_config(exp)
        self.assertIsNotNone(model)
        self.assertIsNotNone(history)
        self.assertIn("val_main_loss", history.history)

    def test_run_arrow_train_from_config_with_rejection_threshold_includes_pass_rate_metric(
        self,
    ):
        """With chart_validity_rejection_threshold set, training logs chart_validity_pass_rate."""
        with _temp_model_and_callback_dirs(with_callbacks=True) as (
            model_output_dir,
            callback_root_dir,
        ):
            exp = _make_arrow_experiment_config(
                model_output_dir,
                run_kwargs={
                    "callback_root_dir": callback_root_dir,
                    "chart_validity_rejection_threshold": 0.99,
                    "chart_validity_rejection_scale": 10.0,
                },
            )
            _, history = trainers.run_arrow_train_from_config(exp)
        self.assertIn("chart_validity_pass_rate_0_99", history.history)
        self.assertIn("val_chart_validity_pass_rate_0_99", history.history)

    def test_run_arrow_train_from_config_with_label_smoothing_completes(self):
        """run_arrow_train_from_config with label_smoothing > 0 builds and runs one epoch."""
        with _temp_model_and_callback_dirs(with_callbacks=True) as (
            model_output_dir,
            callback_root_dir,
        ):
            exp = _make_arrow_experiment_config(
                model_output_dir,
                run_kwargs={
                    "epoch": 1,
                    "take_count": 1,
                    "callback_root_dir": callback_root_dir,
                    "label_smoothing": 0.1,
                },
            )
            model, history = trainers.run_arrow_train_from_config(exp)
        self.assertIsNotNone(model)
        self.assertIsNotNone(history)

    def test_run_arrow_train_from_config_with_aux_interval_weight_completes(self):
        """With aux_interval_weight > 0, model has two outputs and dataset yields aux targets/mask; compile uses aux loss."""
        dataset_config = config.ArrowDatasetConfig(
            data_dir=TEST_DATA_DIR,
            val_data_dir=TEST_DATA_DIR,
            batch_size=1,
            use_aux_interval_target=True,
        )
        model_config = config.ArrowModelConfig.from_dict(
            {
                "model_type": "gru",
                "gru": {"units": 32, "num_layers": 1, "dropout_rate": 0.0},
            }
        )
        run_config = config.ArrowRunConfig(
            epoch=1,
            take_count=1,
            model_output_dir=pathlib.Path(tempfile.gettempdir()) / "arrow_aux_test",
            aux_interval_weight=0.3,
        )
        use_aux_interval = run_config.aux_interval_weight > 0
        self.assertTrue(use_aux_interval)
        train_ds = datasets.create_arrow_dataset(
            data_dir=dataset_config.data_dir,
            batch_size=dataset_config.batch_size,
            use_aux_interval_target=use_aux_interval,
        )
        # Dataset batch (before _split_aux_batch) includes aux_interval_target and aux_interval_mask
        batch = next(iter(train_ds.take(1)))
        out, cols = batch
        self.assertIn("aux_interval_target", out)
        self.assertIn("aux_interval_mask", out)
        model = models.build_arrow_model_from_config(
            model_config,
            input_options=models.ArrowInputOptions(),
            output_options=models.ArrowOutputOptions(use_aux_interval=True),
        )
        # Model outputs are a dict keyed by output layer names when aux head enabled.
        self.assertEqual(
            sorted(model.output_names), ["aux_interval", "output_probabilities"]
        )
        # Compile with same loss setup as run_arrow_train_from_config (ensures loss and masking are wired)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-3, clipnorm=1.0),
            loss={
                "output_probabilities": keras.losses.SparseCategoricalCrossentropy(
                    ignore_class=0
                ),
                "aux_interval": losses.masked_mse_aux_interval,
            },
            loss_weights={"output_probabilities": 1.0, "aux_interval": 0.3},
        )
        # One train step with manual batch split (x, y, sample_weight) to verify loss computation.
        x = {
            k: v
            for k, v in out.items()
            if k not in ("aux_interval_target", "aux_interval_mask")
        }
        y = {"output_probabilities": cols, "aux_interval": out["aux_interval_target"]}
        sw = {
            "output_probabilities": tf.ones_like(cols, dtype=tf.float32),
            "aux_interval": out["aux_interval_mask"],
        }
        _ = model.train_on_batch(x, y, sample_weight=sw)
        self.assertIsNotNone(model)

    def test_run_arrow_train_from_config_with_aux_interval_weight_end_to_end(self):
        """run_arrow_train_from_config integrates aux_interval targets, mask, and sample_weight without errors."""
        with _temp_model_and_callback_dirs(with_callbacks=True) as (
            model_output_dir,
            callback_root_dir,
        ):
            exp = config.ArrowExperimentConfig(
                dataset=config.ArrowDatasetConfig(
                    data_dir=TEST_DATA_DIR,
                    val_data_dir=TEST_DATA_DIR,
                    batch_size=1,
                    use_aux_interval_target=True,
                ),
                model=config.ArrowModelConfig.from_dict(
                    {
                        "model_type": "gru",
                        "gru": {"units": 32, "num_layers": 1, "dropout_rate": 0.0},
                    }
                ),
                run=config.ArrowRunConfig(
                    epoch=1,
                    take_count=1,
                    model_output_dir=model_output_dir,
                    callback_root_dir=callback_root_dir,
                    aux_interval_weight=0.3,
                ),
            )
            model, history = trainers.run_arrow_train_from_config(exp)
        self.assertIsNotNone(model)
        self.assertIsNotNone(history)

    def test_run_arrow_train_from_config_uses_dataset_config_for_aux_interval_target(
        self,
    ):
        """create_arrow_dataset is called with dataset_config.use_aux_interval_target, not run_config.aux_interval_weight."""
        dataset_config = config.ArrowDatasetConfig(
            data_dir=TEST_DATA_DIR,
            val_data_dir=TEST_DATA_DIR,
            batch_size=1,
            use_aux_interval_target=True,
        )
        model_config = config.ArrowModelConfig.from_dict(
            {
                "model_type": "gru",
                "gru": {"units": 32, "num_layers": 1, "dropout_rate": 0.0},
            }
        )
        with mock.patch.object(
            datasets,
            "create_arrow_dataset",
            wraps=datasets.create_arrow_dataset,
            autospec=True,
        ) as create_ds:
            with _temp_model_and_callback_dirs(with_callbacks=False) as (
                model_output_dir,
                _,
            ):
                run_config = config.ArrowRunConfig(
                    epoch=1,
                    take_count=1,
                    model_output_dir=model_output_dir,
                    aux_interval_weight=0.0,
                )
                exp = config.ArrowExperimentConfig(
                    dataset=dataset_config, model=model_config, run=run_config
                )
                trainers.run_arrow_train_from_config(exp)
            self.assertGreaterEqual(create_ds.call_count, 2)
            for call in create_ds.call_args_list:
                kwargs = call.kwargs
                self.assertIn("use_aux_interval_target", kwargs)
                self.assertTrue(
                    kwargs["use_aux_interval_target"],
                    "create_arrow_dataset must be called with use_aux_interval_target "
                    "from dataset_config (True), not from run_config.aux_interval_weight.",
                )


class LearningRateScheduleTests(unittest.TestCase):
    def test_cosine_warmup_schedule_with_multiple_warmup_epochs(self):
        scheduler = trainers._build_cosine_warmup_schedule(
            total_epochs=5,
            warmup_epochs=2,
            lr_peak=0.1,
            lr_min=0.01,
        )
        lr0 = scheduler.schedule(0, 0.0)
        lr1 = scheduler.schedule(1, 0.0)
        lr2 = scheduler.schedule(2, 0.0)
        lr4 = scheduler.schedule(4, 0.0)

        # Warmup ramps from lr_min to lr_peak, then cosine decays toward lr_min.
        self.assertAlmostEqual(lr0, 0.01, places=7)
        self.assertAlmostEqual(lr1, 0.1, places=7)
        # First decay epoch starts at peak, then decreases.
        self.assertAlmostEqual(lr2, 0.1, places=7)
        self.assertLess(lr4, lr2)
        self.assertGreater(lr4, 0.01)

    def test_cosine_warmup_schedule_single_warmup_epoch_sets_peak_immediately(self):
        scheduler = trainers._build_cosine_warmup_schedule(
            total_epochs=3,
            warmup_epochs=1,
            lr_peak=0.05,
            lr_min=0.01,
        )
        lr0 = scheduler.schedule(0, 0.0)
        lr1 = scheduler.schedule(1, 0.0)
        lr2 = scheduler.schedule(2, 0.0)

        # With a single warmup epoch, we hit lr_peak immediately then decay.
        self.assertAlmostEqual(lr0, 0.05, places=7)
        self.assertAlmostEqual(lr1, 0.05, places=7)
        self.assertLess(lr2, lr1)
        self.assertGreater(lr2, 0.01)


if __name__ == "__main__":
    unittest.main()
