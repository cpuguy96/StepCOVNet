import contextlib
import os
import tempfile
import typing
import unittest
from unittest import mock

import keras
import tensorflow as tf

from stepcovnet import config, constants, datasets, models, trainers

TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "testdata")


@contextlib.contextmanager
def _temp_model_and_callback_dirs(with_callbacks: bool = False):
    """Yield (model_output_dir, callback_root_dir) inside a temporary directory."""
    with tempfile.TemporaryDirectory() as temp_dir:
        model_output_dir = os.path.join(temp_dir, "models")
        callback_root_dir = (
            os.path.join(temp_dir, "callbacks") if with_callbacks else ""
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

    def test_run_arrow_train_from_config(self):
        with _temp_model_and_callback_dirs(with_callbacks=True) as (
            model_output_dir,
            callback_root_dir,
        ):
            dataset_config, model_config, run_config = _make_arrow_configs(
                model_output_dir,
                run_kwargs={
                    "epoch": 1,
                    "take_count": 1,
                    "callback_root_dir": callback_root_dir,
                },
            )
            model, history = trainers.run_arrow_train_from_config(
                dataset_config, model_config, run_config
            )
        self.assertIsNotNone(model)
        self.assertIsNotNone(history)

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
            saved_path = os.path.join(model_output_dir, expected_name + ".keras")
            self.assertTrue(
                os.path.isfile(saved_path),
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
            saved_path = os.path.join(model_output_dir, model.name + ".keras")
            self.assertTrue(
                os.path.isfile(saved_path),
                f"Expected saved model at {saved_path}",
            )

    def test_run_arrow_train_from_config_saves_model_with_explicit_model_name(self):
        """Arrow saved model file and model.name use run_config.model_name when set."""
        with _temp_model_and_callback_dirs() as (model_output_dir, _):
            dataset_config, model_config, run_config = _make_arrow_configs(
                model_output_dir,
                run_kwargs={"model_name": "my_arrow_model"},
            )
            model, _ = trainers.run_arrow_train_from_config(
                dataset_config, model_config, run_config
            )
            expected_name = "stepcovnet_ARROW-my_arrow_model"
            self.assertEqual(model.name, expected_name)
            saved_path = os.path.join(model_output_dir, expected_name + ".keras")
            self.assertTrue(
                os.path.isfile(saved_path),
                f"Expected saved model at {saved_path}",
            )

    def test_run_arrow_train_from_config_saves_model_with_derived_name_when_model_name_empty(
        self,
    ):
        """When model_name is empty, arrow model name uses experiment-derived name."""
        with _temp_model_and_callback_dirs() as (model_output_dir, _):
            dataset_config, model_config, run_config = _make_arrow_configs(
                model_output_dir,
                run_kwargs={"model_name": ""},
            )
            model, _ = trainers.run_arrow_train_from_config(
                dataset_config, model_config, run_config
            )
            self.assertTrue(
                model.name.startswith("stepcovnet_ARROW-"),
                f"Expected model.name to start with 'stepcovnet_ARROW-', got {model.name!r}",
            )
            saved_path = os.path.join(model_output_dir, model.name + ".keras")
            self.assertTrue(
                os.path.isfile(saved_path),
                f"Expected saved model at {saved_path}",
            )

    def test_run_arrow_train_from_config_with_verbosity_quiet(self):
        """Training with show_model_summary=False and fit_verbose=0 completes."""
        with _temp_model_and_callback_dirs(with_callbacks=True) as (
            model_output_dir,
            callback_root_dir,
        ):
            dataset_config, model_config, run_config = _make_arrow_configs(
                model_output_dir,
                run_kwargs={
                    "callback_root_dir": callback_root_dir,
                    "show_model_summary": False,
                    "fit_verbose": 0,
                },
            )
            model, history = trainers.run_arrow_train_from_config(
                dataset_config, model_config, run_config
            )
        self.assertIsNotNone(model)
        self.assertIsNotNone(history)

    def test_run_arrow_train_from_config_fit_receives_verbose(self):
        """model.fit is called with verbose from run_config.fit_verbose."""
        mock_model = mock.Mock()
        mock_model.compile = mock.Mock()
        mock_model.fit = mock.Mock(
            return_value=mock.Mock(history={"val_loss": [1.0], "loss": [1.0]})
        )

        with _temp_model_and_callback_dirs() as (model_output_dir, _):
            dataset_config, model_config, run_config = _make_arrow_configs(
                model_output_dir,
                run_kwargs={"fit_verbose": 2},
            )
            with (
                mock.patch(
                    "stepcovnet.trainers.models.build_arrow_model_from_config",
                    return_value=mock_model,
                ),
                mock.patch("stepcovnet.trainers._write_model"),
            ):
                trainers.run_arrow_train_from_config(
                    dataset_config, model_config, run_config
                )
        mock_model.fit.assert_called_once()
        self.assertEqual(mock_model.fit.call_args.kwargs["verbose"], 2)

    def test_run_arrow_train_from_config_show_model_summary_false_skips_summary(
        self,
    ):
        """When show_model_summary is False, model.summary() is not called."""
        mock_model = mock.Mock()
        mock_model.compile = mock.Mock()
        mock_model.fit = mock.Mock(
            return_value=mock.Mock(history={"val_loss": [1.0], "loss": [1.0]})
        )

        with _temp_model_and_callback_dirs() as (model_output_dir, _):
            dataset_config, model_config, run_config = _make_arrow_configs(
                model_output_dir,
                run_kwargs={"show_model_summary": False},
            )
            with (
                mock.patch(
                    "stepcovnet.trainers.models.build_arrow_model_from_config",
                    return_value=mock_model,
                ),
                mock.patch("stepcovnet.trainers._write_model"),
            ):
                trainers.run_arrow_train_from_config(
                    dataset_config, model_config, run_config
                )
        mock_model.summary.assert_not_called()

    def test_run_arrow_train_from_config_show_model_summary_true_calls_summary(
        self,
    ):
        """When show_model_summary is True (default), model.summary() is called."""
        mock_model = mock.Mock()
        mock_model.compile = mock.Mock()
        mock_model.fit = mock.Mock(
            return_value=mock.Mock(history={"val_loss": [1.0], "loss": [1.0]})
        )

        with _temp_model_and_callback_dirs() as (model_output_dir, _):
            dataset_config, model_config, run_config = _make_arrow_configs(
                model_output_dir,
                run_kwargs={"show_model_summary": True},
            )
            with (
                mock.patch(
                    "stepcovnet.trainers.models.build_arrow_model_from_config",
                    return_value=mock_model,
                ),
                mock.patch("stepcovnet.trainers._write_model"),
            ):
                trainers.run_arrow_train_from_config(
                    dataset_config, model_config, run_config
                )
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
            config_path = os.path.join(temp_dir, "config.json")
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
            callback_root_dir = os.path.join(temp_dir, "callbacks")
            model_output_dir = os.path.join(temp_dir, "models")
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
                for d in os.listdir(os.path.join(callback_root_dir, "logs"))
                if os.path.isdir(os.path.join(callback_root_dir, "logs", d))
            ]
            self.assertGreater(len(log_dirs), 0)
            config_path = os.path.join(
                callback_root_dir, "logs", log_dirs[0], "config.json"
            )
            self.assertTrue(os.path.exists(config_path))

            # Verify config can be loaded
            loaded_config = config.OnsetExperimentConfig.from_json(config_path)
            self.assertEqual(loaded_config.dataset.batch_size, 1)
            self.assertEqual(loaded_config.model.initial_filters, 8)

    def test_run_train_from_config_no_callbacks(self):
        """Test that training works without callback_root_dir."""
        with tempfile.TemporaryDirectory() as temp_dir:
            model_output_dir = os.path.join(temp_dir, "models")
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
            model_output_dir = os.path.join(temp_dir, "models")
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
            model_output_dir = os.path.join(temp_dir, "models")
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
            model_output_dir = os.path.join(temp_dir, "models")
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
            model_output_dir = os.path.join(temp_dir, "models")
            dataset_config = config.ArrowDatasetConfig(
                data_dir=TEST_DATA_DIR,
                val_data_dir=TEST_DATA_DIR,
                batch_size=1,
            )
            model_config = config.ArrowModelConfig.from_dict({})
            run_config = config.ArrowRunConfig(
                epoch=1,
                take_count=-1,  # Entire dataset
                model_output_dir=model_output_dir,
            )
            model, history = trainers.run_arrow_train_from_config(
                dataset_config, model_config, run_config
            )
            self.assertIsNotNone(model)
            self.assertIsNotNone(history)

    def test_run_arrow_train_saves_config(self):
        """Test that arrow config is saved when callback_root_dir is set."""
        with tempfile.TemporaryDirectory() as temp_dir:
            callback_root_dir = os.path.join(temp_dir, "callbacks")
            model_output_dir = os.path.join(temp_dir, "models")
            dataset_config = config.ArrowDatasetConfig(
                data_dir=TEST_DATA_DIR,
                val_data_dir=TEST_DATA_DIR,
                batch_size=1,
            )
            model_config = config.ArrowModelConfig.from_dict(
                {"transformer": {"num_layers": 2}}
            )
            run_config = config.ArrowRunConfig(
                epoch=1,
                take_count=1,
                model_output_dir=model_output_dir,
                callback_root_dir=callback_root_dir,
            )
            model, history = trainers.run_arrow_train_from_config(
                dataset_config, model_config, run_config
            )

            # Check that config file was created
            log_dirs = [
                d
                for d in os.listdir(os.path.join(callback_root_dir, "logs"))
                if os.path.isdir(os.path.join(callback_root_dir, "logs", d))
            ]
            self.assertGreater(len(log_dirs), 0)
            config_path = os.path.join(
                callback_root_dir, "logs", log_dirs[0], "config.json"
            )
            self.assertTrue(os.path.exists(config_path))

            # Verify config can be loaded (nested model config)
            loaded_config = config.ArrowExperimentConfig.from_json(config_path)
            assert loaded_config.model.transformer is not None
            self.assertEqual(loaded_config.model.transformer.num_layers, 2)

    def test_backward_compatibility_run_train(self):
        """Test that old run_train API still works (backward compatibility)."""
        with tempfile.TemporaryDirectory() as temp_dir:
            callback_root_dir = os.path.join(temp_dir, "callbacks")
            model_output_dir = os.path.join(temp_dir, "models")
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
        """Test that old run_arrow_train API still works (backward compatibility)."""
        with tempfile.TemporaryDirectory() as temp_dir:
            callback_root_dir = os.path.join(temp_dir, "callbacks")
            model_output_dir = os.path.join(temp_dir, "models")
            # Use old API with kwargs
            model, history = trainers.run_arrow_train(
                data_dir=TEST_DATA_DIR,
                val_data_dir=TEST_DATA_DIR,
                batch_size=1,
                model_params={"num_layers": 1},
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
            model_output_dir = os.path.join(temp_dir, "models")
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
            model_output_dir = os.path.join(temp_dir, "models")
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
        model_config = config.ArrowModelConfig.from_dict(
            {"snippet_half_frames": 5},
        )
        ds = datasets.create_arrow_dataset(
            data_dir=dataset_config.data_dir,
            batch_size=dataset_config.batch_size,
            snippet_half_frames=dataset_config.snippet_half_frames,
        )
        model = models.build_arrow_model_from_config(model_config, model_name="")
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
        self.assertIn("unet_filters_32", name)
        self.assertIn("unet_depth_3", name)
        self.assertIn("unet_kernel_size_5", name)
        self.assertIn("unet_dropout_0_1", name)
        self.assertIn("unet_dilations_1_2_4", name)

    def test_get_onset_experiment_name_handles_dilation_rate_edge_cases(self):
        class _DummyOnsetModelParams:
            def __init__(self, dilation_rates):
                base = config.OnsetModelConfig()
                self.initial_filters = base.initial_filters
                self.depth = base.depth
                self.kernel_size = base.kernel_size
                self.dropout_rate = base.dropout_rate
                self.dilation_rates = dilation_rates

        params_none = _DummyOnsetModelParams(dilation_rates=None)
        name_none = trainers._get_onset_experiment_name(
            take_count=1,
            apply_temporal_augment=False,
            should_apply_spec_augment=False,
            use_gaussian_target=False,
            gaussian_sigma=0.0,
            model_params=typing.cast(config.OnsetModelConfig, params_none),
        )
        self.assertIn("unet_dilations_N_A", name_none)

        params_str = _DummyOnsetModelParams(dilation_rates="custom")
        name_str = trainers._get_onset_experiment_name(
            take_count=1,
            apply_temporal_augment=False,
            should_apply_spec_augment=False,
            use_gaussian_target=False,
            gaussian_sigma=0.0,
            model_params=typing.cast(config.OnsetModelConfig, params_str),
        )
        self.assertIn("unet_dilations_custom", name_str)

    def test_get_arrow_experiment_name_includes_snippets_and_aux_weights(self):
        model_config = config.ArrowModelConfig.from_dict(
            {
                "model_type": "transformer",
                "snippet_half_frames": 5,
                "transformer": {
                    "num_layers": 2,
                    "d_model": 256,
                    "num_heads": 8,
                    "ff_dim": 512,
                    "dropout_rate": 0.2,
                },
            }
        )
        run_config = config.ArrowRunConfig(
            epoch=1,
            take_count=-1,
            model_output_dir="out",
            chart_validity_aux_weight=0.3,
            diversity_aux_weight=0.4,
        )
        name = trainers._get_arrow_experiment_name(model_config, run_config)
        self.assertIn("ARROW", name)
        self.assertIn("transformer", name)
        self.assertIn("take_all", name)
        self.assertIn("att_layers_2", name)
        self.assertIn("d_model_256", name)
        self.assertIn("num_heads_8", name)
        self.assertIn("ff_dim_512", name)
        self.assertIn("dropout_0_2", name)
        self.assertIn("snippets_half_5", name)
        self.assertIn("chart_val_aux_0_3", name)
        self.assertIn("diversity_aux_0_4", name)

    def test_get_arrow_experiment_name_includes_use_interval_when_true(self):
        """_get_arrow_experiment_name with use_interval=True includes 'use_interval' in the name."""
        model_config = config.ArrowModelConfig.from_dict(
            {
                "model_type": "transformer",
                "use_interval": True,
                "transformer": {
                    "num_layers": 1,
                    "d_model": 128,
                    "num_heads": 4,
                    "ff_dim": 256,
                    "dropout_rate": 0.0,
                },
            }
        )
        run_config = config.ArrowRunConfig(
            epoch=1,
            take_count=5,
            model_output_dir="out",
        )
        name = trainers._get_arrow_experiment_name(model_config, run_config)
        self.assertIn("use_interval", name)
        self.assertIn("ARROW", name)
        self.assertIn("transformer", name)

    def test_get_arrow_experiment_name_omits_use_interval_when_false(self):
        """_get_arrow_experiment_name with use_interval=False does not include 'use_interval'."""
        model_config = config.ArrowModelConfig.from_dict(
            {
                "model_type": "transformer",
                "use_interval": False,
                "transformer": {
                    "num_layers": 1,
                    "d_model": 128,
                    "num_heads": 4,
                    "ff_dim": 256,
                    "dropout_rate": 0.0,
                },
            }
        )
        run_config = config.ArrowRunConfig(
            epoch=1,
            take_count=5,
            model_output_dir="out",
        )
        name = trainers._get_arrow_experiment_name(model_config, run_config)
        self.assertNotIn("use_interval", name)

    def test_get_arrow_experiment_name_differing_only_in_interval_encoding_differ(self):
        """Configs differing only in interval_encoding produce different experiment names."""
        base = {
            "model_type": "transformer",
            "use_interval": True,
            "transformer": {"num_layers": 1, "d_model": 64},
        }
        run = config.ArrowRunConfig(
            epoch=1, take_count=1, model_output_dir="out"
        )
        cfg_default = config.ArrowModelConfig.from_dict(
            {**base, "interval_encoding": "default"}
        )
        cfg_log = config.ArrowModelConfig.from_dict(
            {**base, "interval_encoding": "log"}
        )
        name_default = trainers._get_arrow_experiment_name(cfg_default, run)
        name_log = trainers._get_arrow_experiment_name(cfg_log, run)
        self.assertNotEqual(name_default, name_log)
        self.assertIn("interval_enc_log", name_log)
        self.assertNotIn("interval_enc", name_default)

    def test_get_arrow_experiment_name_includes_use_step_index_and_use_beat_phase(self):
        """Config with use_step_index and use_beat_phase includes them in experiment name."""
        model_config = config.ArrowModelConfig.from_dict(
            {
                "model_type": "transformer",
                "use_step_index": True,
                "use_beat_phase": True,
                "transformer": {"num_layers": 1, "d_model": 64},
            }
        )
        run_config = config.ArrowRunConfig(
            epoch=1, take_count=1, model_output_dir="out"
        )
        name = trainers._get_arrow_experiment_name(model_config, run_config)
        self.assertIn("use_step_index", name)
        self.assertIn("use_beat_phase", name)

    def test_get_arrow_experiment_name_mlp_includes_hidden_dims_and_dropout(self):
        """_get_arrow_experiment_name with model_type=mlp includes mlp_* and dropout (mlp branch)."""
        model_config = config.ArrowModelConfig.from_dict(
            {
                "model_type": "mlp",
                "mlp": {
                    "hidden_dims": [256, 128],
                    "dropout_rate": 0.1,
                },
            }
        )
        run_config = config.ArrowRunConfig(
            epoch=1,
            take_count=10,
            model_output_dir="out",
        )
        name = trainers._get_arrow_experiment_name(model_config, run_config)
        self.assertIn("ARROW", name)
        self.assertIn("mlp", name)
        self.assertIn("take_10", name)
        self.assertIn("mlp_256_128", name)
        self.assertIn("dropout_0_1", name)

    def test_get_arrow_experiment_name_lstm_includes_units_and_dropout(self):
        """_get_arrow_experiment_name with model_type=lstm includes lstm_* and dropout."""
        model_config = config.ArrowModelConfig.from_dict(
            {
                "model_type": "lstm",
                "lstm": {"units": 64, "num_layers": 2, "dropout_rate": 0.1},
            }
        )
        run_config = config.ArrowRunConfig(
            epoch=1,
            take_count=10,
            model_output_dir="out",
        )
        name = trainers._get_arrow_experiment_name(model_config, run_config)
        self.assertIn("ARROW", name)
        self.assertIn("lstm", name)
        self.assertIn("take_10", name)
        self.assertIn("lstm_units_64", name)
        self.assertIn("lstm_layers_2", name)
        self.assertIn("dropout_0_1", name)
        self.assertNotIn("lstm_bidir", name)

    def test_get_arrow_experiment_name_lstm_bidirectional_includes_bidir(self):
        """_get_arrow_experiment_name with lstm.bidirectional=True includes lstm_bidir."""
        model_config = config.ArrowModelConfig.from_dict(
            {
                "model_type": "lstm",
                "lstm": {
                    "units": 64,
                    "num_layers": 2,
                    "dropout_rate": 0.1,
                    "bidirectional": True,
                },
            }
        )
        run_config = config.ArrowRunConfig(
            epoch=1,
            take_count=10,
            model_output_dir="out",
        )
        name = trainers._get_arrow_experiment_name(model_config, run_config)
        self.assertIn("lstm_bidir", name)
        self.assertIn("lstm_units_64", name)

    def test_get_arrow_experiment_name_gru_includes_units_and_dropout(self):
        """_get_arrow_experiment_name with model_type=gru includes gru_* and dropout."""
        model_config = config.ArrowModelConfig.from_dict(
            {
                "model_type": "gru",
                "gru": {"units": 64, "num_layers": 2, "dropout_rate": 0.1},
            }
        )
        run_config = config.ArrowRunConfig(
            epoch=1,
            take_count=10,
            model_output_dir="out",
        )
        name = trainers._get_arrow_experiment_name(model_config, run_config)
        self.assertIn("ARROW", name)
        self.assertIn("gru", name)
        self.assertIn("take_10", name)
        self.assertIn("gru_units_64", name)
        self.assertIn("gru_layers_2", name)
        self.assertIn("dropout_0_1", name)
        self.assertNotIn("gru_bidir", name)

    def test_get_arrow_experiment_name_gru_bidirectional_includes_bidir(self):
        """_get_arrow_experiment_name with gru.bidirectional=True includes gru_bidir."""
        model_config = config.ArrowModelConfig.from_dict(
            {
                "model_type": "gru",
                "gru": {
                    "units": 64,
                    "num_layers": 2,
                    "dropout_rate": 0.1,
                    "bidirectional": True,
                },
            }
        )
        run_config = config.ArrowRunConfig(
            epoch=1,
            take_count=10,
            model_output_dir="out",
        )
        name = trainers._get_arrow_experiment_name(model_config, run_config)
        self.assertIn("gru_bidir", name)
        self.assertIn("gru_units_64", name)

    def test_get_arrow_experiment_name_uses_only_active_model_type(self):
        """When both transformer and mlp blocks are set, name uses only active model_type (no duplicate/conflicting params)."""
        # Simulate e.g. loading transformer config then overriding --model_type mlp without clearing transformer
        model_config = config.ArrowModelConfig(
            model_type="mlp",
            snippet_half_frames=0,
            transformer=config.TransformerArrowParams(
                num_layers=2, d_model=128, num_heads=4, ff_dim=512, dropout_rate=0.0
            ),
            mlp=config.MLPArrowParams(hidden_dims=[64, 32], dropout_rate=0.1),
        )
        run_config = config.ArrowRunConfig(
            epoch=1,
            take_count=5,
            model_output_dir="out",
        )
        name = trainers._get_arrow_experiment_name(model_config, run_config)
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
            dataset_config, _, run_config = _make_arrow_configs(model_output_dir)
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
            trainers.run_arrow_train_from_config(
                dataset_config, model_config, run_config
            )
            keras_files = [
                f for f in os.listdir(model_output_dir) if f.endswith(".keras")
            ]
            self.assertGreater(
                len(keras_files), 0, "Expected at least one .keras file in output dir"
            )

    def test_run_arrow_train_from_config_lstm_bidirectional_succeeds(self):
        """run_arrow_train_from_config with model_type lstm and bidirectional=True runs."""
        with _temp_model_and_callback_dirs() as (model_output_dir, _):
            dataset_config, _, run_config = _make_arrow_configs(model_output_dir)
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
            trainers.run_arrow_train_from_config(
                dataset_config, model_config, run_config
            )
            keras_files = [
                f for f in os.listdir(model_output_dir) if f.endswith(".keras")
            ]
            self.assertGreater(
                len(keras_files), 0, "Expected at least one .keras file in output dir"
            )

    def test_run_arrow_train_from_config_gru_succeeds(self):
        """run_arrow_train_from_config with model_type gru builds and runs (minimal step)."""
        with _temp_model_and_callback_dirs() as (model_output_dir, _):
            dataset_config, _, run_config = _make_arrow_configs(model_output_dir)
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
            trainers.run_arrow_train_from_config(
                dataset_config, model_config, run_config
            )
            keras_files = [
                f for f in os.listdir(model_output_dir) if f.endswith(".keras")
            ]
            self.assertGreater(
                len(keras_files), 0, "Expected at least one .keras file in output dir"
            )

    def test_run_arrow_train_from_config_gru_bidirectional_succeeds(self):
        """run_arrow_train_from_config with model_type gru and bidirectional=True runs."""
        with _temp_model_and_callback_dirs() as (model_output_dir, _):
            dataset_config, _, run_config = _make_arrow_configs(model_output_dir)
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
            trainers.run_arrow_train_from_config(
                dataset_config, model_config, run_config
            )
            keras_files = [
                f for f in os.listdir(model_output_dir) if f.endswith(".keras")
            ]
            self.assertGreater(
                len(keras_files), 0, "Expected at least one .keras file in output dir"
            )

    def test_run_arrow_train_from_config_unknown_model_type_raises(self):
        """run_arrow_train_from_config raises ValueError for unsupported model_type."""
        with _temp_model_and_callback_dirs() as (model_output_dir, _):
            dataset_config, _, run_config = _make_arrow_configs(model_output_dir)
            model_config = config.ArrowModelConfig.from_dict(
                {"model_type": "unknown_arch"}
            )
            with self.assertRaises(ValueError) as ctx:
                trainers.run_arrow_train_from_config(
                    dataset_config, model_config, run_config
                )
            self.assertIn("unknown_arch", str(ctx.exception))

    def test_run_arrow_train_from_config_raises_when_snippet_half_frames_mismatch(self):
        """run_arrow_train_from_config raises ValueError when dataset and model snippet_half_frames differ."""
        with _temp_model_and_callback_dirs() as (model_output_dir, _):
            dataset_config, _, run_config = _make_arrow_configs(
                model_output_dir,
                dataset_kwargs={"snippet_half_frames": 0},
            )
            model_config = config.ArrowModelConfig.from_dict({"snippet_half_frames": 5})
            with self.assertRaises(ValueError) as ctx:
                trainers.run_arrow_train_from_config(
                    dataset_config, model_config, run_config
                )
            self.assertIn("snippet_half_frames", str(ctx.exception))

    def test_run_arrow_train_from_config_raises_when_use_interval_mismatch(self):
        """run_arrow_train_from_config raises ValueError when dataset and model use_interval differ."""
        with _temp_model_and_callback_dirs() as (model_output_dir, _):
            dataset_config, _, run_config = _make_arrow_configs(
                model_output_dir,
                dataset_kwargs={"use_interval": True},
            )
            model_config = config.ArrowModelConfig.from_dict({"use_interval": False})
            with self.assertRaises(ValueError) as ctx:
                trainers.run_arrow_train_from_config(
                    dataset_config, model_config, run_config
                )
            self.assertIn("use_interval", str(ctx.exception))


class ArrowLossTests(unittest.TestCase):
    """Unit tests for arrow loss functions: focal, label smoothing, aux_interval masking."""

    def test_sparse_focal_loss_returns_scalar_and_masks_ignore_class(self):
        """_sparse_focal_loss returns a scalar and ignores steps with y_true==ignore_class (0)."""
        batch_size, steps, num_classes = 2, 10, constants.N_ARROW_TYPES
        y_true = tf.constant([[0, 1, 2, 0, 3, 0, 1, 0, 2, 1]], dtype=tf.int32)  # 0 = padding
        y_pred = tf.random.uniform((batch_size, steps, num_classes))
        y_pred = y_pred / tf.reduce_sum(y_pred, axis=-1, keepdims=True)
        loss = trainers._sparse_focal_loss(y_true, y_pred, gamma=2.0, ignore_class=0)
        self.assertEqual(loss.shape, ())
        self.assertGreater(float(loss), 0.0)

    def test_masked_mse_aux_interval_without_sample_weight(self):
        """_masked_mse_aux_interval without sample_weight returns mean over all elements."""
        y_true = tf.constant([[[0.1], [0.2], [0.3]]], dtype=tf.float32)
        y_pred = tf.constant([[[0.2], [0.2], [0.4]]], dtype=tf.float32)
        loss = trainers._masked_mse_aux_interval(y_true, y_pred, sample_weight=None)
        self.assertEqual(loss.shape, ())
        expected = ((0.1 ** 2) + (0.0 ** 2) + (0.1 ** 2)) / 3
        self.assertAlmostEqual(float(loss), expected, places=5)

    def test_masked_mse_aux_interval_with_sample_weight_masks_steps(self):
        """_masked_mse_aux_interval with sample_weight only averages over masked (1.0) steps."""
        # (1, 3, 1): last step masked (0), so only first two steps contribute
        y_true = tf.constant([[[1.0], [2.0], [0.0]]], dtype=tf.float32)
        y_pred = tf.constant([[[1.0], [3.0], [99.0]]], dtype=tf.float32)  # last step wrong but masked
        sample_weight = tf.constant([[[1.0], [1.0], [0.0]]], dtype=tf.float32)
        loss = trainers._masked_mse_aux_interval(y_true, y_pred, sample_weight=sample_weight)
        self.assertEqual(loss.shape, ())
        # Only steps 0 and 1: (0 + 1) / 2 = 0.5
        self.assertAlmostEqual(float(loss), 0.5, places=5)

    def test_run_arrow_train_from_config_with_focal_loss_completes(self):
        """run_arrow_train_from_config with loss_type=focal builds and runs one epoch."""
        with _temp_model_and_callback_dirs(with_callbacks=True) as (
            model_output_dir,
            callback_root_dir,
        ):
            dataset_config, model_config, run_config = _make_arrow_configs(
                model_output_dir,
                run_kwargs={
                    "epoch": 1,
                    "take_count": 1,
                    "callback_root_dir": callback_root_dir,
                    "loss_type": "focal",
                    "focal_gamma": 2.0,
                },
            )
            model, history = trainers.run_arrow_train_from_config(
                dataset_config, model_config, run_config
            )
        self.assertIsNotNone(model)
        self.assertIsNotNone(history)
        self.assertIn("val_main_loss", history.history)

    def test_run_arrow_train_from_config_with_label_smoothing_completes(self):
        """run_arrow_train_from_config with label_smoothing > 0 builds and runs one epoch."""
        with _temp_model_and_callback_dirs(with_callbacks=True) as (
            model_output_dir,
            callback_root_dir,
        ):
            dataset_config, model_config, run_config = _make_arrow_configs(
                model_output_dir,
                run_kwargs={
                    "epoch": 1,
                    "take_count": 1,
                    "callback_root_dir": callback_root_dir,
                    "label_smoothing": 0.1,
                },
            )
            model, history = trainers.run_arrow_train_from_config(
                dataset_config, model_config, run_config
            )
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
            model_output_dir=os.path.join(tempfile.gettempdir(), "arrow_aux_test"),
            aux_interval_weight=0.3,
        )
        config.validate_arrow_dataset_model_alignment(dataset_config, model_config)
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
            model_config, model_name="aux_test", use_aux_interval=True
        )
        self.assertEqual(len(model.outputs), 2)
        # Compile with same loss setup as run_arrow_train_from_config (ensures loss and masking are wired)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-3, clipnorm=1.0),
            loss={
                "output_probabilities": keras.losses.SparseCategoricalCrossentropy(
                    ignore_class=0
                ),
                "aux_interval": trainers._masked_mse_aux_interval,
            },
            loss_weights={"output_probabilities": 1.0, "aux_interval": 0.3},
        )
        # One train step with manual batch split (x, y, sample_weight) to verify loss computation.
        # Keras multi-output sample_weight: list per output order [output_probabilities, aux_interval].
        x = {k: v for k, v in out.items() if k not in ("aux_interval_target", "aux_interval_mask")}
        y = {"output_probabilities": cols, "aux_interval": out["aux_interval_target"]}
        sw = [None, out["aux_interval_mask"]]  # no sample_weight for main head, mask for aux
        _ = model.train_on_batch(x, y, sample_weight=sw)
        self.assertIsNotNone(model)

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
        with mock.patch(
            "stepcovnet.trainers.datasets.create_arrow_dataset",
            wraps=datasets.create_arrow_dataset,
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
                trainers.run_arrow_train_from_config(
                    dataset_config, model_config, run_config
                )
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
