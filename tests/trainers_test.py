import os
import tempfile
import unittest

from stepcovnet import config
from stepcovnet import trainers

TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "testdata")


class TrainersTest(unittest.TestCase):
    def test_run_train(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            callback_root_dir = os.path.join(temp_dir, "callbacks")
            model_output_dir = os.path.join(temp_dir, "models")
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
                epoch=3,
                callback_root_dir=callback_root_dir,
                model_output_dir=model_output_dir,
            )
        self.assertIsNotNone(model)
        self.assertIsNotNone(history)

    @unittest.skip("Test takes too long.")
    def test_run_train_overfits_single_song(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            callback_root_dir = os.path.join(temp_dir, "callbacks")
            model_output_dir = os.path.join(temp_dir, "models")
            model, history = trainers.run_train(
                data_dir=TEST_DATA_DIR,
                val_data_dir=TEST_DATA_DIR,
                batch_size=1,
                apply_temporal_augment=False,
                should_apply_spec_augment=False,
                use_gaussian_target=False,
                gaussian_sigma=0.0,
                model_params={
                    "initial_filters": 16,
                    "depth": 2,
                    "dilation_rates": [1, 2, 4, 8],
                    "dropout_rate": 0.0,
                },
                take_count=1,
                epoch=300,
                callback_root_dir=callback_root_dir,
                model_output_dir=model_output_dir,
            )
        self.assertIsNotNone(model)
        self.assertIsNotNone(history)
        self.assertTrue("val_onset_f1_score" in history.history)
        self.assertAlmostEqual(history.history["val_onset_f1_score"][-1], 1.0, places=2)

    def test_run_arrow_train(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            callback_root_dir = os.path.join(temp_dir, "callbacks")
            model_output_dir = os.path.join(temp_dir, "models")
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
        with tempfile.TemporaryDirectory() as temp_dir:
            callback_root_dir = os.path.join(temp_dir, "callbacks")
            model_output_dir = os.path.join(temp_dir, "models")
            dataset_config = config.OnsetDatasetConfig(
                data_dir=TEST_DATA_DIR,
                val_data_dir=TEST_DATA_DIR,
                batch_size=1,
                apply_temporal_augment=False,
                should_apply_spec_augment=False,
                use_gaussian_target=False,
                gaussian_sigma=0.0,
            )
            model_config = config.OnsetModelConfig(
                initial_filters=8,
                depth=1,
                dilation_rates=[1, 2],
                dropout_rate=0.0,
            )
            run_config = config.RunConfig(
                epoch=3,
                take_count=1,
                model_output_dir=model_output_dir,
                callback_root_dir=callback_root_dir,
            )
            model, history = trainers.run_train_from_config(
                dataset_config, model_config, run_config
            )
        self.assertIsNotNone(model)
        self.assertIsNotNone(history)

    def test_run_arrow_train_from_config(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            callback_root_dir = os.path.join(temp_dir, "callbacks")
            model_output_dir = os.path.join(temp_dir, "models")
            dataset_config = config.ArrowDatasetConfig(
                data_dir=TEST_DATA_DIR,
                val_data_dir=TEST_DATA_DIR,
                batch_size=1,
            )
            model_config = config.ArrowModelConfig()
            run_config = config.RunConfig(
                epoch=1,
                take_count=1,
                model_output_dir=model_output_dir,
                callback_root_dir=callback_root_dir,
            )
            model, history = trainers.run_arrow_train_from_config(
                dataset_config, model_config, run_config
            )
        self.assertIsNotNone(model)
        self.assertIsNotNone(history)

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
            self.assertEqual(loaded_config.dataset.batch_size, dataset_config.batch_size)
            self.assertEqual(
                loaded_config.model.initial_filters, model_config.initial_filters
            )
            self.assertEqual(loaded_config.model.dilation_rates, model_config.dilation_rates)
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
            model_config = config.ArrowModelConfig()
            run_config = config.RunConfig(
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
            model_config = config.ArrowModelConfig(num_layers=2)
            run_config = config.RunConfig(
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

            # Verify config can be loaded
            loaded_config = config.ArrowExperimentConfig.from_json(config_path)
            self.assertEqual(loaded_config.model.num_layers, 2)

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


if __name__ == "__main__":
    unittest.main()
