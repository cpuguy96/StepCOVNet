import json
import os
import tempfile
import unittest

from stepcovnet import config


class OnsetDatasetConfigTest(unittest.TestCase):
    def test_create_with_required_fields(self):
        """Test creating config with only required fields."""
        cfg = config.OnsetDatasetConfig(data_dir="data/train", val_data_dir="data/val")
        self.assertEqual(cfg.data_dir, "data/train")
        self.assertEqual(cfg.val_data_dir, "data/val")
        self.assertEqual(cfg.batch_size, 1)  # default

    def test_create_with_all_fields(self):
        """Test creating config with all fields."""
        cfg = config.OnsetDatasetConfig(
            data_dir="data/train",
            val_data_dir="data/val",
            batch_size=4,
            apply_temporal_augment=True,
            should_apply_spec_augment=True,
            use_gaussian_target=True,
            gaussian_sigma=1.5,
        )
        self.assertEqual(cfg.batch_size, 4)
        self.assertTrue(cfg.apply_temporal_augment)
        self.assertTrue(cfg.should_apply_spec_augment)
        self.assertTrue(cfg.use_gaussian_target)
        self.assertEqual(cfg.gaussian_sigma, 1.5)

    def test_as_dict(self):
        """Test converting config to dictionary."""
        cfg = config.OnsetDatasetConfig(
            data_dir="data/train",
            val_data_dir="data/val",
            batch_size=2,
        )
        d = cfg.as_dict()
        self.assertIsInstance(d, dict)
        self.assertEqual(d["data_dir"], "data/train")
        self.assertEqual(d["val_data_dir"], "data/val")
        self.assertEqual(d["batch_size"], 2)

    def test_from_dict(self):
        """Test creating config from dictionary."""
        data = {
            "data_dir": "data/train",
            "val_data_dir": "data/val",
            "batch_size": 8,
            "apply_temporal_augment": True,
        }
        cfg = config.OnsetDatasetConfig.from_dict(data)
        self.assertEqual(cfg.data_dir, "data/train")
        self.assertEqual(cfg.batch_size, 8)
        self.assertTrue(cfg.apply_temporal_augment)
        self.assertFalse(cfg.should_apply_spec_augment)  # default


class ArrowDatasetConfigTest(unittest.TestCase):
    def test_create_with_required_fields(self):
        """Test creating config with only required fields."""
        cfg = config.ArrowDatasetConfig(data_dir="data/train", val_data_dir="data/val")
        self.assertEqual(cfg.data_dir, "data/train")
        self.assertEqual(cfg.val_data_dir, "data/val")
        self.assertEqual(cfg.batch_size, 1)  # default
        self.assertEqual(cfg.snippet_half_frames, 0)  # default: no snippets

    def test_as_dict(self):
        """Test converting config to dictionary."""
        cfg = config.ArrowDatasetConfig(
            data_dir="data/train", val_data_dir="data/val", batch_size=4
        )
        d = cfg.as_dict()
        self.assertEqual(d["batch_size"], 4)

    def test_from_dict(self):
        """Test creating config from dictionary."""
        data = {"data_dir": "data/train", "val_data_dir": "data/val", "batch_size": 2}
        cfg = config.ArrowDatasetConfig.from_dict(data)
        self.assertEqual(cfg.batch_size, 2)

    def test_from_dict_with_snippet_half_frames(self):
        """Test creating config with snippet_half_frames (use_audio_snippets stripped for backwards compat)."""
        data = {
            "data_dir": "data/train",
            "val_data_dir": "data/val",
            "snippet_half_frames": 5,
        }
        cfg = config.ArrowDatasetConfig.from_dict(data)
        self.assertEqual(cfg.snippet_half_frames, 5)


class OnsetModelConfigTest(unittest.TestCase):
    def test_create_with_defaults(self):
        """Test creating config with default values."""
        cfg = config.OnsetModelConfig()
        self.assertEqual(cfg.initial_filters, 16)
        self.assertEqual(cfg.depth, 2)
        self.assertEqual(cfg.dilation_rates, [1, 2, 4, 8])
        self.assertEqual(cfg.kernel_size, 3)
        self.assertEqual(cfg.dropout_rate, 0.0)

    def test_create_with_custom_values(self):
        """Test creating config with custom values."""
        cfg = config.OnsetModelConfig(
            initial_filters=32,
            depth=3,
            dilation_rates=[1, 2, 4],
            kernel_size=5,
            dropout_rate=0.2,
        )
        self.assertEqual(cfg.initial_filters, 32)
        self.assertEqual(cfg.depth, 3)
        self.assertEqual(cfg.dilation_rates, [1, 2, 4])
        self.assertEqual(cfg.kernel_size, 5)
        self.assertEqual(cfg.dropout_rate, 0.2)

    def test_as_dict(self):
        """Test converting config to dictionary."""
        cfg = config.OnsetModelConfig(initial_filters=8, depth=1)
        d = cfg.as_dict()
        self.assertEqual(d["initial_filters"], 8)
        self.assertEqual(d["depth"], 1)
        self.assertEqual(d["dilation_rates"], [1, 2, 4, 8])

    def test_from_dict(self):
        """Test creating config from dictionary."""
        data = {"initial_filters": 32, "depth": 3, "dropout_rate": 0.1}
        cfg = config.OnsetModelConfig.from_dict(data)
        self.assertEqual(cfg.initial_filters, 32)
        self.assertEqual(cfg.depth, 3)
        self.assertEqual(cfg.dropout_rate, 0.1)
        # Should use defaults for missing fields
        self.assertEqual(cfg.kernel_size, 3)


class ArrowModelConfigTest(unittest.TestCase):
    def test_create_with_defaults(self):
        """Test creating config with default values."""
        cfg = config.ArrowModelConfig()
        self.assertEqual(cfg.num_layers, 1)
        self.assertEqual(cfg.d_model, 128)
        self.assertEqual(cfg.num_heads, 4)
        self.assertEqual(cfg.ff_dim, 512)
        self.assertEqual(cfg.dropout_rate, 0.0)
        self.assertEqual(cfg.snippet_half_frames, 0)  # default: no snippets

    def test_as_dict(self):
        """Test converting config to dictionary."""
        cfg = config.ArrowModelConfig(num_layers=2, d_model=256)
        d = cfg.as_dict()
        self.assertEqual(d["num_layers"], 2)
        self.assertEqual(d["d_model"], 256)

    def test_from_dict(self):
        """Test creating config from dictionary."""
        data = {"num_layers": 3, "dropout_rate": 0.2}
        cfg = config.ArrowModelConfig.from_dict(data)
        self.assertEqual(cfg.num_layers, 3)
        self.assertEqual(cfg.dropout_rate, 0.2)
        # Should use defaults
        self.assertEqual(cfg.d_model, 128)

    def test_from_dict_with_snippet_half_frames(self):
        """Test creating config with snippet_half_frames."""
        data = {"snippet_half_frames": 5}
        cfg = config.ArrowModelConfig.from_dict(data)
        self.assertEqual(cfg.snippet_half_frames, 5)


class RunConfigTest(unittest.TestCase):
    def test_create_with_required_fields(self):
        """Test creating config with only required fields."""
        cfg = config.RunConfig(epoch=10, take_count=100, model_output_dir="out")
        self.assertEqual(cfg.epoch, 10)
        self.assertEqual(cfg.take_count, 100)
        self.assertEqual(cfg.model_output_dir, "out")
        self.assertEqual(cfg.callback_root_dir, "")  # default
        self.assertIsNone(cfg.seed)  # default

    def test_create_with_all_fields(self):
        """Test creating config with all fields."""
        cfg = config.RunConfig(
            epoch=20,
            take_count=-1,
            model_output_dir="out",
            callback_root_dir="callbacks",
            model_name="test_model",
            seed=42,
        )
        self.assertEqual(cfg.epoch, 20)
        self.assertEqual(cfg.take_count, -1)
        self.assertEqual(cfg.model_name, "test_model")
        self.assertEqual(cfg.seed, 42)

    def test_as_dict(self):
        """Test converting config to dictionary."""
        cfg = config.RunConfig(epoch=5, take_count=50, model_output_dir="out", seed=123)
        d = cfg.as_dict()
        self.assertEqual(d["epoch"], 5)
        self.assertEqual(d["seed"], 123)

    def test_from_dict(self):
        """Test creating config from dictionary."""
        data = {
            "epoch": 15,
            "take_count": 200,
            "model_output_dir": "models",
            "callback_root_dir": "cb",
        }
        cfg = config.RunConfig.from_dict(data)
        self.assertEqual(cfg.epoch, 15)
        self.assertEqual(cfg.callback_root_dir, "cb")

    def test_aux_weights_default_zero(self):
        """Default aux weights are 0 and valid."""
        cfg = config.RunConfig(epoch=1, take_count=1, model_output_dir="out")
        self.assertEqual(cfg.chart_validity_aux_weight, 0.0)
        self.assertEqual(cfg.diversity_aux_weight, 0.0)

    def test_aux_weights_accept_non_negative(self):
        """Non-negative aux weights are accepted."""
        cfg = config.RunConfig(
            epoch=1,
            take_count=1,
            model_output_dir="out",
            chart_validity_aux_weight=0.5,
            diversity_aux_weight=0.2,
        )
        self.assertEqual(cfg.chart_validity_aux_weight, 0.5)
        self.assertEqual(cfg.diversity_aux_weight, 0.2)
        cfg_zero = config.RunConfig(
            epoch=1,
            take_count=1,
            model_output_dir="out",
            chart_validity_aux_weight=0.0,
            diversity_aux_weight=0.0,
        )
        self.assertEqual(cfg_zero.chart_validity_aux_weight, 0.0)
        self.assertEqual(cfg_zero.diversity_aux_weight, 0.0)

    def test_negative_chart_validity_aux_weight_raises(self):
        """chart_validity_aux_weight < 0 raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            config.RunConfig(
                epoch=1,
                take_count=1,
                model_output_dir="out",
                chart_validity_aux_weight=-0.1,
            )
        self.assertIn("chart_validity_aux_weight", str(ctx.exception))
        self.assertIn("-0.1", str(ctx.exception))

    def test_negative_diversity_aux_weight_raises(self):
        """diversity_aux_weight < 0 raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            config.RunConfig(
                epoch=1,
                take_count=1,
                model_output_dir="out",
                diversity_aux_weight=-1.0,
            )
        self.assertIn("diversity_aux_weight", str(ctx.exception))
        self.assertIn("-1.0", str(ctx.exception))

    def test_from_dict_negative_aux_weight_raises(self):
        """from_dict with negative aux weight raises ValueError."""
        data = {
            "epoch": 1,
            "take_count": 1,
            "model_output_dir": "out",
            "chart_validity_aux_weight": -0.5,
        }
        with self.assertRaises(ValueError) as ctx:
            config.RunConfig.from_dict(data)
        self.assertIn("chart_validity_aux_weight", str(ctx.exception))

    def test_epoch_zero_raises(self):
        """epoch < 1 raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            config.RunConfig(
                epoch=0,
                take_count=1,
                model_output_dir="out",
            )
        self.assertIn("epoch", str(ctx.exception))
        self.assertIn("at least 1", str(ctx.exception))

    def test_take_count_zero_raises(self):
        """take_count 0 (and not -1) raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            config.RunConfig(
                epoch=1,
                take_count=0,
                model_output_dir="out",
            )
        self.assertIn("take_count", str(ctx.exception))

    def test_take_count_minus_one_ok(self):
        """take_count=-1 is valid (entire dataset)."""
        cfg = config.RunConfig(epoch=1, take_count=-1, model_output_dir="out")
        self.assertEqual(cfg.take_count, -1)

    def test_val_take_count_zero_raises(self):
        """val_take_count 0 (and not -1) raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            config.RunConfig(
                epoch=1,
                take_count=1,
                model_output_dir="out",
                val_take_count=0,
            )
        self.assertIn("val_take_count", str(ctx.exception))

    def test_val_take_count_minus_one_ok(self):
        """val_take_count=-1 is valid (entire dataset)."""
        cfg = config.RunConfig(
            epoch=1,
            take_count=1,
            model_output_dir="out",
            val_take_count=-1,
        )
        self.assertEqual(cfg.val_take_count, -1)

    def test_verbosity_defaults(self):
        """show_model_summary and fit_verbose default to True and 1."""
        cfg = config.RunConfig(epoch=1, take_count=1, model_output_dir="out")
        self.assertTrue(cfg.show_model_summary)
        self.assertEqual(cfg.fit_verbose, 1)

    def test_show_model_summary_explicit(self):
        """show_model_summary can be set to False."""
        cfg = config.RunConfig(
            epoch=1,
            take_count=1,
            model_output_dir="out",
            show_model_summary=False,
        )
        self.assertFalse(cfg.show_model_summary)

    def test_fit_verbose_accepts_0_1_2(self):
        """fit_verbose accepts 0, 1, and 2."""
        for v in (0, 1, 2):
            cfg = config.RunConfig(
                epoch=1,
                take_count=1,
                model_output_dir="out",
                fit_verbose=v,
            )
            self.assertEqual(cfg.fit_verbose, v)

    def test_fit_verbose_invalid_raises(self):
        """fit_verbose other than 0, 1, 2 raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            config.RunConfig(
                epoch=1,
                take_count=1,
                model_output_dir="out",
                fit_verbose=3,
            )
        self.assertIn("fit_verbose", str(ctx.exception))
        with self.assertRaises(ValueError):
            config.RunConfig(
                epoch=1,
                take_count=1,
                model_output_dir="out",
                fit_verbose=-1,
            )

    def test_from_dict_verbosity(self):
        """from_dict accepts show_model_summary and fit_verbose."""
        data = {
            "epoch": 1,
            "take_count": 1,
            "model_output_dir": "out",
            "show_model_summary": False,
            "fit_verbose": 0,
        }
        cfg = config.RunConfig.from_dict(data)
        self.assertFalse(cfg.show_model_summary)
        self.assertEqual(cfg.fit_verbose, 0)


class OnsetExperimentConfigTest(unittest.TestCase):
    def test_create_experiment_config(self):
        """Test creating complete experiment config."""
        dataset_cfg = config.OnsetDatasetConfig(
            data_dir="data/train", val_data_dir="data/val"
        )
        model_cfg = config.OnsetModelConfig()
        run_cfg = config.RunConfig(epoch=10, take_count=1, model_output_dir="out")
        exp_cfg = config.OnsetExperimentConfig(
            dataset=dataset_cfg, model=model_cfg, run=run_cfg
        )
        self.assertEqual(exp_cfg.dataset, dataset_cfg)
        self.assertEqual(exp_cfg.model, model_cfg)
        self.assertEqual(exp_cfg.run, run_cfg)

    def test_as_dict(self):
        """Test converting experiment config to dictionary."""
        dataset_cfg = config.OnsetDatasetConfig(
            data_dir="data/train", val_data_dir="data/val", batch_size=4
        )
        model_cfg = config.OnsetModelConfig(initial_filters=16)
        run_cfg = config.RunConfig(epoch=10, take_count=1, model_output_dir="out")
        exp_cfg = config.OnsetExperimentConfig(
            dataset=dataset_cfg, model=model_cfg, run=run_cfg
        )
        d = exp_cfg.as_dict()
        self.assertIn("dataset", d)
        self.assertIn("model", d)
        self.assertIn("run", d)
        self.assertEqual(d["dataset"]["batch_size"], 4)
        self.assertEqual(d["model"]["initial_filters"], 16)

    def test_from_dict(self):
        """Test creating experiment config from dictionary."""
        data = {
            "dataset": {
                "data_dir": "data/train",
                "val_data_dir": "data/val",
                "batch_size": 2,
            },
            "model": {"initial_filters": 8, "depth": 1},
            "run": {"epoch": 5, "take_count": 10, "model_output_dir": "out"},
        }
        exp_cfg = config.OnsetExperimentConfig.from_dict(data)
        self.assertEqual(exp_cfg.dataset.batch_size, 2)
        self.assertEqual(exp_cfg.model.initial_filters, 8)
        self.assertEqual(exp_cfg.run.epoch, 5)

    def test_from_dict_missing_key(self):
        """Test that missing keys raise KeyError."""
        data = {
            "dataset": {"data_dir": "data/train", "val_data_dir": "data/val"},
            # Missing "model" and "run"
        }
        with self.assertRaises(KeyError):
            config.OnsetExperimentConfig.from_dict(data)

    def test_to_json_and_from_json(self):
        """Test saving and loading config from JSON file."""
        dataset_cfg = config.OnsetDatasetConfig(
            data_dir="data/train",
            val_data_dir="data/val",
            batch_size=4,
        )
        model_cfg = config.OnsetModelConfig(initial_filters=16, depth=2)
        run_cfg = config.RunConfig(
            epoch=20, take_count=-1, model_output_dir="out", seed=42
        )
        exp_cfg = config.OnsetExperimentConfig(
            dataset=dataset_cfg, model=model_cfg, run=run_cfg
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = os.path.join(temp_dir, "test_config.json")
            exp_cfg.to_json(config_path)

            # Verify file exists
            self.assertTrue(os.path.exists(config_path))

            # Load it back
            loaded_cfg = config.OnsetExperimentConfig.from_json(config_path)
            self.assertEqual(loaded_cfg.dataset.batch_size, 4)
            self.assertEqual(loaded_cfg.model.initial_filters, 16)
            self.assertEqual(loaded_cfg.run.seed, 42)

    def test_to_json_creates_directory(self):
        """Test that to_json creates directory if it doesn't exist."""
        dataset_cfg = config.OnsetDatasetConfig(
            data_dir="data/train", val_data_dir="data/val"
        )
        model_cfg = config.OnsetModelConfig()
        run_cfg = config.RunConfig(epoch=10, take_count=1, model_output_dir="out")
        exp_cfg = config.OnsetExperimentConfig(
            dataset=dataset_cfg, model=model_cfg, run=run_cfg
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = os.path.join(temp_dir, "subdir", "config.json")
            exp_cfg.to_json(config_path)
            self.assertTrue(os.path.exists(config_path))

    def test_from_json_file_not_found(self):
        """Test that loading non-existent file raises FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            config.OnsetExperimentConfig.from_json("nonexistent.json")

    def test_from_json_invalid_json(self):
        """Test that invalid JSON raises JSONDecodeError."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = os.path.join(temp_dir, "invalid.json")
            with open(config_path, "w") as f:
                f.write("invalid json content {")
            with self.assertRaises(json.JSONDecodeError):
                config.OnsetExperimentConfig.from_json(config_path)


class ArrowExperimentConfigTest(unittest.TestCase):
    def test_create_experiment_config(self):
        """Test creating complete experiment config."""
        dataset_cfg = config.ArrowDatasetConfig(
            data_dir="data/train", val_data_dir="data/val"
        )
        model_cfg = config.ArrowModelConfig()
        run_cfg = config.RunConfig(epoch=10, take_count=-1, model_output_dir="out")
        exp_cfg = config.ArrowExperimentConfig(
            dataset=dataset_cfg, model=model_cfg, run=run_cfg
        )
        self.assertEqual(exp_cfg.dataset, dataset_cfg)
        self.assertEqual(exp_cfg.model, model_cfg)
        self.assertEqual(exp_cfg.run, run_cfg)

    def test_as_dict(self):
        """Test converting experiment config to dictionary."""
        dataset_cfg = config.ArrowDatasetConfig(
            data_dir="data/train", val_data_dir="data/val", batch_size=2
        )
        model_cfg = config.ArrowModelConfig(num_layers=2)
        run_cfg = config.RunConfig(epoch=10, take_count=-1, model_output_dir="out")
        exp_cfg = config.ArrowExperimentConfig(
            dataset=dataset_cfg, model=model_cfg, run=run_cfg
        )
        d = exp_cfg.as_dict()
        self.assertEqual(d["dataset"]["batch_size"], 2)
        self.assertEqual(d["model"]["num_layers"], 2)

    def test_to_json_and_from_json(self):
        """Test saving and loading config from JSON file."""
        dataset_cfg = config.ArrowDatasetConfig(
            data_dir="data/train", val_data_dir="data/val"
        )
        model_cfg = config.ArrowModelConfig(num_layers=3, d_model=256)
        run_cfg = config.RunConfig(
            epoch=15, take_count=-1, model_output_dir="out", seed=99
        )
        exp_cfg = config.ArrowExperimentConfig(
            dataset=dataset_cfg, model=model_cfg, run=run_cfg
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = os.path.join(temp_dir, "arrow_config.json")
            exp_cfg.to_json(config_path)

            loaded_cfg = config.ArrowExperimentConfig.from_json(config_path)
            self.assertEqual(loaded_cfg.dataset.data_dir, "data/train")
            self.assertEqual(loaded_cfg.model.num_layers, 3)
            self.assertEqual(loaded_cfg.run.seed, 99)


if __name__ == "__main__":
    unittest.main()
