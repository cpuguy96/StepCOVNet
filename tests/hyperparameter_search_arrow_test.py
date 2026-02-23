import importlib.util
import json
import os
import tempfile
import unittest
from unittest import mock

import keras

from stepcovnet import config

# Load the script as a module so we can test its functions without running main()
_SCRIPT_PATH = os.path.join(
    os.path.dirname(__file__), "..", "scripts", "hyperparameter_search_arrow.py"
)
_spec = importlib.util.spec_from_file_location(
    "hyperparameter_search_arrow",
    _SCRIPT_PATH,
    submodule_search_locations=[os.path.join(os.path.dirname(__file__), "..")],
)
assert _spec is not None and _spec.loader is not None
sweep_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sweep_module)

TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "testdata")


class SweepConfigLoadingTest(unittest.TestCase):
    """Sweep config loading: valid and invalid keys."""

    def test_load_valid_sweep_config(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(
                {
                    "base_config": "configs/arrow_baseline.json",
                    "search_space": {
                        "model.dropout_rate": [0.0, 0.1],
                        "run.chart_validity_aux_weight": [0.0],
                    },
                    "optimize": {"metric": "val_loss", "mode": "min"},
                },
                f,
            )
            path = f.name
        try:
            data = sweep_module.load_sweep_config(path)
            self.assertIsInstance(data, dict)
            self.assertEqual(data["base_config"], "configs/arrow_baseline.json")
            self.assertEqual(data["search_space"]["model.dropout_rate"], [0.0, 0.1])
            self.assertEqual(
                data["search_space"]["run.chart_validity_aux_weight"], [0.0]
            )
            self.assertEqual(data["optimize"]["metric"], "val_loss")
            self.assertEqual(data["optimize"]["mode"], "min")
        finally:
            os.unlink(path)

    def test_load_rejects_missing_base_config(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(
                {
                    "search_space": {"model.dropout_rate": [0.0]},
                    "optimize": {"metric": "val_loss", "mode": "min"},
                },
                f,
            )
            path = f.name
        try:
            with self.assertRaises(ValueError) as ctx:
                sweep_module.load_sweep_config(path)
            self.assertIn("base_config", str(ctx.exception))
        finally:
            os.unlink(path)

    def test_load_rejects_missing_optimize(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(
                {
                    "base_config": "configs/arrow_baseline.json",
                    "search_space": {"model.dropout_rate": [0.0]},
                },
                f,
            )
            path = f.name
        try:
            with self.assertRaises(ValueError) as ctx:
                sweep_module.load_sweep_config(path)
            self.assertIn("optimize", str(ctx.exception))
        finally:
            os.unlink(path)

    def test_load_rejects_invalid_optimize_mode(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(
                {
                    "base_config": "configs/arrow_baseline.json",
                    "search_space": {"model.dropout_rate": [0.0]},
                    "optimize": {"metric": "val_loss", "mode": "invalid"},
                },
                f,
            )
            path = f.name
        try:
            with self.assertRaises(ValueError) as ctx:
                sweep_module.load_sweep_config(path)
            self.assertIn("mode", str(ctx.exception))
        finally:
            os.unlink(path)

    def test_load_rejects_forbidden_key_run_val_take_count(self):
        """run.val_take_count is fixed; cannot be in search_space."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(
                {
                    "base_config": "configs/arrow_baseline.json",
                    "search_space": {"run.val_take_count": [1, 2]},
                    "optimize": {"metric": "val_loss", "mode": "min"},
                },
                f,
            )
            path = f.name
        try:
            with self.assertRaises(ValueError) as ctx:
                sweep_module.load_sweep_config(path)
            self.assertIn("forbidden", str(ctx.exception).lower())
        finally:
            os.unlink(path)


class GridExpansionTest(unittest.TestCase):
    """Grid expansion: number of combinations and structure."""

    def test_expand_grid_2x3x2(self):
        search_space = {
            "model.dropout_rate": [0.0, 0.1],
            "model.num_layers": [1, 2, 3],
            "run.chart_validity_aux_weight": [0.0, 0.3],
        }
        combinations = sweep_module.expand_grid(search_space)
        self.assertEqual(len(combinations), 2 * 3 * 2)
        for combo in combinations:
            self.assertIsInstance(combo, dict)
            self.assertIn("model.dropout_rate", combo)
            self.assertIn("model.num_layers", combo)
            self.assertIn("run.chart_validity_aux_weight", combo)
            self.assertIn(combo["model.dropout_rate"], [0.0, 0.1])
            self.assertIn(combo["model.num_layers"], [1, 2, 3])
            self.assertIn(combo["run.chart_validity_aux_weight"], [0.0, 0.3])

    def test_expand_grid_single_param(self):
        search_space = {"model.num_layers": [1, 2]}
        combinations = sweep_module.expand_grid(search_space)
        self.assertEqual(len(combinations), 2)
        self.assertEqual(combinations[0], {"model.num_layers": 1})
        self.assertEqual(combinations[1], {"model.num_layers": 2})


class ApplyOverridesAndFixedValuesTest(unittest.TestCase):
    """Apply overrides and enforce fixed val_take_count, batch_size; epoch/take_count from base or overrides."""

    def _minimal_base_config(self) -> config.ArrowExperimentConfig:
        return config.ArrowExperimentConfig(
            dataset=config.ArrowDatasetConfig(
                data_dir=TEST_DATA_DIR,
                val_data_dir=TEST_DATA_DIR,
                batch_size=4,
                snippet_half_frames=0,
            ),
            model=config.ArrowModelConfig(
                num_layers=1,
                d_model=128,
                dropout_rate=0.5,
            ),
            run=config.RunConfig(
                epoch=10,
                take_count=100,
                model_output_dir="/tmp/models",
                callback_root_dir="",
                val_take_count=5,
            ),
        )

    def test_apply_overrides_updates_model_and_run(self):
        base = self._minimal_base_config()
        overrides = {
            "model.dropout_rate": 0.2,
            "model.num_layers": 2,
            "run.chart_validity_aux_weight": 0.3,
        }
        out = sweep_module.apply_overrides(base, overrides)
        self.assertEqual(out.model.dropout_rate, 0.2)
        self.assertEqual(out.model.num_layers, 2)
        self.assertEqual(out.run.chart_validity_aux_weight, 0.3)

    def test_apply_overrides_forces_val_take_count_minus_one_batch_size_one(self):
        """Epoch comes from base; val_take_count and batch_size are forced."""
        base = self._minimal_base_config()
        self.assertEqual(base.run.epoch, 10)
        self.assertEqual(base.run.val_take_count, 5)
        self.assertEqual(base.dataset.batch_size, 4)
        overrides = {}
        out = sweep_module.apply_overrides(base, overrides)
        self.assertEqual(out.run.epoch, 10)
        self.assertEqual(out.run.val_take_count, -1)
        self.assertEqual(out.dataset.batch_size, 1)

    def test_apply_overrides_epoch_and_take_count_from_base_when_not_overridden(self):
        """epoch and take_count come from base when not in overrides."""
        base = self._minimal_base_config()
        base.run.take_count = 25
        out = sweep_module.apply_overrides(base, {})
        self.assertEqual(out.run.take_count, 25)
        self.assertEqual(out.run.epoch, 10)
        self.assertEqual(out.run.val_take_count, -1)

    def test_apply_overrides_epoch_and_take_count_from_overrides(self):
        """run.epoch and run.take_count in overrides are applied."""
        base = self._minimal_base_config()
        base.run.take_count = 10
        out = sweep_module.apply_overrides(base, {"run.epoch": 50, "run.take_count": 5})
        self.assertEqual(out.run.epoch, 50)
        self.assertEqual(out.run.take_count, 5)


class ForbiddenKeysValidationTest(unittest.TestCase):
    """run.val_take_count and other fixed keys cannot be in search_space."""

    def test_load_sweep_config_rejects_run_val_take_count_in_search_space(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(
                {
                    "base_config": "configs/arrow_baseline.json",
                    "search_space": {"run.val_take_count": [1, 2]},
                    "optimize": {"metric": "val_loss", "mode": "min"},
                },
                f,
            )
            path = f.name
        try:
            with self.assertRaises(ValueError) as ctx:
                sweep_module.load_sweep_config(path)
            self.assertIn("forbidden", str(ctx.exception).lower())
        finally:
            os.unlink(path)


class BestRunSelectionTest(unittest.TestCase):
    """Best run selection by optimize.metric and optimize.mode."""

    def test_select_best_run_min_val_loss(self):
        results = [
            {"best_val_loss": 0.5, "best_val_acc": 0.8},
            {"best_val_loss": 0.3, "best_val_acc": 0.9},
            {"best_val_loss": 0.4, "best_val_acc": 0.85},
        ]
        idx = sweep_module.select_best_run(results, "val_loss", "min")
        self.assertEqual(idx, 1)

    def test_select_best_run_max_val_acc(self):
        results = [
            {"best_val_loss": 0.5, "best_val_acc": 0.8},
            {"best_val_loss": 0.3, "best_val_acc": 0.9},
            {"best_val_loss": 0.4, "best_val_acc": 0.95},
        ]
        idx = sweep_module.select_best_run(results, "val_acc", "max")
        self.assertEqual(idx, 2)

    def test_select_best_run_single_result(self):
        results = [{"best_val_loss": 0.5}]
        idx = sweep_module.select_best_run(results, "val_loss", "min")
        self.assertEqual(idx, 0)

    def test_select_best_run_empty_raises(self):
        with self.assertRaises(ValueError) as ctx:
            sweep_module.select_best_run([], "val_loss", "min")
        self.assertIn("empty", str(ctx.exception))


class ExtractMetricsTest(unittest.TestCase):
    """Extract best/final metrics from history."""

    def test_extract_metrics(self):
        history = mock.Mock()
        history.history = {
            "val_loss": [0.8, 0.5, 0.4],
            "val_acc": [0.6, 0.7, 0.85],
        }
        metrics = sweep_module.extract_metrics(history)
        self.assertEqual(metrics["final_val_loss"], 0.4)
        self.assertEqual(metrics["best_val_loss"], 0.4)
        self.assertEqual(metrics["best_epoch_val_loss"], 3)
        self.assertEqual(metrics["final_val_acc"], 0.85)
        self.assertEqual(metrics["best_val_acc"], 0.85)
        self.assertEqual(metrics["best_epoch_val_acc"], 3)


class EndToEndMinimalTest(unittest.TestCase):
    """One combination, real test data dir; results and best_config written."""

    def test_one_run_produces_results_and_best_config(self):
        base_config_path = os.path.join(
            os.path.dirname(__file__), "..", "configs", "arrow_baseline.json"
        )
        if not os.path.isfile(base_config_path):
            self.skipTest("configs/arrow_baseline.json not found")
        with tempfile.TemporaryDirectory() as temp_dir:
            sweep_path = os.path.join(temp_dir, "sweep.json")
            with open(sweep_path, "w") as f:
                json.dump(
                    {
                        "base_config": base_config_path,
                        "search_space": {
                            "model.dropout_rate": [0.0],
                        },
                        "optimize": {"metric": "val_loss", "mode": "min"},
                        "sweep_output_dir": os.path.join(temp_dir, "sweep_out"),
                    },
                    f,
                )
            # Override base config to use test data dir and take_count=2 (fixed across runs)
            base = config.ArrowExperimentConfig.from_json(base_config_path)
            base.dataset.data_dir = TEST_DATA_DIR
            base.dataset.val_data_dir = TEST_DATA_DIR
            base.run.take_count = 2
            base_config_override = os.path.join(temp_dir, "base_arrow.json")
            base.to_json(base_config_override)
            with open(sweep_path, "r") as f:
                sweep_data = json.load(f)
            sweep_data["base_config"] = base_config_override
            with open(sweep_path, "w") as f:
                json.dump(sweep_data, f)

            with mock.patch(
                "sys.argv",
                [
                    "hyperparameter_search_arrow",
                    "--sweep_config",
                    sweep_path,
                ],
            ):
                exit_code = sweep_module.main()
            self.assertEqual(exit_code, 0)

            sweep_out = os.path.join(temp_dir, "sweep_out")
            results_path = os.path.join(sweep_out, "results.json")
            best_config_path = os.path.join(sweep_out, "best_config.json")
            self.assertTrue(os.path.isfile(results_path), f"Missing {results_path}")
            self.assertTrue(
                os.path.isfile(best_config_path), f"Missing {best_config_path}"
            )
            with open(results_path) as f:
                results = json.load(f)
            self.assertEqual(len(results), 1)
            self.assertIn("overrides", results[0])
            self.assertIn("best_val_loss", results[0])


class SweepVerbosityTest(unittest.TestCase):
    """Sweep sets show_model_summary=False and fit_verbose=0 for each run."""

    def test_sweep_passes_quiet_verbosity_to_trainer(self):
        """run_arrow_train_from_config is called with show_model_summary=False, fit_verbose=0."""
        captured_run_configs = []

        def capture_run_config(dataset_config, model_config, run_config):
            captured_run_configs.append(run_config)
            history = mock.Mock()
            history.history = {"val_loss": [1.0], "val_acc": [0.5]}
            return mock.Mock(), history

        base_config_path = os.path.join(
            os.path.dirname(__file__), "..", "configs", "arrow_baseline.json"
        )
        if not os.path.isfile(base_config_path):
            self.skipTest("configs/arrow_baseline.json not found")
        with tempfile.TemporaryDirectory() as temp_dir:
            sweep_path = os.path.join(temp_dir, "sweep.json")
            with open(sweep_path, "w") as f:
                json.dump(
                    {
                        "base_config": base_config_path,
                        "search_space": {"model.dropout_rate": [0.0]},
                        "optimize": {"metric": "val_loss", "mode": "min"},
                        "sweep_output_dir": os.path.join(temp_dir, "sweep_out"),
                    },
                    f,
                )
            base = config.ArrowExperimentConfig.from_json(base_config_path)
            base.dataset.data_dir = TEST_DATA_DIR
            base.dataset.val_data_dir = TEST_DATA_DIR
            base.run.take_count = 2
            base_config_override = os.path.join(temp_dir, "base_arrow.json")
            base.to_json(base_config_override)
            with open(sweep_path, "r") as f:
                sweep_data = json.load(f)
            sweep_data["base_config"] = base_config_override
            with open(sweep_path, "w") as f:
                json.dump(sweep_data, f)

            with mock.patch(
                "stepcovnet.trainers.run_arrow_train_from_config",
                side_effect=capture_run_config,
            ):
                with mock.patch(
                    "sys.argv",
                    ["hyperparameter_search_arrow", "--sweep_config", sweep_path],
                ):
                    sweep_module.main()

        self.assertGreater(len(captured_run_configs), 0)
        for run_config in captured_run_configs:
            self.assertFalse(
                run_config.show_model_summary,
                "sweep should set show_model_summary=False",
            )
            self.assertEqual(
                run_config.fit_verbose,
                0,
                "sweep should set fit_verbose=0",
            )


if __name__ == "__main__":
    unittest.main()
