import json
import os
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import pytest

from stepcovnet import config, trainers

# Allow importing the script module (scripts/hyperparameter_search_arrow.py)
_SCRIPT_DIR = os.path.join(os.path.dirname(__file__), "..", "scripts")
_SCRIPT_DIR = os.path.abspath(_SCRIPT_DIR)
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

import hyperparameter_search_arrow  # noqa: E402

_SCRIPT_PATH = os.path.join(_SCRIPT_DIR, "hyperparameter_search_arrow.py")
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)

TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "testdata")


def _run_sweep_script(*args):
    """Run the sweep script as a subprocess. Returns the completed process."""
    return subprocess.run(
        [sys.executable, _SCRIPT_PATH] + list(args),
        cwd=_PROJECT_ROOT,
        capture_output=True,
        text=True,
    )


class SweepConfigLoadingTest(unittest.TestCase):
    """Sweep config loading: valid and invalid keys."""

    def test_load_valid_sweep_config(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(
                {
                    "base_config": "configs/arrow_baseline.json",
                    "search_space": {
                        "model.transformer.dropout_rate": [0.0, 0.1],
                        "run.chart_validity_aux_weight": [0.0],
                    },
                    "optimize": {"metric": "val_loss", "mode": "min"},
                },
                f,
            )
            path = f.name
        try:
            data = hyperparameter_search_arrow.load_sweep_config(path)
            self.assertIsInstance(data, dict)
            self.assertEqual(data["base_config"], "configs/arrow_baseline.json")
            self.assertEqual(
                data["search_space"]["model.transformer.dropout_rate"], [0.0, 0.1]
            )
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
                    "search_space": {"model.transformer.dropout_rate": [0.0]},
                    "optimize": {"metric": "val_loss", "mode": "min"},
                },
                f,
            )
            path = f.name
        try:
            with self.assertRaises(ValueError) as ctx:
                hyperparameter_search_arrow.load_sweep_config(path)
            self.assertIn("base_config", str(ctx.exception))
        finally:
            os.unlink(path)

    def test_load_rejects_missing_optimize(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(
                {
                    "base_config": "configs/arrow_baseline.json",
                    "search_space": {"model.transformer.dropout_rate": [0.0]},
                },
                f,
            )
            path = f.name
        try:
            with self.assertRaises(ValueError) as ctx:
                hyperparameter_search_arrow.load_sweep_config(path)
            self.assertIn("optimize", str(ctx.exception))
        finally:
            os.unlink(path)

    def test_load_rejects_invalid_optimize_mode(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(
                {
                    "base_config": "configs/arrow_baseline.json",
                    "search_space": {"model.transformer.dropout_rate": [0.0]},
                    "optimize": {"metric": "val_loss", "mode": "invalid"},
                },
                f,
            )
            path = f.name
        try:
            with self.assertRaises(ValueError) as ctx:
                hyperparameter_search_arrow.load_sweep_config(path)
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
                hyperparameter_search_arrow.load_sweep_config(path)
            self.assertIn("forbidden", str(ctx.exception).lower())
        finally:
            os.unlink(path)

    def test_load_accepts_search_grid_and_random(self):
        """Sweep config may contain 'search': 'grid' or 'random'."""
        for search_val in ("grid", "random"):
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False
            ) as f:
                json.dump(
                    {
                        "base_config": "configs/arrow_baseline.json",
                        "search_space": {"model.transformer.dropout_rate": [0.0]},
                        "optimize": {"metric": "val_loss", "mode": "min"},
                        "search": search_val,
                    },
                    f,
                )
                path = f.name
            try:
                data = hyperparameter_search_arrow.load_sweep_config(path)
                self.assertEqual(data["search"], search_val)
            finally:
                os.unlink(path)

    def test_load_rejects_invalid_search(self):
        """Sweep config 'search' must be 'grid' or 'random'."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(
                {
                    "base_config": "configs/arrow_baseline.json",
                    "search_space": {"model.transformer.dropout_rate": [0.0]},
                    "optimize": {"metric": "val_loss", "mode": "min"},
                    "search": "monte_carlo",
                },
                f,
            )
            path = f.name
        try:
            with self.assertRaises(ValueError) as ctx:
                hyperparameter_search_arrow.load_sweep_config(path)
            self.assertIn("search", str(ctx.exception))
        finally:
            os.unlink(path)


class GridExpansionTest(unittest.TestCase):
    """Grid expansion: number of combinations and structure."""

    def test_expand_grid_2x3x2(self):
        search_space = {
            "model.transformer.dropout_rate": [0.0, 0.1],
            "model.transformer.num_layers": [1, 2, 3],
            "run.chart_validity_aux_weight": [0.0, 0.3],
        }
        combinations = hyperparameter_search_arrow.expand_grid(search_space)
        self.assertEqual(len(combinations), 2 * 3 * 2)
        for combo in combinations:
            self.assertIsInstance(combo, dict)
            self.assertIn("model.transformer.dropout_rate", combo)
            self.assertIn("model.transformer.num_layers", combo)
            self.assertIn("run.chart_validity_aux_weight", combo)
            self.assertIn(combo["model.transformer.dropout_rate"], [0.0, 0.1])
            self.assertIn(combo["model.transformer.num_layers"], [1, 2, 3])
            self.assertIn(combo["run.chart_validity_aux_weight"], [0.0, 0.3])

    def test_expand_grid_single_param(self):
        search_space = {"model.transformer.num_layers": [1, 2]}
        combinations = hyperparameter_search_arrow.expand_grid(search_space)
        self.assertEqual(len(combinations), 2)
        self.assertEqual(combinations[0], {"model.transformer.num_layers": 1})
        self.assertEqual(combinations[1], {"model.transformer.num_layers": 2})


class ApplyOverridesAndFixedValuesTest(unittest.TestCase):
    """Apply overrides and enforce fixed val_take_count, batch_size; epoch/take_count from base or overrides."""

    def setUp(self):
        self._tmp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self._tmp_dir.cleanup()

    def _minimal_base_config(self) -> config.ArrowExperimentConfig:
        tmp_dir = self._tmp_dir.name
        return config.ArrowExperimentConfig(
            dataset=config.ArrowDatasetConfig(
                data_dir=TEST_DATA_DIR,
                val_data_dir=TEST_DATA_DIR,
                batch_size=4,
                snippet_half_frames=0,
            ),
            model=config.ArrowModelConfig.from_dict(
                {
                    "transformer": {
                        "num_layers": 1,
                        "d_model": 128,
                        "dropout_rate": 0.5,
                    },
                }
            ),
            run=config.ArrowRunConfig(
                epoch=10,
                take_count=100,
                model_output_dir=tmp_dir,
                callback_root_dir=tmp_dir,
                val_take_count=5,
            ),
        )

    def test_apply_overrides_updates_model_and_run(self):
        base = self._minimal_base_config()
        overrides = {
            "model.transformer.dropout_rate": 0.2,
            "model.transformer.num_layers": 2,
            "run.chart_validity_aux_weight": 0.3,
        }
        out = hyperparameter_search_arrow.apply_overrides(base, overrides)
        assert out.model.transformer is not None
        self.assertEqual(out.model.transformer.dropout_rate, 0.2)
        self.assertEqual(out.model.transformer.num_layers, 2)
        self.assertEqual(out.run.chart_validity_aux_weight, 0.3)

    def test_apply_overrides_nested_model_keys(self):
        """Multi-level keys (model.transformer.num_layers) apply to nested config."""
        base = self._minimal_base_config()
        overrides = {"model.transformer.num_layers": 3}
        out = hyperparameter_search_arrow.apply_overrides(base, overrides)
        assert out.model.transformer is not None
        self.assertEqual(out.model.transformer.num_layers, 3)

    def test_apply_overrides_creates_missing_intermediate_dict(self):
        """Override targeting a nested key on a non-existent intermediate dict creates it (_set_nested branch)."""
        base = self._minimal_base_config()
        # Base has only transformer. Set model_type to mlp and model.mlp.hidden_dims so mlp block is created and used.
        overrides = {"model.model_type": "mlp", "model.mlp.hidden_dims": [128, 64]}
        out = hyperparameter_search_arrow.apply_overrides(base, overrides)
        self.assertIsNotNone(out.model.mlp)
        assert out.model.mlp is not None
        self.assertEqual(out.model.mlp.hidden_dims, [128, 64])

    def test_set_nested_raises_when_intermediate_is_leaf(self):
        """_set_nested raises ValueError when path goes through a leaf (non-dict) value."""
        d = {"model": {"transformer": 42}}
        with self.assertRaises(ValueError) as ctx:
            hyperparameter_search_arrow._set_nested(
                d["model"], "transformer.num_layers", 2
            )
        self.assertIn("transformer.num_layers", str(ctx.exception))
        self.assertIn("transformer", str(ctx.exception))
        self.assertIn("not a nested object", str(ctx.exception))

    def test_set_nested_raises_when_intermediate_is_none(self):
        """_set_nested raises (does not overwrite) when intermediate segment exists with value None."""
        d = {"model": {"transformer": None}}
        with self.assertRaises(ValueError) as ctx:
            hyperparameter_search_arrow._set_nested(
                d["model"], "transformer.num_layers", 2
            )
        self.assertIn("not a nested object", str(ctx.exception))
        self.assertEqual(d["model"]["transformer"], None)

    def test_apply_overrides_raises_for_invalid_model_override_key(self):
        """apply_overrides raises ValueError for model.<single-key> when key is not model_type."""
        base = self._minimal_base_config()
        overrides = {"model.dropout_rate": 0.5}
        with self.assertRaises(ValueError) as ctx:
            hyperparameter_search_arrow.apply_overrides(base, overrides)
        self.assertIn("model.dropout_rate", str(ctx.exception))
        self.assertIn("model.model_type or model.<block>.<param>", str(ctx.exception))

    def test_apply_overrides_model_type_can_be_set(self):
        """Sweep can set model.model_type (e.g. transformer vs mlp) in overrides."""
        base = self._minimal_base_config()
        overrides = {"model.model_type": "mlp"}
        out = hyperparameter_search_arrow.apply_overrides(base, overrides)
        self.assertEqual(out.model.model_type, "mlp")
        self.assertIsNotNone(out.model.mlp)

    def test_apply_overrides_model_lstm_keys(self):
        """Overrides like model.model_type=lstm and model.lstm.units apply correctly."""
        base = self._minimal_base_config()
        overrides = {
            "model.model_type": "lstm",
            "model.lstm.units": 64,
            "model.lstm.num_layers": 2,
            "model.lstm.dropout_rate": 0.1,
        }
        out = hyperparameter_search_arrow.apply_overrides(base, overrides)
        self.assertEqual(out.model.model_type, "lstm")
        self.assertIsNotNone(out.model.lstm)
        assert out.model.lstm is not None
        self.assertEqual(out.model.lstm.units, 64)
        self.assertEqual(out.model.lstm.num_layers, 2)
        self.assertEqual(out.model.lstm.dropout_rate, 0.1)

    def test_apply_overrides_model_gru_keys(self):
        """Overrides like model.model_type=gru and model.gru.units apply correctly."""
        base = self._minimal_base_config()
        overrides = {
            "model.model_type": "gru",
            "model.gru.units": 64,
            "model.gru.num_layers": 2,
            "model.gru.dropout_rate": 0.1,
        }
        out = hyperparameter_search_arrow.apply_overrides(base, overrides)
        self.assertEqual(out.model.model_type, "gru")
        self.assertIsNotNone(out.model.gru)
        assert out.model.gru is not None
        self.assertEqual(out.model.gru.units, 64)
        self.assertEqual(out.model.gru.num_layers, 2)
        self.assertEqual(out.model.gru.dropout_rate, 0.1)

    def test_apply_overrides_forces_val_take_count_minus_one_batch_size_one(self):
        """Epoch comes from base; val_take_count and batch_size are forced."""
        base = self._minimal_base_config()
        self.assertEqual(base.run.epoch, 10)
        self.assertEqual(base.run.val_take_count, 5)
        self.assertEqual(base.dataset.batch_size, 4)
        overrides = {}
        out = hyperparameter_search_arrow.apply_overrides(base, overrides)
        self.assertEqual(out.run.epoch, 10)
        self.assertEqual(out.run.val_take_count, -1)
        self.assertEqual(out.dataset.batch_size, 1)

    def test_apply_overrides_epoch_and_take_count_from_base_when_not_overridden(self):
        """epoch and take_count come from base when not in overrides."""
        base = self._minimal_base_config()
        base.run.take_count = 25
        out = hyperparameter_search_arrow.apply_overrides(base, {})
        self.assertEqual(out.run.take_count, 25)
        self.assertEqual(out.run.epoch, 10)
        self.assertEqual(out.run.val_take_count, -1)

    def test_apply_overrides_epoch_and_take_count_from_overrides(self):
        """run.epoch and run.take_count in overrides are applied."""
        base = self._minimal_base_config()
        base.run.take_count = 10
        out = hyperparameter_search_arrow.apply_overrides(
            base, {"run.epoch": 50, "run.take_count": 5}
        )
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
                hyperparameter_search_arrow.load_sweep_config(path)
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
        idx = hyperparameter_search_arrow.select_best_run(results, "val_loss", "min")
        self.assertEqual(idx, 1)

    def test_select_best_run_max_val_acc(self):
        results = [
            {"best_val_loss": 0.5, "best_val_acc": 0.8},
            {"best_val_loss": 0.3, "best_val_acc": 0.9},
            {"best_val_loss": 0.4, "best_val_acc": 0.95},
        ]
        idx = hyperparameter_search_arrow.select_best_run(results, "val_acc", "max")
        self.assertEqual(idx, 2)

    def test_select_best_run_single_result(self):
        results = [{"best_val_loss": 0.5}]
        idx = hyperparameter_search_arrow.select_best_run(results, "val_loss", "min")
        self.assertEqual(idx, 0)

    def test_select_best_run_empty_raises(self):
        with self.assertRaises(ValueError) as ctx:
            hyperparameter_search_arrow.select_best_run([], "val_loss", "min")
        self.assertIn("empty", str(ctx.exception))


class IsBetterThanTest(unittest.TestCase):
    """_is_better_than for best-so-far comparison (min/max, None)."""

    def test_min_mode_lower_is_better(self):
        self.assertTrue(hyperparameter_search_arrow._is_better_than(0.3, None, "min"))
        self.assertTrue(hyperparameter_search_arrow._is_better_than(0.3, 0.5, "min"))
        self.assertFalse(hyperparameter_search_arrow._is_better_than(0.5, 0.3, "min"))
        self.assertFalse(hyperparameter_search_arrow._is_better_than(0.3, 0.3, "min"))

    def test_max_mode_higher_is_better(self):
        self.assertTrue(hyperparameter_search_arrow._is_better_than(0.9, None, "max"))
        self.assertTrue(hyperparameter_search_arrow._is_better_than(0.9, 0.5, "max"))
        self.assertFalse(hyperparameter_search_arrow._is_better_than(0.3, 0.5, "max"))
        self.assertFalse(hyperparameter_search_arrow._is_better_than(0.5, 0.5, "max"))


class ExtractMetricsTest(unittest.TestCase):
    """Extract best/final metrics from history."""

    def test_extract_metrics(self):
        history = mock.Mock()
        history.history = {
            "val_loss": [0.8, 0.5, 0.4],
            "val_acc": [0.6, 0.7, 0.85],
        }
        metrics = hyperparameter_search_arrow.extract_metrics(history)
        self.assertEqual(metrics["final_val_loss"], 0.4)
        self.assertEqual(metrics["best_val_loss"], 0.4)
        self.assertEqual(metrics["best_epoch_val_loss"], 3)
        self.assertEqual(metrics["final_val_acc"], 0.85)
        self.assertEqual(metrics["best_val_acc"], 0.85)
        self.assertEqual(metrics["best_epoch_val_acc"], 3)

    def test_extract_metrics_treats_loss_suffix_as_minimize(self):
        """val_main_loss and other *_loss metrics use min for best, not max."""
        history = mock.Mock()
        history.history = {
            "val_main_loss": [0.9, 0.5, 0.7],
            "val_chart_validity_aux_loss": [0.2, 0.1, 0.15],
            "val_acc": [0.6, 0.8, 0.7],
        }
        metrics = hyperparameter_search_arrow.extract_metrics(history)
        self.assertEqual(metrics["best_val_main_loss"], 0.5)
        self.assertEqual(metrics["best_epoch_val_main_loss"], 2)
        self.assertEqual(metrics["best_val_chart_validity_aux_loss"], 0.1)
        self.assertEqual(metrics["best_val_acc"], 0.8)


@pytest.mark.slow
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
                            "model.transformer.dropout_rate": [0.0],
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
            with open(sweep_path) as f:
                sweep_data = json.load(f)
            sweep_data["base_config"] = base_config_override
            with open(sweep_path, "w") as f:
                json.dump(sweep_data, f)

            result = _run_sweep_script("--sweep_config", os.path.abspath(sweep_path))
            self.assertEqual(result.returncode, 0, result.stderr or result.stdout)

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


@pytest.mark.slow
class RandomSearchTest(unittest.TestCase):
    """Random search samples a subset of combinations; seed gives reproducibility."""

    def test_random_search_samples_subset_and_saves_seed(self):
        """With --search=random --max_runs=2 --seed=42, exactly 2 runs are executed and sweep_config records seed."""
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
                            "model.transformer.dropout_rate": [0.0, 0.1],
                            "run.chart_validity_aux_weight": [0.0, 0.3],
                        },
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
            with open(sweep_path) as f:
                sweep_data = json.load(f)
            sweep_data["base_config"] = base_config_override
            with open(sweep_path, "w") as f:
                json.dump(sweep_data, f)

            result = _run_sweep_script(
                "--sweep_config",
                os.path.abspath(sweep_path),
                "--search",
                "random",
                "--max_runs",
                "2",
                "--seed",
                "42",
            )
            self.assertEqual(result.returncode, 0, result.stderr or result.stdout)
            results_path = os.path.join(temp_dir, "sweep_out", "results.json")
            self.assertTrue(os.path.isfile(results_path))
            with open(results_path) as f:
                results = json.load(f)
            self.assertEqual(
                len(results), 2, "random search should run exactly 2 trials"
            )
            full_combinations = hyperparameter_search_arrow.expand_grid(
                {
                    "model.transformer.dropout_rate": [0.0, 0.1],
                    "run.chart_validity_aux_weight": [0.0, 0.3],
                }
            )
            for r in results:
                self.assertIn(r["overrides"], full_combinations)
            sweep_config_path = os.path.join(temp_dir, "sweep_out", "sweep_config.json")
            with open(sweep_config_path) as f:
                saved = json.load(f)
            self.assertEqual(saved.get("_effective_search"), "random")
            self.assertEqual(saved.get("_effective_seed"), 42)

    def test_search_from_config_when_cli_omitted(self):
        """Sweep config 'search': 'random' is used when --search is not passed."""
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
                            "model.transformer.dropout_rate": [0.0, 0.1],
                            "run.chart_validity_aux_weight": [0.0, 0.3],
                        },
                        "optimize": {"metric": "val_loss", "mode": "min"},
                        "sweep_output_dir": os.path.join(temp_dir, "sweep_out"),
                        "search": "random",
                        "max_runs": 2,
                        "seed": 42,
                    },
                    f,
                )
            base = config.ArrowExperimentConfig.from_json(base_config_path)
            base.dataset.data_dir = TEST_DATA_DIR
            base.dataset.val_data_dir = TEST_DATA_DIR
            base.run.take_count = 2
            base_config_override = os.path.join(temp_dir, "base_arrow.json")
            base.to_json(base_config_override)
            with open(sweep_path) as f:
                sweep_data = json.load(f)
            sweep_data["base_config"] = base_config_override
            with open(sweep_path, "w") as f:
                json.dump(sweep_data, f)

            # Do not pass --search; config "search": "random" must be used
            result = _run_sweep_script("--sweep_config", os.path.abspath(sweep_path))
            self.assertEqual(result.returncode, 0, result.stderr or result.stdout)
            sweep_config_path = os.path.join(temp_dir, "sweep_out", "sweep_config.json")
            with open(sweep_config_path) as f:
                saved = json.load(f)
            self.assertEqual(
                saved.get("_effective_search"),
                "random",
                "config 'search' should be honored when --search not passed",
            )
            with open(os.path.join(temp_dir, "sweep_out", "results.json")) as f:
                results = json.load(f)
            self.assertEqual(len(results), 2)

    def test_random_search_same_seed_same_order(self):
        """Same seed produces the same sampled overrides (reproducibility)."""
        search_space = {
            "model.transformer.dropout_rate": [0.0, 0.1, 0.2],
            "run.epoch": [1, 2],
        }
        full = hyperparameter_search_arrow.expand_grid(search_space)
        self.assertEqual(len(full), 6)
        import random

        random.seed(99)
        first = random.sample(full, 3)
        random.seed(99)
        second = random.sample(full, 3)
        self.assertEqual(first, second)


@pytest.mark.slow
class ResumeSweepTest(unittest.TestCase):
    """--resume_from loads partial results and runs only missing runs."""

    def test_resume_from_and_sweep_config_mutually_exclusive(self):
        """Passing both --resume_from and --sweep_config exits with error."""
        with tempfile.TemporaryDirectory() as temp_dir:
            resume_dir = os.path.join(temp_dir, "resume_dir")
            os.makedirs(resume_dir, exist_ok=True)
            sweep_path = os.path.join(temp_dir, "sweep.json")
            with open(sweep_path, "w") as f:
                json.dump(
                    {
                        "base_config": "configs/arrow_baseline.json",
                        "search_space": {"model.transformer.dropout_rate": [0.0]},
                        "optimize": {"metric": "val_loss", "mode": "min"},
                        "sweep_output_dir": temp_dir,
                    },
                    f,
                )
            result = _run_sweep_script(
                "--resume_from",
                os.path.abspath(resume_dir),
                "--sweep_config",
                os.path.abspath(sweep_path),
            )
            self.assertNotEqual(result.returncode, 0)
            out = (result.stdout or "") + (result.stderr or "")
            self.assertIn("resume_from", out)
            self.assertIn("sweep_config", out)

    def test_resume_from_partial_sweep_dir_runs_remaining(self):
        """With a sweep dir containing sweep_config.json and partial results.json, --resume_from runs only missing runs."""
        base_config_path = os.path.join(
            os.path.dirname(__file__), "..", "configs", "arrow_baseline.json"
        )
        if not os.path.isfile(base_config_path):
            self.skipTest("configs/arrow_baseline.json not found")
        with tempfile.TemporaryDirectory() as temp_dir:
            base = config.ArrowExperimentConfig.from_json(base_config_path)
            base.dataset.data_dir = TEST_DATA_DIR
            base.dataset.val_data_dir = TEST_DATA_DIR
            base.run.take_count = 2
            base_config_override = os.path.join(temp_dir, "base_arrow.json")
            base.to_json(base_config_override)

            resume_dir = os.path.join(temp_dir, "sweep_resume")
            os.makedirs(resume_dir, exist_ok=True)
            os.makedirs(os.path.join(resume_dir, "models"), exist_ok=True)
            os.makedirs(os.path.join(resume_dir, "callbacks"), exist_ok=True)

            search_space = {"model.transformer.dropout_rate": [0.0, 0.1]}
            combinations = hyperparameter_search_arrow.expand_grid(search_space)
            sweep_config = {
                "base_config": os.path.abspath(base_config_override),
                "search_space": search_space,
                "optimize": {"metric": "val_loss", "mode": "min"},
                "_effective_search": "grid",
                "max_runs": None,
            }
            with open(os.path.join(resume_dir, "sweep_config.json"), "w") as f:
                json.dump(sweep_config, f, indent=2)

            # One run done (index 0), one pending (index 1)
            partial_results = [
                {
                    "run_index": 0,
                    "overrides": combinations[0],
                    "best_val_loss": 0.5,
                    "final_val_loss": 0.5,
                },
                None,
            ]
            with open(os.path.join(resume_dir, "results.json"), "w") as f:
                json.dump(partial_results, f, indent=2)

            result = _run_sweep_script("--resume_from", os.path.abspath(resume_dir))
            self.assertEqual(
                result.returncode, 0, (result.stdout or "") + (result.stderr or "")
            )
            with open(os.path.join(resume_dir, "results.json")) as f:
                results = json.load(f)
            self.assertEqual(len(results), 2)
            self.assertIsNotNone(results[0])
            self.assertIsNotNone(results[1])
            self.assertIn("best_val_loss", results[1])
            self.assertTrue(
                os.path.isfile(os.path.join(resume_dir, "best_config.json"))
            )


class WorkersOptionTest(unittest.TestCase):
    """--workers must be >= 1; parallel path uses ProcessPoolExecutor."""

    def test_workers_zero_exits_with_error(self):
        """--workers=0 causes main to call parser.error and exit."""
        with (
            mock.patch(
                "sys.argv",
                [
                    "hyperparameter_search_arrow",
                    "--sweep_config",
                    "x.json",
                    "--workers",
                    "0",
                ],
            ),
            mock.patch.object(
                hyperparameter_search_arrow.PARSER, "error", side_effect=SystemExit(2)
            ),
        ):
            with self.assertRaises(SystemExit):
                hyperparameter_search_arrow.main()

    def test_workers_two_uses_parallel_path(self):
        """With --workers=2 and 2 grid points, executor receives 2 submit() calls and results are collected."""
        from concurrent.futures import Future

        submitted_futures = []

        class FakeExecutor:
            def __init__(self, max_workers=None, max_tasks_per_child=None, **kwargs):
                self.max_workers = max_workers
                self.max_tasks_per_child = max_tasks_per_child

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def submit(self, fn, *args):
                run_index, overrides = args[0], args[1]
                fut = Future()
                fut.set_result(
                    (run_index, {"best_val_loss": 0.5 - run_index * 0.1}, overrides)
                )
                submitted_futures.append(fut)
                return fut

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
                        "search_space": {"model.transformer.dropout_rate": [0.0, 0.1]},
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
            with open(sweep_path) as f:
                sweep_data = json.load(f)
            sweep_data["base_config"] = base_config_override
            with open(sweep_path, "w") as f:
                json.dump(sweep_data, f)

            with (
                mock.patch.object(
                    hyperparameter_search_arrow.futures,
                    "ProcessPoolExecutor",
                    return_value=FakeExecutor(),
                ),
                mock.patch(
                    "sys.argv",
                    [
                        "hyperparameter_search_arrow",
                        "--sweep_config",
                        sweep_path,
                        "--workers",
                        "2",
                    ],
                ),
            ):
                exit_code = hyperparameter_search_arrow.main()
            self.assertEqual(exit_code, 0)
            self.assertEqual(
                len(submitted_futures), 2, "should submit 2 runs (2 grid points)"
            )
            results_path = os.path.join(temp_dir, "sweep_out", "results.json")
            self.assertTrue(os.path.isfile(results_path))
            with open(results_path) as f:
                results = json.load(f)
            self.assertEqual(len(results), 2)
            self.assertEqual(results[0]["run_index"], 0)
            self.assertEqual(results[1]["run_index"], 1)

    def test_new_best_printed_when_run_has_best_val_metric_so_far(self):
        """When a run has the best optimize metric seen so far, 'NEW BEST' is printed."""
        from concurrent.futures import Future

        submitted_futures = []
        # Run 0: 0.5, run 1: 0.3 (best), run 2: 0.4
        metrics_by_run = {0: 0.5, 1: 0.3, 2: 0.4}

        class FakeExecutor:
            def __init__(self, max_workers=None, max_tasks_per_child=None, **kwargs):
                self.max_workers = max_workers
                self.max_tasks_per_child = max_tasks_per_child

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def submit(self, fn, *args):
                run_index, overrides = args[0], args[1]
                fut = Future()
                fut.set_result(
                    (
                        run_index,
                        {"best_val_main_loss": metrics_by_run[run_index]},
                        overrides,
                    )
                )
                submitted_futures.append(fut)
                return fut

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
                            "model.transformer.dropout_rate": [0.0, 0.1, 0.2],
                        },
                        "optimize": {"metric": "val_main_loss", "mode": "min"},
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
            with open(sweep_path) as f:
                sweep_data = json.load(f)
            sweep_data["base_config"] = base_config_override
            with open(sweep_path, "w") as f:
                json.dump(sweep_data, f)

            # Complete in order: run 1 first (best 0.3), then run 0 (0.5), then run 2 (0.4).
            # So we should see exactly one "NEW BEST" for run 1.
            def ordered_as_completed(future_to_index):
                order = [1, 0, 2]
                for run_idx in order:
                    for fut, idx in future_to_index.items():
                        if idx == run_idx:
                            yield fut
                            break

            print_calls = []

            def capture_print(*args, **kwargs):
                print_calls.append(" ".join(str(a) for a in args))

            with (
                mock.patch.object(
                    hyperparameter_search_arrow.futures,
                    "ProcessPoolExecutor",
                    return_value=FakeExecutor(),
                ),
                mock.patch.object(
                    hyperparameter_search_arrow.futures,
                    "as_completed",
                    side_effect=ordered_as_completed,
                ),
                mock.patch("builtins.print", side_effect=capture_print),
                mock.patch(
                    "sys.argv",
                    [
                        "hyperparameter_search_arrow",
                        "--sweep_config",
                        sweep_path,
                        "--workers",
                        "1",
                    ],
                ),
            ):
                exit_code = hyperparameter_search_arrow.main()
            self.assertEqual(exit_code, 0)
            out = "\n".join(print_calls)
            self.assertIn("NEW BEST", out, "Expected at least one 'NEW BEST' message")
            self.assertIn(
                "run 2/3",
                out,
                "NEW BEST should mention run 2/3 (run index 1, best metric)",
            )
            self.assertIn(
                "0.300000",
                out,
                "NEW BEST should show best value 0.3",
            )
            # Only run 1 is best when it completes first; runs 0 and 2 are worse
            self.assertEqual(
                out.count("NEW BEST"),
                1,
                "Exactly one run should be 'new best' when best completes first",
            )


@pytest.mark.slow
@pytest.mark.memory
class MemoryBoundedSweepTest(unittest.TestCase):
    """In-process sweep runs: assert process RSS growth is bounded (no unbounded leak)."""

    def test_in_process_sweep_rss_bounded(self):
        """Run 4 in-process training runs; RSS growth (max - min) stays under threshold."""
        try:
            import psutil
        except ImportError:
            self.skipTest("psutil required for memory test")
        base_config_path = os.path.join(
            os.path.dirname(__file__), "..", "configs", "arrow_baseline.json"
        )
        if not os.path.isfile(base_config_path):
            self.skipTest("configs/arrow_baseline.json not found")
        base_config = config.ArrowExperimentConfig.from_json(base_config_path)
        base_config.dataset.data_dir = TEST_DATA_DIR
        base_config.dataset.val_data_dir = TEST_DATA_DIR
        base_config.run.take_count = 2
        base_config.run.epoch = 1
        search_space = {"model.transformer.dropout_rate": [0.0, 0.1, 0.2, 0.25]}
        combinations = hyperparameter_search_arrow.expand_grid(search_space)
        self.assertGreaterEqual(len(combinations), 4)
        combinations = combinations[:4]
        process = psutil.Process(os.getpid())
        rss_after_run = []
        with tempfile.TemporaryDirectory() as temp_dir:
            for i, overrides in enumerate(combinations):
                run_config = hyperparameter_search_arrow.apply_overrides(
                    base_config, overrides
                )
                run_config.run.model_output_dir = os.path.join(
                    temp_dir, "models", f"run_{i}"
                )
                run_config.run.callback_root_dir = os.path.join(
                    temp_dir, "callbacks", f"run_{i}"
                )
                os.makedirs(run_config.run.model_output_dir, exist_ok=True)
                os.makedirs(run_config.run.callback_root_dir, exist_ok=True)
                model, history = trainers.run_arrow_train_from_config(
                    run_config.dataset,
                    run_config.model,
                    run_config.run,
                )
                hyperparameter_search_arrow.extract_metrics(history)
                del model, history
                hyperparameter_search_arrow._clear_tf_memory()
                rss_after_run.append(process.memory_info().rss)
        self.assertEqual(len(rss_after_run), 4)
        rss_min, rss_max = min(rss_after_run), max(rss_after_run)
        growth_bytes = rss_max - rss_min
        # Threshold: TF often does not return memory to OS so some growth is normal; catch severe leaks (e.g. full model/dataset per run). Calibrate per environment if needed.
        threshold_bytes = 800 * 1024 * 1024
        self.assertLess(
            growth_bytes,
            threshold_bytes,
            f"RSS growth {growth_bytes / (1024 * 1024):.1f} MB exceeds {threshold_bytes // (1024 * 1024)} MB (unbounded memory?)",
        )


@pytest.mark.slow
class SweepVerbosityTest(unittest.TestCase):
    """Sweep sets show_model_summary=False and fit_verbose=0 for each run."""

    def test_sweep_passes_quiet_verbosity_to_trainer(self):
        """Sweep saves config with show_model_summary=False and fit_verbose=0."""
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
                        "search_space": {"model.transformer.dropout_rate": [0.0]},
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
            with open(sweep_path) as f:
                sweep_data = json.load(f)
            sweep_data["base_config"] = base_config_override
            with open(sweep_path, "w") as f:
                json.dump(sweep_data, f)

            result = _run_sweep_script("--sweep_config", os.path.abspath(sweep_path))
            self.assertEqual(result.returncode, 0, result.stderr or result.stdout)

            # Config is saved under callbacks/run_0/logs/<callback_name>/config.json
            callbacks_run0 = os.path.join(temp_dir, "sweep_out", "callbacks", "run_0")
            logs_dir = os.path.join(callbacks_run0, "logs")
            self.assertTrue(os.path.isdir(logs_dir), f"Missing {logs_dir}")
            log_subdirs = [
                d
                for d in os.listdir(logs_dir)
                if os.path.isdir(os.path.join(logs_dir, d))
            ]
            self.assertGreater(len(log_subdirs), 0, "No log subdir found")
            config_path = os.path.join(logs_dir, log_subdirs[0], "config.json")
            self.assertTrue(os.path.isfile(config_path), f"Missing {config_path}")
            with open(config_path) as f:
                saved = json.load(f)
            run_saved = saved["run"]
            self.assertFalse(
                run_saved.get("show_model_summary", True),
                "sweep should set show_model_summary=False",
            )
            self.assertEqual(
                run_saved.get("fit_verbose", 1),
                0,
                "sweep should set fit_verbose=0",
            )


if __name__ == "__main__":
    unittest.main()
