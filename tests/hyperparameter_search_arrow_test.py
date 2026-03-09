import concurrent.futures
import json
import os
import random
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


def _minimal_inline_experiment_config(
    model_output_dir="models",
    callback_root_dir="callbacks",
):
    return {
        "dataset": {
            "data_dir": TEST_DATA_DIR,
            "val_data_dir": TEST_DATA_DIR,
            "batch_size": 1,
        },
        "model": {
            "model_type": "transformer",
            "transformer": {
                "num_layers": 1,
                "d_model": 128,
                "num_heads": 4,
                "ff_dim": 512,
                "dropout_rate": 0.0,
            },
        },
        "run": {
            "epoch": 2,
            "take_count": 2,
            "model_output_dir": model_output_dir,
            "callback_root_dir": callback_root_dir,
            "model_name": "arrow_test_model",
            "seed": 42,
        },
    }


def _minimal_sweep_config(
    sweep_output_dir,
    *,
    search_space=None,
    optimize=None,
    extra=None,
):
    data = {
        **_minimal_inline_experiment_config(),
        "search_space": search_space or {"model.transformer.dropout_rate": [0.0]},
        "optimize": optimize or {"metric": "val_loss", "mode": "min"},
        "sweep_output_dir": sweep_output_dir,
    }
    if extra:
        data.update(extra)
    return data


def _write_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)


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

    def _write_sweep_config(self, tmpdir, data):
        path = os.path.join(tmpdir, "sweep_config.json")
        _write_json(path, data)
        return path

    def test_load_valid_sweep_config(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._write_sweep_config(
                tmpdir,
                {
                    **_minimal_inline_experiment_config(),
                    "search_space": {
                        "model.transformer.dropout_rate": [0.0, 0.1],
                        "run.chart_validity_aux_weight": [0.0],
                    },
                    "optimize": {"metric": "val_loss", "mode": "min"},
                },
            )
            data = hyperparameter_search_arrow.load_sweep_config(path)
            self.assertIsInstance(data, dict)
            self.assertEqual(data["dataset"]["data_dir"], TEST_DATA_DIR)
            self.assertEqual(
                data["search_space"]["model.transformer.dropout_rate"], [0.0, 0.1]
            )
            self.assertEqual(
                data["search_space"]["run.chart_validity_aux_weight"], [0.0]
            )
            self.assertEqual(data["optimize"]["metric"], "val_loss")
            self.assertEqual(data["optimize"]["mode"], "min")

    def test_load_rejects_missing_required_inline_config_section(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            for missing_key in ("dataset", "model", "run"):
                with self.subTest(missing_key=missing_key):
                    data = {
                        **_minimal_inline_experiment_config(),
                        "search_space": {"model.transformer.dropout_rate": [0.0]},
                        "optimize": {"metric": "val_loss", "mode": "min"},
                    }
                    del data[missing_key]
                    path = self._write_sweep_config(tmpdir, data)
                    with self.assertRaises(ValueError) as ctx:
                        hyperparameter_search_arrow.load_sweep_config(path)
                    self.assertIn(missing_key, str(ctx.exception))

    def test_load_rejects_missing_optimize(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._write_sweep_config(
                tmpdir,
                {
                    **_minimal_inline_experiment_config(),
                    "search_space": {"model.transformer.dropout_rate": [0.0]},
                },
            )
            with self.assertRaises(ValueError) as ctx:
                hyperparameter_search_arrow.load_sweep_config(path)
            self.assertIn("optimize", str(ctx.exception))

    def test_load_rejects_invalid_optimize_mode(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._write_sweep_config(
                tmpdir,
                {
                    **_minimal_inline_experiment_config(),
                    "search_space": {"model.transformer.dropout_rate": [0.0]},
                    "optimize": {"metric": "val_loss", "mode": "invalid"},
                },
            )
            with self.assertRaises(ValueError) as ctx:
                hyperparameter_search_arrow.load_sweep_config(path)
            self.assertIn("mode", str(ctx.exception))

    def test_load_rejects_forbidden_key_run_val_take_count(self):
        """run.val_take_count is fixed; cannot be in search_space."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._write_sweep_config(
                tmpdir,
                {
                    **_minimal_inline_experiment_config(),
                    "search_space": {"run.val_take_count": [1, 2]},
                    "optimize": {"metric": "val_loss", "mode": "min"},
                },
            )
            with self.assertRaises(ValueError) as ctx:
                hyperparameter_search_arrow.load_sweep_config(path)
            self.assertIn("forbidden", str(ctx.exception).lower())

    def test_load_accepts_search_grid_and_random(self):
        """Sweep config may contain 'search': 'grid' or 'random'."""
        for search_val in ("grid", "random"):
            with tempfile.TemporaryDirectory() as tmpdir:
                path = self._write_sweep_config(
                    tmpdir,
                    {
                        **_minimal_inline_experiment_config(),
                        "search_space": {"model.transformer.dropout_rate": [0.0]},
                        "optimize": {"metric": "val_loss", "mode": "min"},
                        "search": search_val,
                    },
                )
                data = hyperparameter_search_arrow.load_sweep_config(path)
                self.assertEqual(data["search"], search_val)

    def test_load_rejects_invalid_search(self):
        """Sweep config 'search' must be 'grid' or 'random'."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._write_sweep_config(
                tmpdir,
                {
                    **_minimal_inline_experiment_config(),
                    "search_space": {"model.transformer.dropout_rate": [0.0]},
                    "optimize": {"metric": "val_loss", "mode": "min"},
                    "search": "monte_carlo",
                },
            )
            with self.assertRaises(ValueError) as ctx:
                hyperparameter_search_arrow.load_sweep_config(path)
            self.assertIn("search", str(ctx.exception))

    def test_load_accepts_optional_workers(self):
        """Sweep config may include optional 'workers' (int >= 1)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "sweep.json")
            with open(path, "w") as f:
                json.dump(
                    {
                        **_minimal_inline_experiment_config(),
                        "search_space": {"model.transformer.dropout_rate": [0.0]},
                        "optimize": {"metric": "val_loss", "mode": "min"},
                        "workers": 4,
                    },
                    f,
                )
            data = hyperparameter_search_arrow.load_sweep_config(path)
            self.assertEqual(data["workers"], 4)

    def test_load_rejects_invalid_workers(self):
        """Sweep config 'workers' must be an integer >= 1."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "sweep.json")
            with open(path, "w") as f:
                json.dump(
                    {
                        **_minimal_inline_experiment_config(),
                        "search_space": {"model.transformer.dropout_rate": [0.0]},
                        "optimize": {"metric": "val_loss", "mode": "min"},
                        "workers": 0,
                    },
                    f,
                )
            with self.assertRaises(ValueError) as ctx:
                hyperparameter_search_arrow.load_sweep_config(path)
            self.assertIn("workers", str(ctx.exception))

    def test_load_accepts_validity_gate(self):
        """Sweep config with validity_gate (min_fraction and optional fields) loads."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._write_sweep_config(
                tmpdir,
                {
                    **_minimal_inline_experiment_config(),
                    "search_space": {"run.epoch": [1]},
                    "optimize": {"metric": "val_loss", "mode": "min"},
                    "validity_gate": {
                        "min_fraction": 0.95,
                        "validity_metric": "val_chart_validity_pass_rate_0_99",
                        "optimize_metric": "val_arrow_dist_match",
                        "optimize_mode": "max",
                    },
                },
            )
            data = hyperparameter_search_arrow.load_sweep_config(path)
            self.assertIn("validity_gate", data)
            self.assertEqual(data["validity_gate"]["min_fraction"], 0.95)
            self.assertEqual(
                data["validity_gate"]["validity_metric"],
                "val_chart_validity_pass_rate_0_99",
            )
            self.assertEqual(
                data["validity_gate"]["optimize_metric"], "val_arrow_dist_match"
            )
            self.assertEqual(data["validity_gate"]["optimize_mode"], "max")

    def test_load_validity_gate_requires_min_fraction(self):
        """validity_gate without min_fraction raises."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._write_sweep_config(
                tmpdir,
                {
                    **_minimal_inline_experiment_config(),
                    "search_space": {"run.epoch": [1]},
                    "optimize": {"metric": "val_loss", "mode": "min"},
                    "validity_gate": {},
                },
            )
            with self.assertRaises(ValueError) as ctx:
                hyperparameter_search_arrow.load_sweep_config(path)
            self.assertIn("min_fraction", str(ctx.exception))

    def test_load_validity_gate_min_fraction_range(self):
        """validity_gate.min_fraction must be in [0, 1]."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._write_sweep_config(
                tmpdir,
                {
                    **_minimal_inline_experiment_config(),
                    "search_space": {"run.epoch": [1]},
                    "optimize": {"metric": "val_loss", "mode": "min"},
                    "validity_gate": {"min_fraction": 1.5},
                },
            )
            with self.assertRaises(ValueError) as ctx:
                hyperparameter_search_arrow.load_sweep_config(path)
            self.assertIn("min_fraction", str(ctx.exception))

    def test_load_validity_gate_rejects_empty_validity_metric(self):
        """validity_gate.validity_metric must be non-empty when set."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._write_sweep_config(
                tmpdir,
                {
                    **_minimal_inline_experiment_config(),
                    "search_space": {"run.epoch": [1]},
                    "optimize": {"metric": "val_loss", "mode": "min"},
                    "validity_gate": {
                        "min_fraction": 0.95,
                        "validity_metric": "",
                    },
                },
            )
            with self.assertRaises(ValueError) as ctx:
                hyperparameter_search_arrow.load_sweep_config(path)
            self.assertIn("validity_metric", str(ctx.exception))

    def test_load_validity_gate_invalid_mode(self):
        """validity_gate.optimize_mode must be min or max when set."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._write_sweep_config(
                tmpdir,
                {
                    **_minimal_inline_experiment_config(),
                    "search_space": {"run.epoch": [1]},
                    "optimize": {"metric": "val_loss", "mode": "min"},
                    "validity_gate": {
                        "min_fraction": 0.95,
                        "optimize_mode": "invalid",
                    },
                },
            )
            with self.assertRaises(ValueError) as ctx:
                hyperparameter_search_arrow.load_sweep_config(path)
            self.assertIn("optimize_mode", str(ctx.exception))


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

    def test_expand_grid_without_model_type_falls_back_to_simple_grid(self):
        """Without model.model_type, expand_grid delegates to the simple Cartesian product."""
        search_space = {
            "model.tcn.filters": [64, 128],
            "run.epoch": [1],
        }
        with mock.patch.object(
            hyperparameter_search_arrow,
            "_expand_grid_simple",
            wraps=hyperparameter_search_arrow._expand_grid_simple,
        ) as mock_expand:
            combinations = hyperparameter_search_arrow.expand_grid(search_space)
        mock_expand.assert_called_once_with(search_space)
        self.assertEqual(
            combinations,
            [
                {"model.tcn.filters": 64, "run.epoch": 1},
                {"model.tcn.filters": 128, "run.epoch": 1},
            ],
        )

    def test_expand_grid_single_model_type_falls_back_to_simple_grid(self):
        """A single model.model_type value uses the simple Cartesian product path."""
        search_space = {
            "model.model_type": ["tcn"],
            "model.tcn.filters": [64, 128],
            "run.epoch": [1, 2],
        }
        with mock.patch.object(
            hyperparameter_search_arrow,
            "_expand_grid_simple",
            wraps=hyperparameter_search_arrow._expand_grid_simple,
        ) as mock_expand:
            combinations = hyperparameter_search_arrow.expand_grid(search_space)
        mock_expand.assert_called_once_with(search_space)
        self.assertEqual(len(combinations), 4)
        self.assertEqual(
            combinations,
            [
                {
                    "model.model_type": "tcn",
                    "model.tcn.filters": 64,
                    "run.epoch": 1,
                },
                {
                    "model.model_type": "tcn",
                    "model.tcn.filters": 64,
                    "run.epoch": 2,
                },
                {
                    "model.model_type": "tcn",
                    "model.tcn.filters": 128,
                    "run.epoch": 1,
                },
                {
                    "model.model_type": "tcn",
                    "model.tcn.filters": 128,
                    "run.epoch": 2,
                },
            ],
        )

    def test_expand_grid_multi_model_type_only_matching_block_per_combo(self):
        """With multiple model.model_type values, each combo has only that type's model block keys."""
        search_space = {
            "model.model_type": ["tcn", "gru"],
            "model.tcn.filters": [64, 128],
            "model.gru.units": [32],
            "run.epoch": [1],
        }
        combinations = hyperparameter_search_arrow.expand_grid(search_space)
        # tcn: 2 filters * 1 epoch = 2; gru: 1 units * 1 epoch = 1 → 3 total
        self.assertEqual(len(combinations), 3)
        tcn_combos = [c for c in combinations if c.get("model.model_type") == "tcn"]
        gru_combos = [c for c in combinations if c.get("model.model_type") == "gru"]
        self.assertEqual(len(tcn_combos), 2)
        self.assertEqual(len(gru_combos), 1)
        for c in tcn_combos:
            self.assertIn("model.tcn.filters", c)
            self.assertNotIn("model.gru.units", c)
        for c in gru_combos:
            self.assertIn("model.gru.units", c)
            self.assertNotIn("model.tcn.filters", c)
        self.assertEqual(
            {c["model.tcn.filters"] for c in tcn_combos},
            {64, 128},
        )
        self.assertEqual(gru_combos[0]["model.gru.units"], 32)

    def test_expand_grid_multi_model_type_keeps_global_model_keys(self):
        """Global model.<param> keys are included for every multi-model combination."""
        search_space = {
            "model.model_type": ["tcn", "gru"],
            "model.learning_rate": [0.001, 0.01],
            "model.tcn.filters": [64],
            "model.gru.units": [32],
            "run.epoch": [1],
        }
        combinations = hyperparameter_search_arrow.expand_grid(search_space)
        self.assertEqual(len(combinations), 4)
        expected = [
            {
                "model.model_type": "tcn",
                "model.learning_rate": 0.001,
                "model.tcn.filters": 64,
                "run.epoch": 1,
            },
            {
                "model.model_type": "tcn",
                "model.learning_rate": 0.01,
                "model.tcn.filters": 64,
                "run.epoch": 1,
            },
            {
                "model.model_type": "gru",
                "model.learning_rate": 0.001,
                "model.gru.units": 32,
                "run.epoch": 1,
            },
            {
                "model.model_type": "gru",
                "model.learning_rate": 0.01,
                "model.gru.units": 32,
                "run.epoch": 1,
            },
        ]
        self.assertEqual(combinations, expected)


class FilterValidModelCombinationsTest(unittest.TestCase):
    """filter_valid_model_combinations: only keep combos where model.<block>.* matches model_type."""

    def test_kept_when_model_type_and_block_match(self):
        """Combination with model.model_type=transformer and model.transformer.d_model is kept."""
        combinations = [
            {
                "model.model_type": "transformer",
                "model.transformer.d_model": 128,
                "run.epoch": 10,
            }
        ]
        out = hyperparameter_search_arrow.filter_valid_model_combinations(
            combinations, default_model_type="gru"
        )
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["model.model_type"], "transformer")
        self.assertEqual(out[0]["model.transformer.d_model"], 128)

    def test_excluded_when_model_type_differs_from_block(self):
        """Combination with model.model_type=mlp but model.transformer.d_model is excluded."""
        combinations = [
            {"model.model_type": "mlp", "model.transformer.d_model": 128},
            {"model.model_type": "transformer", "model.transformer.d_model": 128},
        ]
        out = hyperparameter_search_arrow.filter_valid_model_combinations(
            combinations, default_model_type="gru"
        )
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["model.model_type"], "transformer")

    def test_default_model_type_used_when_not_in_overrides(self):
        """When model.model_type is not in overrides, default is used; transformer block kept if default is transformer."""
        combinations = [
            {"model.transformer.d_model": 128},
            {"model.mlp.hidden_dims": [64]},
        ]
        out = hyperparameter_search_arrow.filter_valid_model_combinations(
            combinations, default_model_type="transformer"
        )
        self.assertEqual(len(out), 1)
        self.assertIn("model.transformer.d_model", out[0])
        self.assertEqual(out[0]["model.transformer.d_model"], 128)

    def test_no_model_block_keys_all_kept(self):
        """Search space with only dataset/run keys: no combinations excluded."""
        combinations = [
            {"dataset.use_interval": True, "run.epoch": 10},
            {"dataset.use_interval": False, "run.epoch": 20},
        ]
        out = hyperparameter_search_arrow.filter_valid_model_combinations(
            combinations, default_model_type="transformer"
        )
        self.assertEqual(len(out), 2)
        self.assertEqual(out, combinations)

    def test_mixed_model_types_filters_correctly(self):
        """Grid with model_type in [mlp, transformer] and transformer.d_model: only transformer combos kept."""
        combinations = [
            {
                "model.model_type": "mlp",
                "model.transformer.d_model": 128,
                "run.epoch": 10,
            },
            {
                "model.model_type": "transformer",
                "model.transformer.d_model": 128,
                "run.epoch": 10,
            },
        ]
        out = hyperparameter_search_arrow.filter_valid_model_combinations(
            combinations, default_model_type="gru"
        )
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["model.model_type"], "transformer")
        self.assertEqual(out[0]["model.transformer.d_model"], 128)

    def test_empty_combinations_returns_empty(self):
        out = hyperparameter_search_arrow.filter_valid_model_combinations(
            [], default_model_type="transformer"
        )
        self.assertEqual(out, [])


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

    def test_set_nested_promotes_none_to_dict_when_setting_nested_key(self):
        """_set_nested promotes None to {} when setting a nested key so param blocks can be created."""
        d = {"model": {"transformer": None}}
        hyperparameter_search_arrow._set_nested(d["model"], "transformer.num_layers", 2)
        self.assertIsInstance(d["model"]["transformer"], dict)
        assert d["model"]["transformer"] is not None
        self.assertEqual(d["model"]["transformer"]["num_layers"], 2)

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
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "sweep.json")
            _write_json(
                path,
                {
                    **_minimal_inline_experiment_config(),
                    "search_space": {"run.val_take_count": [1, 2]},
                    "optimize": {"metric": "val_loss", "mode": "min"},
                },
            )
            with self.assertRaises(ValueError) as ctx:
                hyperparameter_search_arrow.load_sweep_config(path)
            self.assertIn("forbidden", str(ctx.exception).lower())

    def test_load_sweep_config_allows_snippet_half_frames_search_space_keys(self):
        """snippet_half_frames keys are no longer treated as forbidden search-space entries."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "sweep.json")
            _write_json(
                path,
                {
                    **_minimal_inline_experiment_config(),
                    "search_space": {
                        "dataset.snippet_half_frames": [0, 5],
                        "model.snippet_half_frames": [0, 5],
                    },
                    "optimize": {"metric": "val_loss", "mode": "min"},
                },
            )
            data = hyperparameter_search_arrow.load_sweep_config(path)
            self.assertEqual(
                data["search_space"]["dataset.snippet_half_frames"], [0, 5]
            )
            self.assertEqual(data["search_space"]["model.snippet_half_frames"], [0, 5])


class SweepCombinationValidationTest(unittest.TestCase):
    """Fresh sweeps validate all expanded combinations before training starts."""

    def test_main_rejects_invalid_full_grid_before_random_sampling(self):
        """Random search validates the whole grid before sampling or launching workers."""
        with tempfile.TemporaryDirectory() as temp_dir:
            sweep_output_dir = os.path.join(temp_dir, "sweep_out")
            sweep_path = os.path.join(temp_dir, "sweep.json")
            _write_json(
                sweep_path,
                _minimal_sweep_config(
                    sweep_output_dir,
                    search_space={
                        "run.epoch": [101, 10],
                        "run.warmup_epochs": [100],
                    },
                    extra={"search": "random", "max_runs": 1, "seed": 7},
                ),
            )
            with (
                mock.patch.object(
                    hyperparameter_search_arrow.random,
                    "sample",
                    return_value=[{"run.epoch": 101, "run.warmup_epochs": 100}],
                ) as sample_mock,
                mock.patch.object(
                    hyperparameter_search_arrow.futures,
                    "ProcessPoolExecutor",
                ) as executor_mock,
                mock.patch(
                    "sys.argv",
                    [
                        "hyperparameter_search_arrow",
                        "--sweep_config",
                        sweep_path,
                        "--search",
                        "random",
                        "--max_runs",
                        "1",
                        "--seed",
                        "7",
                    ],
                ),
            ):
                with self.assertRaises(ValueError) as ctx:
                    hyperparameter_search_arrow.main()
            self.assertIn("warmup_epochs", str(ctx.exception))
            sample_mock.assert_not_called()
            executor_mock.assert_not_called()
            self.assertFalse(os.path.exists(sweep_output_dir))

    def test_main_rejects_mismatched_model_block_before_training(self):
        """Fresh setup fails fast when search_space targets a different model block."""
        with tempfile.TemporaryDirectory() as temp_dir:
            sweep_output_dir = os.path.join(temp_dir, "sweep_out")
            sweep_path = os.path.join(temp_dir, "sweep.json")
            _write_json(
                sweep_path,
                _minimal_sweep_config(
                    sweep_output_dir,
                    search_space={"model.gru.units": [64]},
                ),
            )
            with (
                mock.patch.object(
                    hyperparameter_search_arrow.futures,
                    "ProcessPoolExecutor",
                ) as executor_mock,
                mock.patch(
                    "sys.argv",
                    [
                        "hyperparameter_search_arrow",
                        "--sweep_config",
                        sweep_path,
                    ],
                ),
            ):
                with self.assertRaises(ValueError) as ctx:
                    hyperparameter_search_arrow.main()
            self.assertIn("model.gru.units", str(ctx.exception))
            self.assertIn("model_type", str(ctx.exception))
            executor_mock.assert_not_called()
            self.assertFalse(os.path.exists(sweep_output_dir))


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


class ResolveValidityMetricTest(unittest.TestCase):
    """_resolve_validity_metric: explicit metric or auto-detect from result keys."""

    def test_explicit_metric_returns_best_key(self):
        """When validity_metric is set, return best_<metric>."""
        result = {"best_val_chart_validity_pass_rate_0_99": 1.0}
        key = hyperparameter_search_arrow._resolve_validity_metric(
            [result], "val_chart_validity_pass_rate_0_99"
        )
        self.assertEqual(key, "best_val_chart_validity_pass_rate_0_99")

    def test_auto_detect_prefers_pass_rate(self):
        """Auto-detect prefers best_val_chart_validity_pass_rate_* over best_val_chart_validity."""
        result = {
            "best_val_chart_validity": 0.9,
            "best_val_chart_validity_pass_rate_0_99": 1.0,
        }
        key = hyperparameter_search_arrow._resolve_validity_metric([result], None)
        self.assertEqual(key, "best_val_chart_validity_pass_rate_0_99")

    def test_auto_detect_prefers_highest_pass_rate_threshold(self):
        """Auto-detect chooses the strictest pass-rate threshold when multiple exist."""
        result = {
            "best_val_chart_validity_pass_rate_0_95": 1.0,
            "best_val_chart_validity_pass_rate_0_99": 0.8,
        }
        key = hyperparameter_search_arrow._resolve_validity_metric([result], None)
        self.assertEqual(key, "best_val_chart_validity_pass_rate_0_99")

    def test_auto_detect_fallback_chart_validity(self):
        """When no pass_rate key, use best_val_chart_validity."""
        result = {"best_val_chart_validity": 0.95, "best_val_loss": 0.5}
        key = hyperparameter_search_arrow._resolve_validity_metric([result], None)
        self.assertEqual(key, "best_val_chart_validity")

    def test_auto_detect_empty_results_returns_none(self):
        """Empty results returns None."""
        key = hyperparameter_search_arrow._resolve_validity_metric([], None)
        self.assertIsNone(key)

    def test_auto_detect_no_validity_key_returns_none(self):
        """When no validity key in results, return None."""
        result = {"best_val_loss": 0.5, "best_val_acc": 0.8}
        key = hyperparameter_search_arrow._resolve_validity_metric([result], None)
        self.assertIsNone(key)


class SelectBestRunWithValidityGateTest(unittest.TestCase):
    """_select_best_run_with_validity_gate: filter by validity then optimize by metric."""

    def _sweep_save(self, validity_gate=None):
        out = {"optimize": {"metric": "val_main_loss", "mode": "min"}}
        if validity_gate is not None:
            out["validity_gate"] = validity_gate
        return out

    def test_no_gate_same_as_select_best_run(self):
        """Without validity_gate, returns same index as select_best_run by main metric."""
        results = [
            {
                "overrides": {},
                "best_val_main_loss": 2.5,
                "best_val_arrow_dist_match": 0.8,
            },
            {
                "overrides": {},
                "best_val_main_loss": 2.0,
                "best_val_arrow_dist_match": 0.7,
            },
            {
                "overrides": {},
                "best_val_main_loss": 2.2,
                "best_val_arrow_dist_match": 0.9,
            },
        ]
        sweep_save = self._sweep_save()
        idx, gate_info = (
            hyperparameter_search_arrow._select_best_run_with_validity_gate(
                results, sweep_save
            )
        )
        self.assertEqual(idx, 1)
        self.assertIsNone(gate_info)

    def test_gate_all_pass_best_by_secondary_metric(self):
        """When all runs pass the gate, best is by gate's optimize_metric (max arrow_dist_match)."""
        results = [
            {
                "overrides": {},
                "best_val_main_loss": 2.5,
                "best_val_arrow_dist_match": 0.8,
                "best_val_chart_validity_pass_rate_0_99": 1.0,
            },
            {
                "overrides": {},
                "best_val_main_loss": 2.0,
                "best_val_arrow_dist_match": 0.95,
                "best_val_chart_validity_pass_rate_0_99": 1.0,
            },
            {
                "overrides": {},
                "best_val_main_loss": 2.2,
                "best_val_arrow_dist_match": 0.85,
                "best_val_chart_validity_pass_rate_0_99": 0.98,
            },
        ]
        sweep_save = self._sweep_save(
            {
                "min_fraction": 0.95,
                "validity_metric": "val_chart_validity_pass_rate_0_99",
                "optimize_metric": "val_arrow_dist_match",
                "optimize_mode": "max",
            }
        )
        idx, gate_info = (
            hyperparameter_search_arrow._select_best_run_with_validity_gate(
                results, sweep_save
            )
        )
        self.assertEqual(idx, 1)
        assert gate_info is not None
        self.assertFalse(gate_info["used_fallback"])
        self.assertEqual(gate_info["n_passed"], 3)
        self.assertEqual(gate_info["n_total"], 3)

    def test_gate_some_pass_best_among_passing(self):
        """When only some runs pass, best is the one with highest arrow_dist_match among passing."""
        results = [
            {
                "overrides": {},
                "best_val_main_loss": 2.5,
                "best_val_arrow_dist_match": 0.9,
                "best_val_chart_validity_pass_rate_0_99": 0.8,
            },
            {
                "overrides": {},
                "best_val_main_loss": 2.2,
                "best_val_arrow_dist_match": 0.85,
                "best_val_chart_validity_pass_rate_0_99": 1.0,
            },
            {
                "overrides": {},
                "best_val_main_loss": 2.0,
                "best_val_arrow_dist_match": 0.95,
                "best_val_chart_validity_pass_rate_0_99": 0.97,
            },
        ]
        sweep_save = self._sweep_save(
            {
                "min_fraction": 0.95,
                "validity_metric": "val_chart_validity_pass_rate_0_99",
                "optimize_metric": "val_arrow_dist_match",
                "optimize_mode": "max",
            }
        )
        idx, gate_info = (
            hyperparameter_search_arrow._select_best_run_with_validity_gate(
                results, sweep_save
            )
        )
        self.assertEqual(idx, 2)
        assert gate_info is not None
        self.assertFalse(gate_info["used_fallback"])
        self.assertEqual(gate_info["n_passed"], 2)

    def test_gate_none_pass_fallback(self):
        """When no run passes the gate, fall back to main optimize metric and set used_fallback."""
        results = [
            {
                "overrides": {},
                "best_val_main_loss": 2.0,
                "best_val_arrow_dist_match": 0.9,
                "best_val_chart_validity_pass_rate_0_99": 0.5,
            },
            {
                "overrides": {},
                "best_val_main_loss": 2.5,
                "best_val_arrow_dist_match": 0.95,
                "best_val_chart_validity_pass_rate_0_99": 0.4,
            },
        ]
        sweep_save = self._sweep_save(
            {
                "min_fraction": 0.95,
                "validity_metric": "val_chart_validity_pass_rate_0_99",
                "optimize_metric": "val_arrow_dist_match",
                "optimize_mode": "max",
            }
        )
        idx, gate_info = (
            hyperparameter_search_arrow._select_best_run_with_validity_gate(
                results, sweep_save
            )
        )
        self.assertEqual(idx, 0)
        assert gate_info is not None
        self.assertTrue(gate_info["used_fallback"])
        self.assertEqual(gate_info["n_passed"], 0)

    def test_gate_none_pass_fallback_prints_warning(self):
        """When no run passes, fallback selects by main metric; capture print for warning."""
        results = [
            {
                "overrides": {},
                "best_val_main_loss": 2.0,
                "best_val_chart_validity_pass_rate_0_99": 0.5,
            },
            {
                "overrides": {},
                "best_val_main_loss": 2.5,
                "best_val_chart_validity_pass_rate_0_99": 0.4,
            },
        ]
        sweep_save = self._sweep_save(
            {
                "min_fraction": 0.99,
                "validity_metric": "val_chart_validity_pass_rate_0_99",
            }
        )
        with mock.patch("builtins.print") as mock_print:
            idx, gate_info = (
                hyperparameter_search_arrow._select_best_run_with_validity_gate(
                    results, sweep_save
                )
            )
        self.assertEqual(idx, 0)
        assert gate_info is not None
        self.assertTrue(gate_info["used_fallback"])
        mock_print.assert_called()
        call_str = " ".join(str(c) for c in mock_print.call_args_list)
        self.assertIn("validity gate", call_str.lower())
        self.assertIn("WARNING", call_str)

    def test_gate_missing_validity_metric_falls_back_to_main_metric(self):
        """When no validity metric is present in results, fallback uses the main optimize metric."""
        results = [
            {
                "overrides": {},
                "best_val_main_loss": 2.0,
                "best_val_arrow_dist_match": 0.9,
            },
            {
                "overrides": {},
                "best_val_main_loss": 1.8,
                "best_val_arrow_dist_match": 0.7,
            },
        ]
        sweep_save = self._sweep_save({"min_fraction": 0.95})
        with mock.patch("builtins.print") as mock_print:
            idx, gate_info = (
                hyperparameter_search_arrow._select_best_run_with_validity_gate(
                    results, sweep_save
                )
            )
        self.assertEqual(idx, 1)
        assert gate_info is not None
        self.assertEqual(gate_info["validity_metric"], "(auto)")
        self.assertEqual(gate_info["min_fraction"], 0.95)
        self.assertEqual(gate_info["n_passed"], 0)
        self.assertEqual(gate_info["n_total"], 2)
        self.assertTrue(gate_info["used_fallback"])
        mock_print.assert_called()
        call_str = " ".join(str(c) for c in mock_print.call_args_list)
        self.assertIn("no validity metric found", call_str.lower())
        self.assertIn("main optimize metric", call_str.lower())


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
        with tempfile.TemporaryDirectory() as temp_dir:
            sweep_path = os.path.join(temp_dir, "sweep.json")
            _write_json(
                sweep_path,
                _minimal_sweep_config(
                    os.path.join(temp_dir, "sweep_out"),
                    search_space={"model.transformer.dropout_rate": [0.0]},
                ),
            )

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
        with tempfile.TemporaryDirectory() as temp_dir:
            sweep_path = os.path.join(temp_dir, "sweep.json")
            _write_json(
                sweep_path,
                _minimal_sweep_config(
                    os.path.join(temp_dir, "sweep_out"),
                    search_space={
                        "model.transformer.dropout_rate": [0.0, 0.1],
                        "run.chart_validity_aux_weight": [0.0, 0.3],
                    },
                ),
            )

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
        with tempfile.TemporaryDirectory() as temp_dir:
            sweep_path = os.path.join(temp_dir, "sweep.json")
            _write_json(
                sweep_path,
                _minimal_sweep_config(
                    os.path.join(temp_dir, "sweep_out"),
                    search_space={
                        "model.transformer.dropout_rate": [0.0, 0.1],
                        "run.chart_validity_aux_weight": [0.0, 0.3],
                    },
                    extra={"search": "random", "max_runs": 2, "seed": 42},
                ),
            )

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
            _write_json(
                sweep_path,
                _minimal_sweep_config(
                    temp_dir,
                    search_space={"model.transformer.dropout_rate": [0.0]},
                ),
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
        with tempfile.TemporaryDirectory() as temp_dir:
            resume_dir = os.path.join(temp_dir, "sweep_resume")
            os.makedirs(resume_dir, exist_ok=True)
            os.makedirs(os.path.join(resume_dir, "models"), exist_ok=True)
            os.makedirs(os.path.join(resume_dir, "callbacks"), exist_ok=True)

            search_space = {"model.transformer.dropout_rate": [0.0, 0.1]}
            combinations = hyperparameter_search_arrow.expand_grid(search_space)
            sweep_config = {
                **_minimal_inline_experiment_config(),
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
        with tempfile.TemporaryDirectory() as tmpdir:
            sweep_path = os.path.join(tmpdir, "sweep.json")
            _write_json(
                sweep_path,
                _minimal_sweep_config(
                    os.path.join(tmpdir, "sweep_out"),
                    search_space={"model.transformer.dropout_rate": [0.0]},
                ),
            )
            with (
                mock.patch(
                    "sys.argv",
                    [
                        "hyperparameter_search_arrow",
                        "--sweep_config",
                        sweep_path,
                        "--workers",
                        "0",
                    ],
                ),
                mock.patch.object(
                    hyperparameter_search_arrow.PARSER,
                    "error",
                    side_effect=SystemExit(2),
                ),
            ):
                with self.assertRaises(SystemExit):
                    hyperparameter_search_arrow.main()

    def test_workers_two_uses_parallel_path(self):
        """With --workers=2 and 2 grid points, executor receives 2 submit() calls and results are collected."""
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
                fut = concurrent.futures.Future()
                fut.set_result(
                    (run_index, {"best_val_loss": 0.5 - run_index * 0.1}, overrides)
                )
                submitted_futures.append(fut)
                return fut

        with tempfile.TemporaryDirectory() as temp_dir:
            sweep_path = os.path.join(temp_dir, "sweep.json")
            _write_json(
                sweep_path,
                _minimal_sweep_config(
                    os.path.join(temp_dir, "sweep_out"),
                    search_space={"model.transformer.dropout_rate": [0.0, 0.1]},
                ),
            )

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

    def test_workers_from_sweep_config_when_cli_omitted(self):
        """When --workers is not passed, workers from sweep config is used."""
        captured_max_workers = []

        class FakeExecutor:
            def __init__(self, max_workers=None, max_tasks_per_child=None, **kwargs):
                captured_max_workers.append(max_workers)

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def submit(self, fn, *args):
                run_index, overrides = args[0], args[1]
                fut = concurrent.futures.Future()
                fut.set_result((run_index, {"best_val_loss": 0.5}, overrides))
                return fut

            def as_completed(self, futures):
                return iter(futures)

        with tempfile.TemporaryDirectory() as temp_dir:
            sweep_path = os.path.join(temp_dir, "sweep.json")
            _write_json(
                sweep_path,
                _minimal_sweep_config(
                    os.path.join(temp_dir, "sweep_out"),
                    search_space={"model.transformer.dropout_rate": [0.0]},
                    extra={"workers": 3},
                ),
            )

            def make_executor(*args, **kwargs):
                return FakeExecutor(*args, **kwargs)

            with (
                mock.patch.object(
                    hyperparameter_search_arrow.futures,
                    "ProcessPoolExecutor",
                    side_effect=make_executor,
                ),
                mock.patch(
                    "sys.argv",
                    [
                        "hyperparameter_search_arrow",
                        "--sweep_config",
                        sweep_path,
                    ],
                ),
            ):
                exit_code = hyperparameter_search_arrow.main()
            self.assertEqual(exit_code, 0)
            self.assertEqual(
                captured_max_workers,
                [3],
                "ProcessPoolExecutor should be called with max_workers=3 from config",
            )

    def test_workers_defaults_to_one_when_omitted_from_cli_and_config(self):
        """When --workers is not passed and sweep config has no 'workers', default to 1."""
        captured_max_workers = []

        class FakeExecutor:
            def __init__(self, max_workers=None, max_tasks_per_child=None, **kwargs):
                captured_max_workers.append(max_workers)

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def submit(self, fn, *args):
                run_index, overrides = args[0], args[1]
                fut = concurrent.futures.Future()
                fut.set_result((run_index, {"best_val_loss": 0.5}, overrides))
                return fut

            def as_completed(self, futures):
                return iter(futures)

        with tempfile.TemporaryDirectory() as temp_dir:
            sweep_path = os.path.join(temp_dir, "sweep.json")
            _write_json(
                sweep_path,
                _minimal_sweep_config(
                    os.path.join(temp_dir, "sweep_out"),
                    search_space={"model.transformer.dropout_rate": [0.0]},
                ),
            )

            def make_executor(*args, **kwargs):
                return FakeExecutor(*args, **kwargs)

            with (
                mock.patch.object(
                    hyperparameter_search_arrow.futures,
                    "ProcessPoolExecutor",
                    side_effect=make_executor,
                ),
                mock.patch(
                    "sys.argv",
                    [
                        "hyperparameter_search_arrow",
                        "--sweep_config",
                        sweep_path,
                    ],
                ),
            ):
                exit_code = hyperparameter_search_arrow.main()
            self.assertEqual(exit_code, 0)
            self.assertEqual(
                captured_max_workers,
                [1],
                "ProcessPoolExecutor should be called with max_workers=1 when omitted from CLI and config",
            )

    def test_resume_with_invalid_workers_type_exits_with_validation_error(self):
        """When resuming, if sweep_config.json has non-integer workers (e.g. string), main exits with clear error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sweep_config = {
                **_minimal_inline_experiment_config(),
                "search_space": {"model.transformer.dropout_rate": [0.0]},
                "optimize": {"metric": "val_loss", "mode": "min"},
                "workers": "2",
            }
            with open(os.path.join(tmpdir, "sweep_config.json"), "w") as f:
                json.dump(sweep_config, f, indent=2)
            with open(os.path.join(tmpdir, "results.json"), "w") as f:
                json.dump([], f)
            with (
                mock.patch(
                    "sys.argv",
                    [
                        "hyperparameter_search_arrow",
                        "--resume_from",
                        os.path.abspath(tmpdir),
                    ],
                ),
                mock.patch.object(
                    hyperparameter_search_arrow.PARSER,
                    "error",
                    side_effect=SystemExit(2),
                ) as err_mock,
            ):
                with self.assertRaises(SystemExit):
                    hyperparameter_search_arrow.main()
            err_mock.assert_called_once()
            msg = err_mock.call_args[0][0]
            self.assertIn("workers", msg)
            self.assertIn("integer", msg)

    def test_new_best_printed_when_run_has_best_val_metric_so_far(self):
        """When a run has the best optimize metric seen so far, 'NEW BEST' is printed."""
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
                fut = concurrent.futures.Future()
                fut.set_result(
                    (
                        run_index,
                        {"best_val_main_loss": metrics_by_run[run_index]},
                        overrides,
                    )
                )
                submitted_futures.append(fut)
                return fut

        with tempfile.TemporaryDirectory() as temp_dir:
            sweep_path = os.path.join(temp_dir, "sweep.json")
            _write_json(
                sweep_path,
                _minimal_sweep_config(
                    os.path.join(temp_dir, "sweep_out"),
                    search_space={"model.transformer.dropout_rate": [0.0, 0.1, 0.2]},
                    optimize={"metric": "val_main_loss", "mode": "min"},
                ),
            )

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
                model, history = trainers.run_arrow_train_from_config(run_config)
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
        with tempfile.TemporaryDirectory() as temp_dir:
            sweep_path = os.path.join(temp_dir, "sweep.json")
            _write_json(
                sweep_path,
                _minimal_sweep_config(
                    os.path.join(temp_dir, "sweep_out"),
                    search_space={"model.transformer.dropout_rate": [0.0]},
                ),
            )

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
