"""Tests for scripts/train_arrow.py CLI and config resolution."""

import argparse
import os
import sys
import tempfile
import unittest
from unittest import mock

import tensorflow as tf

from stepcovnet import config, models

# Allow importing the script module (defer import so parse_args can be patched)
_SCRIPT_DIR = os.path.join(os.path.dirname(__file__), "..", "scripts")
_SCRIPT_DIR = os.path.abspath(_SCRIPT_DIR)
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)


def _make_args(config, set_overrides=None):
    """Build an argparse.Namespace with --config and optional --set overrides."""
    return argparse.Namespace(config=config, set=set_overrides or [])


def _minimal_config_path(
    tmpdir, data_dir=None, val_data_dir=None, model_output_dir=None, **run_overrides
):
    """Write a minimal ArrowExperimentConfig to tmpdir and return its path."""
    data_dir = data_dir or tmpdir
    val_data_dir = val_data_dir or tmpdir
    model_output_dir = model_output_dir or tmpdir
    path = os.path.join(tmpdir, "arrow.json")
    cfg = config.ArrowExperimentConfig(
        dataset=config.ArrowDatasetConfig(
            data_dir=data_dir,
            val_data_dir=val_data_dir,
            batch_size=1,
            snippet_half_frames=0,
        ),
        model=config.ArrowModelConfig.from_dict(
            {"model_type": "transformer", "transformer": {}}
        ),
        run=config.ArrowRunConfig(
            epoch=1,
            take_count=1,
            model_output_dir=model_output_dir,
            **run_overrides,
        ),
    )
    cfg.to_json(path)
    return path


def _run_train_arrow_main(args):
    """Import train_arrow with patched parse_args, then run main() with patched trainer. Returns (run_mock, model_config)."""
    if "train_arrow" in sys.modules:
        del sys.modules["train_arrow"]
    with mock.patch.object(argparse.ArgumentParser, "parse_args", return_value=args):
        import train_arrow  # noqa: E402
    with mock.patch("stepcovnet.trainers.run_arrow_train_from_config") as run_mock:
        train_arrow.main()
    (experiment_config,) = run_mock.call_args[0]
    return run_mock, experiment_config.model


def _run_train_arrow_main_with_run_config(args):
    """Run main() and return (run_mock, dataset_config, model_config, run_config)."""
    if "train_arrow" in sys.modules:
        del sys.modules["train_arrow"]
    with mock.patch.object(argparse.ArgumentParser, "parse_args", return_value=args):
        import train_arrow  # noqa: E402
    with mock.patch("stepcovnet.trainers.run_arrow_train_from_config") as run_mock:
        train_arrow.main()
    (experiment_config,) = run_mock.call_args[0]
    return (
        run_mock,
        experiment_config.dataset,
        experiment_config.model,
        experiment_config.run,
    )


class ApplyOverridesFromCliTest(unittest.TestCase):
    """Test apply_overrides_from_cli helper (coercion and dotted paths)."""

    def test_coercion_int_float_bool_str(self):
        """String overrides are coerced to int, float, bool, or left as str."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _minimal_config_path(tmpdir)
            args = _make_args(path, [])
            if "train_arrow" in sys.modules:
                del sys.modules["train_arrow"]
            with mock.patch.object(
                argparse.ArgumentParser, "parse_args", return_value=args
            ):
                import train_arrow  # noqa: E402
            base = config.ArrowExperimentConfig.from_json(path)
            result = train_arrow.apply_overrides_from_cli(
                base,
                ["run.epoch=5", "run.lr_peak=0.002", "run.seed=42"],
            )
        self.assertEqual(result.run.epoch, 5)
        self.assertEqual(result.run.lr_peak, 0.002)
        self.assertEqual(result.run.seed, 42)

    def test_coercion_float_and_bool(self):
        """Float and bool overrides are applied."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _minimal_config_path(tmpdir)
            args = _make_args(path, [])
            if "train_arrow" in sys.modules:
                del sys.modules["train_arrow"]
            with mock.patch.object(
                argparse.ArgumentParser, "parse_args", return_value=args
            ):
                import train_arrow  # noqa: E402
            base = config.ArrowExperimentConfig.from_json(path)
            result = train_arrow.apply_overrides_from_cli(
                base,
                ["run.chart_validity_aux_weight=0.4", "run.diversity_aux_weight=0.2"],
            )
        self.assertEqual(result.run.chart_validity_aux_weight, 0.4)
        self.assertEqual(result.run.diversity_aux_weight, 0.2)

    def test_nested_model_path(self):
        """Nested keys like model.lstm.units are set correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _minimal_config_path(tmpdir)
            args = _make_args(path, [])
            if "train_arrow" in sys.modules:
                del sys.modules["train_arrow"]
            with mock.patch.object(
                argparse.ArgumentParser, "parse_args", return_value=args
            ):
                import train_arrow  # noqa: E402
            base = config.ArrowExperimentConfig.from_json(path)
            result = train_arrow.apply_overrides_from_cli(
                base,
                [
                    "model.model_type=lstm",
                    "model.lstm.units=64",
                    "model.lstm.num_layers=2",
                ],
            )
        self.assertEqual(result.model.model_type, "lstm")
        self.assertIsNotNone(result.model.lstm)
        assert result.model.lstm is not None
        self.assertEqual(result.model.lstm.units, 64)
        self.assertEqual(result.model.lstm.num_layers, 2)

    def test_empty_overrides_returns_base(self):
        """Empty overrides list returns same config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _minimal_config_path(tmpdir)
            args = _make_args(path, [])
            if "train_arrow" in sys.modules:
                del sys.modules["train_arrow"]
            with mock.patch.object(
                argparse.ArgumentParser, "parse_args", return_value=args
            ):
                import train_arrow  # noqa: E402
            base = config.ArrowExperimentConfig.from_json(path)
            result = train_arrow.apply_overrides_from_cli(base, [])
        self.assertIs(result, base)

    def test_coercion_bool_and_skip_malformed_overrides(self):
        """Bool coercion (true/false) and malformed entries are skipped without error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _minimal_config_path(tmpdir)
            args = _make_args(path, [])
            if "train_arrow" in sys.modules:
                del sys.modules["train_arrow"]
            with mock.patch.object(
                argparse.ArgumentParser, "parse_args", return_value=args
            ):
                import train_arrow  # noqa: E402
            base = config.ArrowExperimentConfig.from_json(path)
            # Valid: run.show_model_summary=false (bool); run.epoch=2 (int).
            # Skipped: no "=", no ".", or prefix not dataset/model/run.
            result = train_arrow.apply_overrides_from_cli(
                base,
                [
                    "run.show_model_summary=false",
                    "run.epoch=2",
                    "noequals",
                    "run_nodot=3",
                    "other.foo=1",
                ],
            )
        self.assertFalse(result.run.show_model_summary)
        self.assertEqual(result.run.epoch, 2)

    def test_coercion_bool_true(self):
        """Bool override run.show_model_summary=true coerces to True (covers _coerce_value true branch)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _minimal_config_path(tmpdir)
            args = _make_args(path, [])
            if "train_arrow" in sys.modules:
                del sys.modules["train_arrow"]
            with mock.patch.object(
                argparse.ArgumentParser, "parse_args", return_value=args
            ):
                import train_arrow  # noqa: E402
            base = config.ArrowExperimentConfig.from_json(path)
            result = train_arrow.apply_overrides_from_cli(
                base,
                ["run.show_model_summary=true"],
            )
        self.assertTrue(result.run.show_model_summary)

    def test_empty_key_path_components_skipped(self):
        """Overrides with empty path components (e.g. dataset.=value) are skipped; no TypeError on from_dict."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _minimal_config_path(tmpdir)
            args = _make_args(path, [])
            if "train_arrow" in sys.modules:
                del sys.modules["train_arrow"]
            with mock.patch.object(
                argparse.ArgumentParser, "parse_args", return_value=args
            ):
                import train_arrow  # noqa: E402
            base = config.ArrowExperimentConfig.from_json(path)
            result = train_arrow.apply_overrides_from_cli(
                base,
                [
                    "run.epoch=3",
                    "dataset.=value",
                    "run..epoch=99",
                    "model.lstm.=0",
                ],
            )
        self.assertEqual(result.run.epoch, 3)

    def test_set_nested_rejects_empty_path_components(self):
        """_set_nested does not set keys when path has empty components (no empty string key)."""
        if "train_arrow" in sys.modules:
            del sys.modules["train_arrow"]
        with mock.patch.object(
            argparse.ArgumentParser, "parse_args", return_value=_make_args(None)
        ):
            import train_arrow  # noqa: E402
        d = {}
        train_arrow._set_nested(d, "", "x")
        self.assertNotIn("", d)
        d2 = {"a": {}}
        train_arrow._set_nested(d2["a"], "..b", "y")
        self.assertNotIn("", d2["a"])

    def test_set_nested_raises_when_intermediate_is_leaf(self):
        """_set_nested raises ValueError when path goes through a leaf (e.g. run.epoch.foo=bar)."""
        if "train_arrow" in sys.modules:
            del sys.modules["train_arrow"]
        with mock.patch.object(
            argparse.ArgumentParser, "parse_args", return_value=_make_args(None)
        ):
            import train_arrow  # noqa: E402
        d = {"run": {"epoch": 10}}
        with self.assertRaises(ValueError) as ctx:
            train_arrow._set_nested(d["run"], "epoch.foo", "bar")
        self.assertIn("epoch.foo", str(ctx.exception))
        self.assertIn("epoch", str(ctx.exception))
        self.assertIn("not a nested object", str(ctx.exception))

    def test_set_nested_promotes_none_to_dict_when_setting_nested_key(self):
        """_set_nested promotes None to {} when setting a nested key so param blocks can be created."""
        if "train_arrow" in sys.modules:
            del sys.modules["train_arrow"]
        with mock.patch.object(
            argparse.ArgumentParser, "parse_args", return_value=_make_args(None)
        ):
            import train_arrow  # noqa: E402
        d = {"run": {"epoch": None}}
        train_arrow._set_nested(d["run"], "epoch.foo", "bar")
        self.assertIsInstance(d["run"]["epoch"], dict)
        assert d["run"]["epoch"] is not None
        self.assertEqual(d["run"]["epoch"]["foo"], "bar")

    def test_apply_overrides_raises_when_nested_path_targets_leaf(self):
        """apply_overrides_from_cli raises ValueError for run.epoch.foo=bar (epoch is int)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _minimal_config_path(tmpdir)
            args = _make_args(path, [])
            if "train_arrow" in sys.modules:
                del sys.modules["train_arrow"]
            with mock.patch.object(
                argparse.ArgumentParser, "parse_args", return_value=args
            ):
                import train_arrow  # noqa: E402
            base = config.ArrowExperimentConfig.from_json(path)
            with self.assertRaises(ValueError) as ctx:
                train_arrow.apply_overrides_from_cli(base, ["run.epoch.foo=bar"])
            self.assertIn("epoch", str(ctx.exception))
            self.assertIn("not a nested object", str(ctx.exception))


class TrainArrowModelTypeMlpTest(unittest.TestCase):
    """Test model_type=mlp via --set and MLP block initialization."""

    def test_model_type_mlp_without_dropout_rate_uses_defaults(self):
        """With --set model.model_type=mlp, MLP block uses default dropout."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _minimal_config_path(tmpdir)
            args = _make_args(path, ["model.model_type=mlp"])
            run_mock, model_config = _run_train_arrow_main(args)
        run_mock.assert_called_once()
        self.assertEqual(model_config.model_type, "mlp")
        self.assertIsNotNone(model_config.mlp)
        self.assertEqual(model_config.mlp.dropout_rate, 0.0)
        models.build_arrow_model_from_config(
            model_config,
            models.ArrowInputOptions(),
            models.ArrowOutputOptions(),
        )

    def test_model_type_mlp_with_dropout_rate_produces_valid_config(self):
        """With --set model.model_type=mlp and model.mlp.dropout_rate=0.25, model is buildable."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _minimal_config_path(tmpdir)
            args = _make_args(
                path, ["model.model_type=mlp", "model.mlp.dropout_rate=0.25"]
            )
            run_mock, model_config = _run_train_arrow_main(args)
        run_mock.assert_called_once()
        self.assertEqual(model_config.model_type, "mlp")
        self.assertIsNotNone(model_config.mlp)
        self.assertEqual(model_config.mlp.dropout_rate, 0.25)
        models.build_arrow_model_from_config(
            model_config,
            models.ArrowInputOptions(),
            models.ArrowOutputOptions(),
        )

    def test_model_type_transformer_with_dropout_rate_unchanged(self):
        """With model_type=transformer and model.transformer.dropout_rate=0.3, transformer gets dropout."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _minimal_config_path(tmpdir)
            args = _make_args(
                path,
                ["model.model_type=transformer", "model.transformer.dropout_rate=0.3"],
            )
            run_mock, model_config = _run_train_arrow_main(args)
        self.assertEqual(model_config.model_type, "transformer")
        self.assertIsNotNone(model_config.transformer)
        self.assertEqual(model_config.transformer.dropout_rate, 0.3)


class TrainArrowConfigFileAndOverridesTest(unittest.TestCase):
    """Test --config and --set overrides applied on top of config."""

    def test_config_file_with_model_type_override(self):
        """Config has transformer; --set model.model_type=mlp and model.mlp.dropout_rate=0.2 override."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "arrow.json")
            experiment = config.ArrowExperimentConfig(
                dataset=config.ArrowDatasetConfig(
                    data_dir=os.path.join(tmpdir, "train"),
                    val_data_dir=os.path.join(tmpdir, "val"),
                    snippet_half_frames=0,
                ),
                model=config.ArrowModelConfig.from_dict(
                    {"model_type": "transformer", "transformer": {"num_layers": 2}}
                ),
                run=config.ArrowRunConfig(
                    epoch=5,
                    take_count=1,
                    model_output_dir=os.path.join(tmpdir, "out"),
                ),
            )
            experiment.to_json(config_path)
            args = _make_args(
                config_path, ["model.model_type=mlp", "model.mlp.dropout_rate=0.2"]
            )
            run_mock, _dataset_config, model_config, _run_config = (
                _run_train_arrow_main_with_run_config(args)
            )
        run_mock.assert_called_once()
        self.assertEqual(model_config.model_type, "mlp")
        self.assertIsNotNone(model_config.mlp)
        self.assertIsNone(
            model_config.transformer,
            "inactive transformer block must be cleared when overriding to mlp",
        )
        self.assertEqual(model_config.mlp.dropout_rate, 0.2)

    def test_all_transformer_cli_overrides_applied(self):
        """--set model.transformer.* applies num_layers, d_model, num_heads, ff_dim, dropout_rate."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _minimal_config_path(tmpdir)
            args = _make_args(
                path,
                [
                    "model.model_type=transformer",
                    "model.transformer.num_layers=2",
                    "model.transformer.d_model=64",
                    "model.transformer.num_heads=2",
                    "model.transformer.ff_dim=256",
                    "model.transformer.dropout_rate=0.1",
                ],
            )
            _run_mock, _dc, model_config, _rc = _run_train_arrow_main_with_run_config(
                args
            )
        self.assertEqual(model_config.transformer.num_layers, 2)
        self.assertEqual(model_config.transformer.d_model, 64)
        self.assertEqual(model_config.transformer.num_heads, 2)
        self.assertEqual(model_config.transformer.ff_dim, 256)
        self.assertEqual(model_config.transformer.dropout_rate, 0.1)

    def test_model_type_lstm_with_overrides_produces_valid_config(self):
        """--set model.model_type=lstm and model.lstm.* produces buildable config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _minimal_config_path(tmpdir)
            args = _make_args(
                path,
                [
                    "model.model_type=lstm",
                    "model.lstm.units=64",
                    "model.lstm.num_layers=2",
                    "model.lstm.dropout_rate=0.1",
                ],
            )
            run_mock, model_config = _run_train_arrow_main(args)
        run_mock.assert_called_once()
        self.assertEqual(model_config.model_type, "lstm")
        self.assertIsNotNone(model_config.lstm)
        self.assertEqual(model_config.lstm.units, 64)
        self.assertEqual(model_config.lstm.num_layers, 2)
        self.assertEqual(model_config.lstm.dropout_rate, 0.1)
        models.build_arrow_model_from_config(
            model_config,
            models.ArrowInputOptions(),
            models.ArrowOutputOptions(),
        )

    def test_lstm_bidirectional_override_builds(self):
        """--set model.lstm.bidirectional=true results in bidirectional True and model builds."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _minimal_config_path(tmpdir)
            args = _make_args(
                path,
                [
                    "model.model_type=lstm",
                    "model.lstm.units=64",
                    "model.lstm.num_layers=1",
                    "model.lstm.bidirectional=true",
                ],
            )
            run_mock, model_config = _run_train_arrow_main(args)
        run_mock.assert_called_once()
        self.assertEqual(model_config.model_type, "lstm")
        self.assertIsNotNone(model_config.lstm)
        assert model_config.lstm is not None
        self.assertTrue(model_config.lstm.bidirectional)
        models.build_arrow_model_from_config(
            model_config,
            models.ArrowInputOptions(),
            models.ArrowOutputOptions(),
        )

    def test_model_type_gru_with_overrides_produces_valid_config(self):
        """--set model.model_type=gru and model.gru.* produces buildable config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _minimal_config_path(tmpdir)
            args = _make_args(
                path,
                [
                    "model.model_type=gru",
                    "model.gru.units=64",
                    "model.gru.num_layers=2",
                    "model.gru.dropout_rate=0.1",
                ],
            )
            run_mock, model_config = _run_train_arrow_main(args)
        run_mock.assert_called_once()
        self.assertEqual(model_config.model_type, "gru")
        self.assertIsNotNone(model_config.gru)
        self.assertEqual(model_config.gru.units, 64)
        self.assertEqual(model_config.gru.num_layers, 2)
        self.assertEqual(model_config.gru.dropout_rate, 0.1)
        models.build_arrow_model_from_config(
            model_config,
            models.ArrowInputOptions(),
            models.ArrowOutputOptions(),
        )

    def test_gru_bidirectional_override_builds(self):
        """--set model.gru.bidirectional=true results in bidirectional True and model builds."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _minimal_config_path(tmpdir)
            args = _make_args(
                path,
                [
                    "model.model_type=gru",
                    "model.gru.units=64",
                    "model.gru.num_layers=1",
                    "model.gru.bidirectional=true",
                ],
            )
            run_mock, model_config = _run_train_arrow_main(args)
        run_mock.assert_called_once()
        self.assertEqual(model_config.model_type, "gru")
        self.assertIsNotNone(model_config.gru)
        assert model_config.gru is not None
        self.assertTrue(model_config.gru.bidirectional)
        models.build_arrow_model_from_config(
            model_config,
            models.ArrowInputOptions(),
            models.ArrowOutputOptions(),
        )

    def test_snippet_half_frames_override(self):
        """--set dataset.snippet_half_frames updates dataset config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _minimal_config_path(tmpdir)
            args = _make_args(path, [])
            if "train_arrow" in sys.modules:
                del sys.modules["train_arrow"]
            with mock.patch.object(
                argparse.ArgumentParser, "parse_args", return_value=args
            ):
                import train_arrow  # noqa: E402
            base = config.ArrowExperimentConfig.from_json(path)
            result = train_arrow.apply_overrides_from_cli(
                base,
                ["dataset.snippet_half_frames=5"],
            )
        self.assertEqual(result.dataset.snippet_half_frames, 5)

    def test_config_file_with_snippet_half_frames_override(self):
        """With --config, --set dataset.snippet_half_frames updates dataset config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = os.path.join(tmpdir, "c.json")
            cfg = config.ArrowExperimentConfig(
                dataset=config.ArrowDatasetConfig(
                    data_dir=tmpdir, val_data_dir=tmpdir, snippet_half_frames=0
                ),
                model=config.ArrowModelConfig(),
                run=config.ArrowRunConfig(
                    epoch=1, take_count=1, model_output_dir=tmpdir
                ),
            )
            cfg.to_json(json_path)
            args = _make_args(json_path, ["dataset.snippet_half_frames=3"])
            _run_mock, dataset_config, model_config, _rc = (
                _run_train_arrow_main_with_run_config(args)
            )
        self.assertEqual(dataset_config.snippet_half_frames, 3)

    def test_run_config_cli_overrides(self):
        """--set run.* applies epoch, take_count, val_take_count, paths, aux weights, lr."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = os.path.join(tmpdir, "out")
            path = _minimal_config_path(tmpdir, model_output_dir=out_dir)
            args = _make_args(
                path,
                [
                    "run.epoch=3",
                    "run.take_count=2",
                    "run.val_take_count=1",
                    "run.model_output_dir=" + out_dir,
                    "run.callback_root_dir=" + os.path.join(tmpdir, "callbacks"),
                    "run.model_name=my_model",
                    "run.chart_validity_aux_weight=0.4",
                    "run.diversity_aux_weight=0.2",
                    "run.warmup_epochs=1",
                    "run.lr_peak=0.002",
                    "run.lr_min=0.000001",
                ],
            )
            _run_mock, _dc, _mc, run_config = _run_train_arrow_main_with_run_config(
                args
            )
        self.assertEqual(run_config.epoch, 3)
        self.assertEqual(run_config.take_count, 2)
        self.assertEqual(run_config.val_take_count, 1)
        self.assertEqual(run_config.model_output_dir, out_dir)
        self.assertEqual(
            run_config.callback_root_dir, os.path.join(tmpdir, "callbacks")
        )
        self.assertEqual(run_config.model_name, "my_model")
        self.assertEqual(run_config.chart_validity_aux_weight, 0.4)
        self.assertEqual(run_config.diversity_aux_weight, 0.2)
        self.assertEqual(run_config.warmup_epochs, 1)
        self.assertEqual(run_config.lr_peak, 0.002)
        self.assertEqual(run_config.lr_min, 1e-6)


class TrainArrowValidationTest(unittest.TestCase):
    """Test validation and error paths in main()."""

    def test_main_errors_when_config_missing(self):
        """Missing --config causes PARSER.error."""
        args = _make_args(None)
        if "train_arrow" in sys.modules:
            del sys.modules["train_arrow"]
        with mock.patch.object(
            argparse.ArgumentParser, "parse_args", return_value=args
        ):
            import train_arrow  # noqa: E402
        with self.assertRaises(SystemExit):
            train_arrow.main()

    def test_main_errors_when_config_has_empty_model_output_dir_and_no_override(self):
        """Config with run.model_output_dir empty and no --set override triggers PARSER.error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = os.path.join(tmpdir, "c.json")
            cfg = config.ArrowExperimentConfig(
                dataset=config.ArrowDatasetConfig(
                    data_dir=tmpdir, val_data_dir=tmpdir, snippet_half_frames=0
                ),
                model=config.ArrowModelConfig.from_dict({}),
                run=config.ArrowRunConfig(epoch=1, take_count=1, model_output_dir=""),
            )
            cfg.to_json(json_path)
            args = _make_args(json_path)
            if "train_arrow" in sys.modules:
                del sys.modules["train_arrow"]
            with mock.patch.object(
                argparse.ArgumentParser, "parse_args", return_value=args
            ):
                import train_arrow  # noqa: E402
            with self.assertRaises(SystemExit):
                train_arrow.main()

    def test_main_errors_when_config_has_empty_data_dir_and_no_override(self):
        """Config with dataset.data_dir empty and no --set override triggers PARSER.error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = os.path.join(tmpdir, "c.json")
            cfg = config.ArrowExperimentConfig(
                dataset=config.ArrowDatasetConfig(
                    data_dir="", val_data_dir=tmpdir, snippet_half_frames=0
                ),
                model=config.ArrowModelConfig.from_dict({}),
                run=config.ArrowRunConfig(
                    epoch=1, take_count=1, model_output_dir=tmpdir
                ),
            )
            cfg.to_json(json_path)
            args = _make_args(json_path)
            if "train_arrow" in sys.modules:
                del sys.modules["train_arrow"]
            with mock.patch.object(
                argparse.ArgumentParser, "parse_args", return_value=args
            ):
                import train_arrow  # noqa: E402
            with self.assertRaises(SystemExit):
                train_arrow.main()


class TrainArrowGpuBlockTest(unittest.TestCase):
    """Test the module-level GPU block (mixed precision + XLA) is executed when GPU is present."""

    def test_gpu_block_runs_when_list_physical_devices_returns_non_empty(self):
        """When tf.config.list_physical_devices('GPU') returns a device, the GPU block runs."""
        if "train_arrow" in sys.modules:
            del sys.modules["train_arrow"]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _minimal_config_path(tmpdir)
            args = _make_args(path)
            with (
                mock.patch.object(
                    argparse.ArgumentParser, "parse_args", return_value=args
                ),
                mock.patch.object(
                    tf.config,
                    "list_physical_devices",
                    return_value=["GPU:0"],
                ),
                mock.patch.object(tf.config.optimizer, "set_jit"),
                mock.patch("keras.mixed_precision.set_global_policy"),
            ):
                import train_arrow  # noqa: E402
        self.assertTrue(hasattr(train_arrow, "main"))
