"""Tests for scripts/train_arrow.py CLI and config resolution."""

import argparse
import os
import sys
import tempfile
import unittest
from unittest import mock

import tensorflow as tf

# Allow importing the script module (defer import so parse_args can be patched)
_SCRIPT_DIR = os.path.join(os.path.dirname(__file__), "..", "scripts")
_SCRIPT_DIR = os.path.abspath(_SCRIPT_DIR)
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)


def _make_args(
    config=None,
    train_data_dir=None,
    val_data_dir=None,
    model_output_dir=None,
    model_type=None,
    dropout_rate=None,
    **kwargs,
):
    """Build an argparse.Namespace with defaults; override with given kwargs."""
    defaults = {
        "config": None,
        "train_data_dir": None,
        "val_data_dir": None,
        "model_output_dir": None,
        "batch_size": None,
        "num_layers": None,
        "d_model": None,
        "num_heads": None,
        "ff_dim": None,
        "dropout_rate": None,
        "epochs": None,
        "take_count": None,
        "val_take_count": None,
        "callback_root_dir": None,
        "model_name": None,
        "snippet_half_frames": None,
        "chart_validity_aux_weight": None,
        "diversity_aux_weight": None,
        "warmup_epochs": None,
        "lr_peak": None,
        "lr_min": None,
        "model_type": None,
    }
    defaults.update(
        config=config,
        train_data_dir=train_data_dir,
        val_data_dir=val_data_dir,
        model_output_dir=model_output_dir,
        model_type=model_type,
        dropout_rate=dropout_rate,
        **kwargs,
    )
    return argparse.Namespace(**defaults)


def _run_train_arrow_main(args):
    """Import train_arrow with patched parse_args, then run main() with patched trainer. Returns (run_mock, model_config)."""
    if "train_arrow" in sys.modules:
        del sys.modules["train_arrow"]
    with mock.patch.object(argparse.ArgumentParser, "parse_args", return_value=args):
        import train_arrow  # noqa: E402
    with mock.patch("stepcovnet.trainers.run_arrow_train_from_config") as run_mock:
        train_arrow.main()
    _dataset_config, model_config, _run_config = run_mock.call_args[0]
    return run_mock, model_config


class TrainArrowModelTypeMlpTest(unittest.TestCase):
    """Test that --model_type mlp with transformer-style args initializes mlp correctly."""

    def test_model_type_mlp_with_dropout_rate_produces_valid_config(self):
        """With --model_type mlp and --dropout_rate, model_config has mlp set and buildable."""
        with tempfile.TemporaryDirectory() as tmpdir:
            args = _make_args(
                train_data_dir=tmpdir,
                val_data_dir=tmpdir,
                model_output_dir=tmpdir,
                model_type="mlp",
                dropout_rate=0.25,
            )
            run_mock, model_config = _run_train_arrow_main(args)
        run_mock.assert_called_once()
        self.assertEqual(model_config.model_type, "mlp")
        self.assertIsNotNone(model_config.mlp)
        self.assertEqual(model_config.mlp.dropout_rate, 0.25)
        # build_arrow_model_from_config should not raise
        from stepcovnet import models

        models.build_arrow_model_from_config(model_config)

    def test_model_type_transformer_with_dropout_rate_unchanged(self):
        """With --model_type transformer and --dropout_rate, transformer gets dropout."""
        with tempfile.TemporaryDirectory() as tmpdir:
            args = _make_args(
                train_data_dir=tmpdir,
                val_data_dir=tmpdir,
                model_output_dir=tmpdir,
                model_type="transformer",
                dropout_rate=0.3,
            )
            run_mock, model_config = _run_train_arrow_main(args)
        self.assertEqual(model_config.model_type, "transformer")
        self.assertIsNotNone(model_config.transformer)
        self.assertEqual(model_config.transformer.dropout_rate, 0.3)


def _run_train_arrow_main_with_run_config(args):
    """Run main() and return (run_mock, dataset_config, model_config, run_config)."""
    if "train_arrow" in sys.modules:
        del sys.modules["train_arrow"]
    with mock.patch.object(argparse.ArgumentParser, "parse_args", return_value=args):
        import train_arrow  # noqa: E402
    with mock.patch("stepcovnet.trainers.run_arrow_train_from_config") as run_mock:
        train_arrow.main()
    dataset_config, model_config, run_config = run_mock.call_args[0]
    return run_mock, dataset_config, model_config, run_config


class TrainArrowConfigFileAndOverridesTest(unittest.TestCase):
    """Test --config path and CLI overrides applied on top of config."""

    def test_config_file_with_model_type_override(self):
        """With --config, model_config comes from file; --model_type and --dropout_rate override."""
        from stepcovnet import config

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
                config=config_path,
                model_type="mlp",
                dropout_rate=0.2,
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
        """num_layers, d_model, num_heads, ff_dim, dropout_rate all applied when model_type is transformer."""
        with tempfile.TemporaryDirectory() as tmpdir:
            args = _make_args(
                train_data_dir=tmpdir,
                val_data_dir=tmpdir,
                model_output_dir=tmpdir,
                model_type="transformer",
                num_layers=2,
                d_model=64,
                num_heads=2,
                ff_dim=256,
                dropout_rate=0.1,
            )
            _run_mock, _dc, model_config, _rc = _run_train_arrow_main_with_run_config(
                args
            )
        self.assertEqual(model_config.transformer.num_layers, 2)
        self.assertEqual(model_config.transformer.d_model, 64)
        self.assertEqual(model_config.transformer.num_heads, 2)
        self.assertEqual(model_config.transformer.ff_dim, 256)
        self.assertEqual(model_config.transformer.dropout_rate, 0.1)

    def test_snippet_half_frames_override(self):
        """snippet_half_frames CLI override updates dataset and model config (no-config path)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            args = _make_args(
                train_data_dir=tmpdir,
                val_data_dir=tmpdir,
                model_output_dir=tmpdir,
                snippet_half_frames=5,
            )
            _run_mock, dataset_config, model_config, _rc = (
                _run_train_arrow_main_with_run_config(args)
            )
        self.assertEqual(dataset_config.snippet_half_frames, 5)
        self.assertEqual(model_config.snippet_half_frames, 5)

    def test_config_file_with_snippet_half_frames_override(self):
        """With --config, snippet_half_frames CLI override still updates dataset and model config."""
        from stepcovnet import config

        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = os.path.join(tmpdir, "c.json")
            cfg = config.ArrowExperimentConfig(
                dataset=config.ArrowDatasetConfig(
                    data_dir=tmpdir, val_data_dir=tmpdir, snippet_half_frames=0
                ),
                model=config.ArrowModelConfig.from_dict({"snippet_half_frames": 0}),
                run=config.ArrowRunConfig(
                    epoch=1, take_count=1, model_output_dir=tmpdir
                ),
            )
            cfg.to_json(json_path)
            args = _make_args(config=json_path, snippet_half_frames=3)
            _run_mock, dataset_config, model_config, _rc = (
                _run_train_arrow_main_with_run_config(args)
            )
        self.assertEqual(dataset_config.snippet_half_frames, 3)
        self.assertEqual(model_config.snippet_half_frames, 3)

    def test_run_config_cli_overrides(self):
        """epochs, take_count, val_take_count, model_output_dir, callback_root_dir, model_name, aux weights, lr applied."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = os.path.join(tmpdir, "out")
            args = _make_args(
                train_data_dir=tmpdir,
                val_data_dir=tmpdir,
                model_output_dir=out_dir,
                epochs=3,
                take_count=2,
                val_take_count=1,
                callback_root_dir=os.path.join(tmpdir, "callbacks"),
                model_name="my_model",
                chart_validity_aux_weight=0.4,
                diversity_aux_weight=0.2,
                warmup_epochs=1,
                lr_peak=2e-3,
                lr_min=1e-6,
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
        self.assertEqual(run_config.lr_peak, 2e-3)
        self.assertEqual(run_config.lr_min, 1e-6)


class TrainArrowValidationTest(unittest.TestCase):
    """Test validation and error paths in main()."""

    def test_main_errors_when_missing_dirs_without_config(self):
        """Without --config, missing train_data_dir/val_data_dir/model_output_dir causes PARSER.error."""
        args = _make_args(
            config=None, train_data_dir=None, val_data_dir=None, model_output_dir=None
        )
        if "train_arrow" in sys.modules:
            del sys.modules["train_arrow"]
        with mock.patch.object(
            argparse.ArgumentParser, "parse_args", return_value=args
        ):
            import train_arrow  # noqa: E402
        with self.assertRaises(SystemExit):
            train_arrow.main()

    def test_main_errors_when_config_has_empty_model_output_dir_and_no_override(self):
        """Config file with run.model_output_dir empty and no --model_output_dir override triggers PARSER.error."""
        from stepcovnet import config

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
            args = _make_args(config=json_path)
            args.model_output_dir = None
            if "train_arrow" in sys.modules:
                del sys.modules["train_arrow"]
            with mock.patch.object(
                argparse.ArgumentParser, "parse_args", return_value=args
            ):
                import train_arrow  # noqa: E402
            with self.assertRaises(SystemExit):
                train_arrow.main()

    def test_main_errors_when_config_has_empty_data_dir_and_no_override(self):
        """Config file with dataset.data_dir empty and no --train_data_dir override triggers PARSER.error."""
        from stepcovnet import config

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
            args = _make_args(config=json_path)
            args.train_data_dir = None
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
            args = _make_args(
                train_data_dir=tmpdir,
                val_data_dir=tmpdir,
                model_output_dir=tmpdir,
            )
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
