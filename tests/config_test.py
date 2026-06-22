import json
import pathlib
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

    def test_feature_source_mert_from_dict(self):
        data = {
            "data_dir": "data/train",
            "val_data_dir": "data/val",
            "feature_source": "mert",
            "mert_features_dir": "data/mert",
        }
        cfg = config.OnsetDatasetConfig.from_dict(data)
        self.assertEqual(cfg.feature_source, config.FeatureSource.MERT)
        self.assertEqual(cfg.mert_features_dir, "data/mert")
        self.assertEqual(cfg.as_dict()["feature_source"], "mert")

    def test_feature_source_waveform_from_dict(self):
        data = {
            "data_dir": "data/train",
            "val_data_dir": "data/val",
            "feature_source": "waveform",
        }
        cfg = config.OnsetDatasetConfig.from_dict(data)
        self.assertEqual(cfg.feature_source, config.FeatureSource.WAVEFORM)

    def test_max_train_songs_default_minus_one(self):
        cfg = config.OnsetDatasetConfig(data_dir="data/train", val_data_dir="data/val")
        self.assertEqual(cfg.max_train_songs, -1)

    def test_max_train_songs_from_dict(self):
        data = {
            "data_dir": "data/train",
            "val_data_dir": "data/val",
            "max_train_songs": 20,
        }
        cfg = config.OnsetDatasetConfig.from_dict(data)
        self.assertEqual(cfg.max_train_songs, 20)
        self.assertEqual(cfg.as_dict()["max_train_songs"], 20)

    def test_max_train_songs_zero_raises(self):
        with self.assertRaises(ValueError) as ctx:
            config.OnsetDatasetConfig(
                data_dir="data/train",
                val_data_dir="data/val",
                max_train_songs=0,
            )
        self.assertIn("max_train_songs", str(ctx.exception))

    def test_resolve_onset_input_features_waveform(self):
        dataset_cfg = config.OnsetDatasetConfig(
            data_dir="data/train",
            val_data_dir="data/val",
            feature_source=config.FeatureSource.WAVEFORM,
        )
        model_cfg = config.OnsetModelConfig(waveform_frontend_filters=48)
        self.assertEqual(
            config.resolve_onset_input_features(dataset_cfg, model_cfg),
            48,
        )


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

    def test_round_trip_new_params_interval_encoding_step_index_beat_phase_aux(self):
        """Round-trip and validation for interval_encoding, use_step_index, use_beat_phase, use_aux_interval_target."""
        data = {
            "data_dir": "d",
            "val_data_dir": "v",
            "interval_encoding": "log",
            "use_step_index": True,
            "use_beat_phase": True,
            "use_aux_interval_target": True,
        }
        cfg = config.ArrowDatasetConfig.from_dict(data)
        self.assertEqual(cfg.interval_encoding, config.IntervalEncoding.LOG)
        self.assertTrue(cfg.use_step_index)
        self.assertTrue(cfg.use_beat_phase)
        self.assertTrue(cfg.use_aux_interval_target)
        d = cfg.as_dict()
        self.assertEqual(d["interval_encoding"], "log")
        self.assertEqual(d["use_step_index"], True)
        self.assertEqual(d["use_beat_phase"], True)
        self.assertEqual(d["use_aux_interval_target"], True)
        cfg2 = config.ArrowDatasetConfig.from_dict(d)
        self.assertEqual(cfg2.interval_encoding, cfg.interval_encoding)
        self.assertEqual(cfg2.use_step_index, cfg.use_step_index)
        self.assertEqual(cfg2.use_beat_phase, cfg.use_beat_phase)
        self.assertEqual(cfg2.use_aux_interval_target, cfg.use_aux_interval_target)

    def test_interval_encoding_enum_values_and_invalid_raises(self):
        """IntervalEncoding has DEFAULT, LOG, MULTI; from_dict with invalid value raises."""
        self.assertEqual(config.IntervalEncoding.DEFAULT.value, "default")
        self.assertEqual(config.IntervalEncoding.LOG.value, "log")
        self.assertEqual(config.IntervalEncoding.MULTI.value, "multi")
        with self.assertRaises(ValueError):
            config.ArrowDatasetConfig.from_dict(
                {"data_dir": "d", "val_data_dir": "v", "interval_encoding": "invalid"}
            )

    def test_timing_jitter_sigma_default_and_round_trip(self):
        """timing_jitter_sigma defaults to 0 and round-trips in as_dict/from_dict."""
        cfg = config.ArrowDatasetConfig(data_dir="d", val_data_dir="v")
        self.assertEqual(cfg.timing_jitter_sigma, 0.0)
        cfg_jitter = config.ArrowDatasetConfig(
            data_dir="d", val_data_dir="v", timing_jitter_sigma=0.02
        )
        d = cfg_jitter.as_dict()
        self.assertEqual(d["timing_jitter_sigma"], 0.02)
        loaded = config.ArrowDatasetConfig.from_dict(d)
        self.assertEqual(loaded.timing_jitter_sigma, 0.02)

    def test_get_experiment_name_parts_timing_jitter(self):
        """get_experiment_name_parts returns timing_jitter token when sigma > 0, empty when 0."""
        cfg_off = config.ArrowDatasetConfig(data_dir="d", val_data_dir="v")
        self.assertEqual(cfg_off.get_experiment_name_parts(), [])
        cfg_on = config.ArrowDatasetConfig(
            data_dir="d", val_data_dir="v", timing_jitter_sigma=0.02
        )
        parts = cfg_on.get_experiment_name_parts()
        self.assertEqual(len(parts), 1)
        self.assertIn("timing_jitter", parts[0])
        self.assertIn("0_02", parts[0])

    def test_get_experiment_name_parts_snippet_half_frames(self):
        cfg = config.ArrowDatasetConfig(
            data_dir="d", val_data_dir="v", snippet_half_frames=5
        )
        parts = cfg.get_experiment_name_parts()
        self.assertEqual(len(parts), 1)
        self.assertIn("snippets_5", parts[0])

    def test_get_experiment_name_parts_use_interval(self):
        cfg = config.ArrowDatasetConfig(
            data_dir="d",
            val_data_dir="v",
            use_interval=True,
            interval_encoding=config.IntervalEncoding.LOG,
        )
        parts = cfg.get_experiment_name_parts()
        self.assertEqual(len(parts), 1)
        self.assertIn("interval_log", parts[0])

    def test_get_experiment_name_parts_use_step_index(self):
        cfg = config.ArrowDatasetConfig(
            data_dir="d", val_data_dir="v", use_step_index=True
        )
        parts = cfg.get_experiment_name_parts()
        self.assertEqual(len(parts), 1)
        self.assertIn("step_index", parts[0])

    def test_get_experiment_name_parts_use_beat_phase(self):
        cfg = config.ArrowDatasetConfig(
            data_dir="d", val_data_dir="v", use_beat_phase=True
        )
        parts = cfg.get_experiment_name_parts()
        self.assertEqual(len(parts), 1)
        self.assertIn("beat_phase", parts[0])


class OnsetModelConfigTest(unittest.TestCase):
    def test_onset_architecture_defaults_to_unet_wavenet(self):
        cfg = config.OnsetModelConfig()
        self.assertEqual(
            cfg.onset_architecture,
            config.OnsetArchitecture.UNET_WAVENET,
        )

    def test_onset_architecture_roundtrip(self):
        cfg = config.OnsetModelConfig(
            onset_architecture=config.OnsetArchitecture.TCN,
            tcn_blocks=3,
        )
        restored = config.OnsetModelConfig.from_dict(cfg.as_dict())
        self.assertEqual(restored.onset_architecture, config.OnsetArchitecture.TCN)
        self.assertEqual(restored.tcn_blocks, 3)

    def test_transformer_rejects_indivisible_heads(self):
        with self.assertRaises(ValueError):
            config.OnsetModelConfig(
                onset_architecture=config.OnsetArchitecture.TRANSFORMER,
                initial_filters=30,
                transformer_heads=4,
            )

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
        """Test creating config with default values (via from_dict for full defaults)."""
        cfg = config.ArrowModelConfig.from_dict({})
        self.assertEqual(cfg.model_type, "transformer")
        self.assertIsNotNone(cfg.transformer)
        assert cfg.transformer is not None
        self.assertEqual(cfg.transformer.num_layers, 1)
        self.assertEqual(cfg.transformer.d_model, 128)
        self.assertEqual(cfg.transformer.num_heads, 4)
        self.assertEqual(cfg.transformer.ff_dim, 512)
        self.assertEqual(cfg.transformer.dropout_rate, 0.0)

    def test_as_dict(self):
        """Test converting config to dictionary (nested shape)."""
        cfg = config.ArrowModelConfig.from_dict(
            {
                "model_type": "transformer",
                "transformer": {"num_layers": 2, "d_model": 256},
            }
        )
        d = cfg.as_dict()
        self.assertEqual(d["model_type"], "transformer")
        self.assertEqual(d["transformer"]["num_layers"], 2)
        self.assertEqual(d["transformer"]["d_model"], 256)

    def test_from_dict_with_invalid_attribute_raises_error(self):
        with self.assertRaises(ValueError) as ctx:
            config.ArrowModelConfig.from_dict({"invalid_attribute": 5})
        self.assertIn("invalid_attribute", str(ctx.exception))
        self.assertIn("Invalid keys", str(ctx.exception))

    def test_from_dict_invalid_model_type_raises_error(self):
        with self.assertRaises(ValueError) as ctx:
            config.ArrowModelConfig.from_dict({"model_type": "unknown_arch"})
        self.assertIn("unknown_arch", str(ctx.exception))
        self.assertIn("Invalid model_type", str(ctx.exception))

    def test_param_blocks_derived_from_config_class(self):
        """Param blocks for from_dict are derived from ArrowModelConfig fields.

        Adding a new param (e.g. resnet: ResNetArrowParams | None) to the config
        class is sufficient; no manual update to from_dict or a separate map is needed.
        """
        blocks = config._arrow_model_param_blocks(config.ArrowModelConfig)
        expected = {"transformer", "mlp", "lstm", "gru", "tcn", "cnn1d"}
        self.assertEqual(set(blocks.keys()), expected)
        self.assertIs(blocks["transformer"], config.TransformerArrowParams)
        self.assertIs(blocks["mlp"], config.MLPArrowParams)
        self.assertIs(blocks["cnn1d"], config.CNN1DArrowParams)

    def test_from_dict_nested_mlp(self):
        """Test creating config with model_type mlp and mlp block."""
        data = {
            "model_type": "mlp",
            "mlp": {"hidden_dims": [128, 64], "dropout_rate": 0.1},
        }
        cfg = config.ArrowModelConfig.from_dict(data)
        self.assertEqual(cfg.model_type, "mlp")
        self.assertIsNotNone(cfg.mlp)
        assert cfg.mlp is not None
        self.assertEqual(cfg.mlp.hidden_dims, [128, 64])
        self.assertEqual(cfg.mlp.dropout_rate, 0.1)

    def test_from_dict_mlp_ignores_nested_transformer_block(self):
        """With model_type=mlp, a nested 'transformer' key must not create a transformer block."""
        data = {
            "model_type": "mlp",
            "transformer": {"num_layers": 2, "d_model": 64},
            "mlp": {"hidden_dims": [128, 64], "dropout_rate": 0.1},
        }
        cfg = config.ArrowModelConfig.from_dict(data)
        self.assertEqual(cfg.model_type, "mlp")
        self.assertIsNone(cfg.transformer)
        self.assertIsNotNone(cfg.mlp)
        assert cfg.mlp is not None
        self.assertEqual(cfg.mlp.hidden_dims, [128, 64])
        self.assertEqual(cfg.mlp.dropout_rate, 0.1)

    def test_as_dict_includes_mlp_when_present(self):
        """as_dict includes mlp key when model has mlp params (for JSON round-trip)."""
        cfg = config.ArrowModelConfig.from_dict(
            {"model_type": "mlp", "mlp": {"hidden_dims": [64], "dropout_rate": 0.0}}
        )
        d = cfg.as_dict()
        self.assertIn("mlp", d)
        self.assertEqual(d["mlp"]["hidden_dims"], [64])
        self.assertEqual(d["mlp"]["dropout_rate"], 0.0)

    def test_mlp_params_from_dict_raises_attribute_error_for_unknown_keys(self):
        with self.assertRaises(TypeError):
            config.MLPArrowParams.from_dict({"unknown_key": 99})

    def test_from_dict_nested_lstm(self):
        """Test creating config with model_type lstm and lstm block."""
        data = {
            "model_type": "lstm",
            "lstm": {
                "units": 64,
                "num_layers": 2,
                "dropout_rate": 0.1,
                "bidirectional": True,
            },
        }
        cfg = config.ArrowModelConfig.from_dict(data)
        self.assertEqual(cfg.model_type, "lstm")
        self.assertIsNotNone(cfg.lstm)
        assert cfg.lstm is not None
        self.assertEqual(cfg.lstm.units, 64)
        self.assertEqual(cfg.lstm.num_layers, 2)
        self.assertEqual(cfg.lstm.dropout_rate, 0.1)
        self.assertTrue(cfg.lstm.bidirectional)

    def test_from_dict_nested_gru(self):
        """Test creating config with model_type gru and gru block."""
        data = {
            "model_type": "gru",
            "gru": {
                "units": 64,
                "num_layers": 2,
                "dropout_rate": 0.1,
                "bidirectional": True,
            },
        }
        cfg = config.ArrowModelConfig.from_dict(data)
        self.assertEqual(cfg.model_type, "gru")
        self.assertIsNotNone(cfg.gru)
        assert cfg.gru is not None
        self.assertEqual(cfg.gru.units, 64)
        self.assertEqual(cfg.gru.num_layers, 2)
        self.assertEqual(cfg.gru.dropout_rate, 0.1)
        self.assertTrue(cfg.gru.bidirectional)

    def test_as_dict_includes_lstm_when_present(self):
        """as_dict includes lstm key when model has lstm params (for JSON round-trip)."""
        cfg = config.ArrowModelConfig.from_dict(
            {
                "model_type": "lstm",
                "lstm": {
                    "units": 64,
                    "num_layers": 1,
                    "dropout_rate": 0.0,
                    "bidirectional": True,
                },
            }
        )
        d = cfg.as_dict()
        self.assertIn("lstm", d)
        self.assertEqual(d["lstm"]["units"], 64)
        self.assertEqual(d["lstm"]["num_layers"], 1)
        self.assertEqual(d["lstm"]["dropout_rate"], 0.0)
        self.assertEqual(d["lstm"]["bidirectional"], True)
        # Round-trip: from_dict(as_dict()) preserves bidirectional
        cfg2 = config.ArrowModelConfig.from_dict(d)
        self.assertIsNotNone(cfg2.lstm)
        assert cfg2.lstm is not None
        self.assertTrue(cfg2.lstm.bidirectional)

    def test_as_dict_includes_gru_when_present(self):
        """as_dict includes gru key when model has gru params (for JSON round-trip)."""
        cfg = config.ArrowModelConfig.from_dict(
            {
                "model_type": "gru",
                "gru": {
                    "units": 64,
                    "num_layers": 1,
                    "dropout_rate": 0.0,
                    "bidirectional": True,
                },
            }
        )
        d = cfg.as_dict()
        self.assertIn("gru", d)
        self.assertEqual(d["gru"]["units"], 64)
        self.assertEqual(d["gru"]["num_layers"], 1)
        self.assertEqual(d["gru"]["dropout_rate"], 0.0)
        self.assertEqual(d["gru"]["bidirectional"], True)
        cfg2 = config.ArrowModelConfig.from_dict(d)
        self.assertIsNotNone(cfg2.gru)
        assert cfg2.gru is not None
        self.assertTrue(cfg2.gru.bidirectional)

    def test_get_experiment_name_parts_transformer(self):
        """get_experiment_name_parts returns transformer param fragments when model_type is transformer."""
        cfg = config.ArrowModelConfig.from_dict(
            {
                "model_type": "transformer",
                "transformer": {"num_layers": 2, "d_model": 64},
            }
        )
        parts = cfg.get_experiment_name_parts()
        self.assertIn("att_layers_2", parts)
        self.assertIn("d_model_64", parts)

    def test_get_experiment_name_parts_transformer_with_timing_position(self):
        """When transformer.use_timing_position is True, experiment name parts include timing_pos."""
        cfg = config.ArrowModelConfig.from_dict(
            {
                "model_type": "transformer",
                "transformer": {
                    "num_layers": 1,
                    "d_model": 64,
                    "use_timing_position": True,
                },
            }
        )
        parts = cfg.get_experiment_name_parts()
        self.assertIn("timing_pos", parts)
        self.assertIn("att_layers_1", parts)

    def test_get_experiment_name_parts_mlp(self):
        """get_experiment_name_parts returns mlp param fragments when model_type is mlp."""
        cfg = config.ArrowModelConfig.from_dict(
            {
                "model_type": "mlp",
                "mlp": {"hidden_dims": [128, 64], "dropout_rate": 0.1},
            }
        )
        parts = cfg.get_experiment_name_parts()
        self.assertIn("mlp_128_64", parts)
        self.assertIn("dropout_0_1", parts)

    def test_get_experiment_name_parts_lstm(self):
        """get_experiment_name_parts returns lstm param fragments when model_type is lstm."""
        cfg = config.ArrowModelConfig.from_dict(
            {
                "model_type": "lstm",
                "lstm": {"units": 64, "num_layers": 2, "dropout_rate": 0.2},
            }
        )
        parts = cfg.get_experiment_name_parts()
        self.assertIn("lstm_units_64", parts)
        self.assertIn("lstm_layers_2", parts)
        self.assertIn("dropout_0_2", parts)
        self.assertNotIn("lstm_bidir", parts)

    def test_get_experiment_name_parts_lstm_bidirectional(self):
        """When lstm.bidirectional is True, experiment name parts include lstm_bidir."""
        cfg = config.ArrowModelConfig.from_dict(
            {
                "model_type": "lstm",
                "lstm": {
                    "units": 64,
                    "num_layers": 2,
                    "dropout_rate": 0.2,
                    "bidirectional": True,
                },
            }
        )
        parts = cfg.get_experiment_name_parts()
        self.assertIn("lstm_bidir", parts)
        self.assertIn("lstm_units_64", parts)

    def test_get_experiment_name_parts_gru(self):
        """get_experiment_name_parts returns gru param fragments when model_type is gru."""
        cfg = config.ArrowModelConfig.from_dict(
            {
                "model_type": "gru",
                "gru": {"units": 64, "num_layers": 2, "dropout_rate": 0.2},
            }
        )
        parts = cfg.get_experiment_name_parts()
        self.assertIn("gru_units_64", parts)
        self.assertIn("gru_layers_2", parts)
        self.assertIn("dropout_0_2", parts)
        self.assertNotIn("gru_bidir", parts)

    def test_get_experiment_name_parts_gru_bidirectional(self):
        """When gru.bidirectional is True, experiment name parts include gru_bidir."""
        cfg = config.ArrowModelConfig.from_dict(
            {
                "model_type": "gru",
                "gru": {
                    "units": 64,
                    "num_layers": 2,
                    "dropout_rate": 0.2,
                    "bidirectional": True,
                },
            }
        )
        parts = cfg.get_experiment_name_parts()
        self.assertIn("gru_bidir", parts)
        self.assertIn("gru_units_64", parts)

    def test_get_experiment_name_parts_gru_with_attention(self):
        """When gru.add_attention_layer is True, experiment name parts include attn, attn_heads, attn_dim."""
        cfg = config.ArrowModelConfig.from_dict(
            {
                "model_type": "gru",
                "gru": {
                    "units": 64,
                    "num_layers": 1,
                    "dropout_rate": 0.0,
                    "add_attention_layer": True,
                    "attention_heads": 8,
                    "attention_dim": 32,
                },
            }
        )
        parts = cfg.get_experiment_name_parts()
        self.assertIn("attn", parts)
        self.assertIn("attn_heads_8", parts)
        self.assertIn("attn_dim_32", parts)
        self.assertIn("gru_units_64", parts)

    def test_get_active_params_block_returns_none_for_unknown_model_type(self):
        """get_active_params_block returns None when model_type is not in the registry."""
        cfg = config.ArrowModelConfig(model_type="unknown_arch")
        self.assertIsNone(cfg.get_active_params_block())
        self.assertEqual(cfg.get_experiment_name_parts(), [])

    def test_get_experiment_name_parts_uses_only_active_block(self):
        """When both transformer and lstm are set, get_experiment_name_parts uses only active model_type."""
        cfg = config.ArrowModelConfig(
            model_type="lstm",
            transformer=config.TransformerArrowParams(num_layers=2, d_model=128),
            mlp=None,
            lstm=config.LSTMArrowParams(units=32, num_layers=1, dropout_rate=0.0),
        )
        parts = cfg.get_experiment_name_parts()
        self.assertIn("lstm_units_32", parts)
        self.assertNotIn("att_layers", parts)
        self.assertNotIn("d_model", parts)

    def test_get_experiment_name_parts_gru_uses_only_active_block(self):
        """When both lstm and gru are set, get_experiment_name_parts uses only active model_type (gru)."""
        cfg = config.ArrowModelConfig(
            model_type="gru",
            transformer=None,
            mlp=None,
            lstm=config.LSTMArrowParams(units=64, num_layers=2, dropout_rate=0.0),
            gru=config.GRUArrowParams(units=32, num_layers=1, dropout_rate=0.0),
        )
        parts = cfg.get_experiment_name_parts()
        self.assertIn("gru_units_32", parts)
        self.assertNotIn("lstm_units", parts)
        self.assertNotIn("lstm_layers", parts)

    def test_get_experiment_name_parts_includes_input_options_when_set(self):
        """get_experiment_name_parts returns only active block parts (input options live on dataset)."""
        cfg = config.ArrowModelConfig.from_dict(
            {
                "model_type": "transformer",
                "transformer": {"num_layers": 1, "d_model": 64},
            }
        )
        parts = cfg.get_experiment_name_parts()
        self.assertIn("att_layers_1", parts)
        self.assertIn("d_model_64", parts)

    def test_get_experiment_name_parts_omits_interval_encoding_when_default(self):
        """get_experiment_name_parts returns only transformer block parts."""
        cfg = config.ArrowModelConfig.from_dict(
            {
                "model_type": "transformer",
                "transformer": {"num_layers": 1, "d_model": 64},
            }
        )
        parts = cfg.get_experiment_name_parts()
        self.assertIn("att_layers_1", parts)
        self.assertIn("d_model_64", parts)

    def test_from_dict_nested_tcn(self):
        """Test creating config with model_type tcn and tcn block; round-trip."""
        data = {
            "model_type": "tcn",
            "tcn": {
                "filters": 32,
                "kernel_size": 3,
                "num_layers": 4,
                "dilation_base": 2,
                "dropout_rate": 0.1,
            },
        }
        cfg = config.ArrowModelConfig.from_dict(data)
        self.assertEqual(cfg.model_type, "tcn")
        self.assertIsNotNone(cfg.tcn)
        assert cfg.tcn is not None
        self.assertEqual(cfg.tcn.filters, 32)
        self.assertEqual(cfg.tcn.kernel_size, 3)
        self.assertEqual(cfg.tcn.num_layers, 4)
        self.assertEqual(cfg.tcn.dilation_base, 2)
        self.assertEqual(cfg.tcn.dropout_rate, 0.1)
        d = cfg.as_dict()
        self.assertIn("tcn", d)
        cfg2 = config.ArrowModelConfig.from_dict(d)
        self.assertEqual(cfg2.model_type, "tcn")
        assert cfg2.tcn is not None
        self.assertEqual(cfg2.tcn.filters, 32)

    def test_from_dict_nested_cnn1d(self):
        """Test creating config with model_type cnn1d and cnn1d block; round-trip."""
        data = {
            "model_type": "cnn1d",
            "cnn1d": {
                "filters": 64,
                "kernel_sizes": [3, 5, 3],
                "dropout_rate": 0.2,
            },
        }
        cfg = config.ArrowModelConfig.from_dict(data)
        self.assertEqual(cfg.model_type, "cnn1d")
        self.assertIsNotNone(cfg.cnn1d)
        assert cfg.cnn1d is not None
        self.assertEqual(cfg.cnn1d.filters, 64)
        self.assertEqual(cfg.cnn1d.kernel_sizes, [3, 5, 3])
        self.assertEqual(cfg.cnn1d.dropout_rate, 0.2)
        d = cfg.as_dict()
        self.assertIn("cnn1d", d)
        cfg2 = config.ArrowModelConfig.from_dict(d)
        self.assertEqual(cfg2.model_type, "cnn1d")
        assert cfg2.cnn1d is not None
        self.assertEqual(cfg2.cnn1d.kernel_sizes, [3, 5, 3])

    def test_from_dict_interval_encoding_use_step_index_use_beat_phase(self):
        """Round-trip for gru block; input options live on dataset, not model config."""
        data = {
            "model_type": "gru",
            "gru": {"units": 64, "num_layers": 1},
        }
        cfg = config.ArrowModelConfig.from_dict(data)
        self.assertEqual(cfg.model_type, "gru")
        self.assertIsNotNone(cfg.gru)
        assert cfg.gru is not None
        self.assertEqual(cfg.gru.units, 64)
        self.assertEqual(cfg.gru.num_layers, 1)
        d = cfg.as_dict()
        self.assertEqual(d["model_type"], "gru")
        self.assertIn("gru", d)
        self.assertEqual(d["gru"]["units"], 64)

    def test_get_experiment_name_parts_tcn(self):
        """get_experiment_name_parts returns tcn param fragments when model_type is tcn."""
        cfg = config.ArrowModelConfig.from_dict(
            {
                "model_type": "tcn",
                "tcn": {"filters": 32, "num_layers": 2, "dilation_base": 2},
            }
        )
        parts = cfg.get_experiment_name_parts()
        self.assertIn("tcn_filters_32", parts)
        self.assertIn("tcn_layers_2", parts)
        self.assertIn("tcn_dilation_base_2", parts)

    def test_get_experiment_name_parts_cnn1d(self):
        """get_experiment_name_parts returns cnn1d param fragments when model_type is cnn1d."""
        cfg = config.ArrowModelConfig.from_dict(
            {
                "model_type": "cnn1d",
                "cnn1d": {"filters": 64, "kernel_sizes": [3, 3], "dropout_rate": 0.0},
            }
        )
        parts = cfg.get_experiment_name_parts()
        self.assertIn("cnn1d_filters_64", parts)
        self.assertIn("cnn1d_kernels_3_3", parts)


class RunConfigTest(unittest.TestCase):
    def test_create_with_required_fields(self):
        """Test creating config with only required fields."""
        cfg = config.RunConfig(epoch=10, take_count=100, model_output_dir="out")
        self.assertEqual(cfg.epoch, 10)
        self.assertEqual(cfg.take_count, 100)
        self.assertEqual(cfg.model_output_dir, "out")
        self.assertEqual(cfg.callback_root_dir, "")  # default
        self.assertIsNone(cfg.seed)  # default
        self.assertEqual(cfg.confidence_threshold, 0.05)
        self.assertEqual(cfg.tolerance_sec, 0.02)
        self.assertEqual(cfg.min_onset_distance_ms, 50.0)
        self.assertEqual(cfg.early_stopping_patience, 25)

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

    def test_from_dict_rejects_unknown_keys(self):
        """from_dict raises TypeError when given unknown keys (e.g. arrow-only aux weights)."""
        data = {
            "epoch": 1,
            "take_count": 1,
            "model_output_dir": "out",
            "chart_validity_aux_weight": 0.5,
            "diversity_aux_weight": 0.2,
        }
        with self.assertRaises(TypeError) as ctx:
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

    def test_run_config_rejects_invalid_dense_eval_fields(self):
        with self.assertRaises(ValueError):
            config.RunConfig(
                epoch=1,
                take_count=1,
                model_output_dir="out",
                confidence_threshold=1.5,
            )
        with self.assertRaises(ValueError):
            config.RunConfig(
                epoch=1,
                take_count=1,
                model_output_dir="out",
                early_stopping_patience=-1,
            )

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

    def test_post_hoc_event_f1_defaults(self):
        """Post-hoc event-F1 export defaults to disabled with a standard grid."""
        cfg = config.RunConfig(epoch=1, take_count=1, model_output_dir="out")
        self.assertFalse(cfg.post_hoc_event_f1_export)
        self.assertEqual(cfg.post_hoc_event_f1_thresholds[0], 0.05)
        self.assertEqual(cfg.post_hoc_event_f1_thresholds[-1], 0.5)

    def test_post_hoc_event_f1_roundtrip(self):
        """from_dict/as_dict preserve post-hoc event-F1 settings."""
        data = {
            "epoch": 1,
            "take_count": 1,
            "model_output_dir": "out",
            "post_hoc_event_f1_export": True,
            "post_hoc_event_f1_thresholds": [0.2, 0.35],
        }
        cfg = config.RunConfig.from_dict(data)
        self.assertTrue(cfg.post_hoc_event_f1_export)
        self.assertEqual(cfg.post_hoc_event_f1_thresholds, [0.2, 0.35])
        self.assertEqual(cfg.as_dict()["post_hoc_event_f1_thresholds"], [0.2, 0.35])

    def test_post_hoc_event_f1_rejects_empty_when_enabled(self):
        """Enabling export with an empty threshold grid raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            config.RunConfig(
                epoch=1,
                take_count=1,
                model_output_dir="out",
                post_hoc_event_f1_export=True,
                post_hoc_event_f1_thresholds=[],
            )
        self.assertIn("post_hoc_event_f1_thresholds", str(ctx.exception))

    def test_post_hoc_event_f1_rejects_out_of_range_threshold(self):
        """Threshold values outside [0, 1] raise ValueError."""
        with self.assertRaises(ValueError) as ctx:
            config.RunConfig(
                epoch=1,
                take_count=1,
                model_output_dir="out",
                post_hoc_event_f1_thresholds=[0.2, 1.5],
            )
        self.assertIn("post_hoc_event_f1_thresholds", str(ctx.exception))


class ArrowRunConfigTest(unittest.TestCase):
    """ArrowRunConfig: RunConfig fields plus chart_validity_aux_weight and diversity_aux_weight."""

    def test_aux_weights_default_zero(self):
        """Default aux weights are 0 and valid."""
        cfg = config.ArrowRunConfig(epoch=1, take_count=1, model_output_dir="out")
        self.assertEqual(cfg.chart_validity_aux_weight, 0.0)
        self.assertEqual(cfg.diversity_aux_weight, 0.0)

    def test_aux_weights_accept_non_negative(self):
        """Non-negative aux weights are accepted."""
        cfg = config.ArrowRunConfig(
            epoch=1,
            take_count=1,
            model_output_dir="out",
            chart_validity_aux_weight=0.5,
            diversity_aux_weight=0.2,
        )
        self.assertEqual(cfg.chart_validity_aux_weight, 0.5)
        self.assertEqual(cfg.diversity_aux_weight, 0.2)

    def test_negative_chart_validity_aux_weight_raises(self):
        """chart_validity_aux_weight < 0 raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            config.ArrowRunConfig(
                epoch=1,
                take_count=1,
                model_output_dir="out",
                chart_validity_aux_weight=-0.1,
            )
        self.assertIn("chart_validity_aux_weight", str(ctx.exception))

    def test_negative_diversity_aux_weight_raises(self):
        """diversity_aux_weight < 0 raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            config.ArrowRunConfig(
                epoch=1,
                take_count=1,
                model_output_dir="out",
                diversity_aux_weight=-1.0,
            )
        self.assertIn("diversity_aux_weight", str(ctx.exception))

    def test_from_dict_negative_aux_weight_raises(self):
        """from_dict with negative aux weight raises ValueError."""
        data = {
            "epoch": 1,
            "take_count": 1,
            "model_output_dir": "out",
            "chart_validity_aux_weight": -0.5,
        }
        with self.assertRaises(ValueError) as ctx:
            config.ArrowRunConfig.from_dict(data)
        self.assertIn("chart_validity_aux_weight", str(ctx.exception))

    def test_from_dict_accepts_aux_weights(self):
        """from_dict accepts chart_validity_aux_weight and diversity_aux_weight."""
        data = {
            "epoch": 2,
            "take_count": 1,
            "model_output_dir": "out",
            "chart_validity_aux_weight": 0.3,
            "diversity_aux_weight": 0.1,
        }
        cfg = config.ArrowRunConfig.from_dict(data)
        self.assertEqual(cfg.chart_validity_aux_weight, 0.3)
        self.assertEqual(cfg.diversity_aux_weight, 0.1)

    def test_as_dict_includes_aux_weights(self):
        """as_dict includes run keys and aux weights for JSON round-trip."""
        cfg = config.ArrowRunConfig(
            epoch=1,
            take_count=1,
            model_output_dir="out",
            chart_validity_aux_weight=0.4,
            diversity_aux_weight=0.0,
        )
        d = cfg.as_dict()
        self.assertEqual(d["chart_validity_aux_weight"], 0.4)
        self.assertEqual(d["diversity_aux_weight"], 0.0)

    def test_from_dict_rejects_unknown_keys(self):
        """from_dict raises TypeError when given unknown keys."""
        data = {
            "epoch": 1,
            "take_count": 1,
            "model_output_dir": "out",
            "chart_validity_aux_weight": 0.1,
            "unknown_param": 99,
        }
        with self.assertRaises(TypeError) as ctx:
            config.ArrowRunConfig.from_dict(data)
        self.assertIn("unknown_param", str(ctx.exception))

    def test_get_experiment_name_parts_returns_take_aux_loss_and_loss_options(self):
        """get_experiment_name_parts returns take_*, aux weights, focal, label_smooth, aux_interval when set."""
        cfg = config.ArrowRunConfig(
            epoch=1,
            take_count=42,
            model_output_dir="out",
            chart_validity_aux_weight=0.2,
            diversity_aux_weight=0.1,
            loss_type="focal",
            focal_gamma=3.0,
            label_smoothing=0.1,
            aux_interval_weight=0.5,
        )
        parts = cfg.get_experiment_name_parts()
        self.assertIn("take_42", parts)
        self.assertIn("chart_val_aux_0_2", parts)
        self.assertIn("diversity_aux_0_1", parts)
        self.assertIn("focal_gamma_3_0", parts)
        self.assertIn("label_smooth_0_1", parts)
        self.assertIn("aux_interval_0_5", parts)

    def test_get_experiment_name_parts_take_all(self):
        """get_experiment_name_parts returns take_all when take_count is -1."""
        cfg = config.ArrowRunConfig(epoch=1, take_count=-1, model_output_dir="out")
        parts = cfg.get_experiment_name_parts()
        self.assertIn("take_all", parts)

    def test_get_experiment_name_parts_omits_defaults(self):
        """get_experiment_name_parts omits aux weights and loss options when zero/default."""
        cfg = config.ArrowRunConfig(
            epoch=1,
            take_count=5,
            model_output_dir="out",
            chart_validity_aux_weight=0.0,
            diversity_aux_weight=0.0,
            loss_type="crossentropy",
            label_smoothing=0.0,
            aux_interval_weight=0.0,
        )
        parts = cfg.get_experiment_name_parts()
        self.assertIn("take_5", parts)
        self.assertNotIn("chart_val_aux", parts)
        self.assertNotIn("diversity_aux", parts)
        self.assertNotIn("focal_gamma", parts)
        self.assertNotIn("label_smooth", parts)
        self.assertNotIn("aux_interval", parts)

    def test_get_experiment_name_parts_includes_warmup_epochs_when_set(self):
        """get_experiment_name_parts includes warmup_epochs token when > 0."""
        cfg = config.ArrowRunConfig(
            epoch=5,
            take_count=1,
            model_output_dir="out",
            warmup_epochs=2,
        )
        parts = cfg.get_experiment_name_parts()
        self.assertIn("warmup_epochs_2", parts)

    def test_get_experiment_name_parts_omits_warmup_epochs_when_zero(self):
        """get_experiment_name_parts omits warmup_epochs token when warmup_epochs == 0."""
        cfg = config.ArrowRunConfig(
            epoch=5,
            take_count=1,
            model_output_dir="out",
            warmup_epochs=0,
        )
        parts = cfg.get_experiment_name_parts()
        self.assertNotIn("warmup_epochs_0", parts)

    def test_chart_validity_rejection_defaults(self):
        """chart_validity_rejection_* default to None, 10.0, 50.0."""
        cfg = config.ArrowRunConfig(epoch=1, take_count=1, model_output_dir="out")
        self.assertIsNone(cfg.chart_validity_rejection_threshold)
        self.assertEqual(cfg.chart_validity_rejection_scale, 10.0)
        self.assertEqual(cfg.chart_validity_rejection_temperature, 50.0)

    def test_chart_validity_rejection_threshold_valid_accepted(self):
        """chart_validity_rejection_threshold in (0, 1] with scale > 0 is accepted."""
        cfg = config.ArrowRunConfig(
            epoch=1,
            take_count=1,
            model_output_dir="out",
            chart_validity_rejection_threshold=0.99,
            chart_validity_rejection_scale=100.0,
        )
        self.assertEqual(cfg.chart_validity_rejection_threshold, 0.99)
        self.assertEqual(cfg.chart_validity_rejection_scale, 100.0)

    def test_chart_validity_rejection_threshold_zero_raises(self):
        """chart_validity_rejection_threshold 0 raises when set."""
        with self.assertRaises(ValueError) as ctx:
            config.ArrowRunConfig(
                epoch=1,
                take_count=1,
                model_output_dir="out",
                chart_validity_rejection_threshold=0.0,
                chart_validity_rejection_scale=10.0,
            )
        self.assertIn("chart_validity_rejection_threshold", str(ctx.exception))

    def test_chart_validity_rejection_threshold_above_one_raises(self):
        """chart_validity_rejection_threshold > 1 raises when set."""
        with self.assertRaises(ValueError) as ctx:
            config.ArrowRunConfig(
                epoch=1,
                take_count=1,
                model_output_dir="out",
                chart_validity_rejection_threshold=1.01,
                chart_validity_rejection_scale=10.0,
            )
        self.assertIn("chart_validity_rejection_threshold", str(ctx.exception))

    def test_chart_validity_rejection_scale_zero_when_threshold_set_raises(self):
        """chart_validity_rejection_scale <= 0 raises when threshold is set."""
        with self.assertRaises(ValueError) as ctx:
            config.ArrowRunConfig(
                epoch=1,
                take_count=1,
                model_output_dir="out",
                chart_validity_rejection_threshold=0.99,
                chart_validity_rejection_scale=0.0,
            )
        self.assertIn("chart_validity_rejection_scale", str(ctx.exception))

    def test_chart_validity_rejection_temperature_zero_when_threshold_set_raises(self):
        """chart_validity_rejection_temperature <= 0 raises when threshold is set."""
        with self.assertRaises(ValueError) as ctx:
            config.ArrowRunConfig(
                epoch=1,
                take_count=1,
                model_output_dir="out",
                chart_validity_rejection_threshold=0.99,
                chart_validity_rejection_scale=10.0,
                chart_validity_rejection_temperature=0.0,
            )
        self.assertIn("chart_validity_rejection_temperature", str(ctx.exception))

    def test_get_experiment_name_parts_includes_chart_val_rej_when_threshold_set(self):
        """get_experiment_name_parts includes chart_val_rej threshold, scale, and temp when threshold is set."""
        cfg = config.ArrowRunConfig(
            epoch=1,
            take_count=1,
            model_output_dir="out",
            chart_validity_rejection_threshold=0.99,
            chart_validity_rejection_scale=5.0,
            chart_validity_rejection_temperature=25.0,
        )
        parts = cfg.get_experiment_name_parts()
        self.assertIn("chart_val_rej_0_99", parts)
        self.assertIn("chart_val_rej_scale_5_0", parts)
        self.assertIn("chart_val_rej_temp_25_0", parts)

    def test_as_dict_includes_rejection_params(self):
        """as_dict includes chart_validity_rejection_* for round-trip."""
        cfg = config.ArrowRunConfig(
            epoch=1,
            take_count=1,
            model_output_dir="out",
            chart_validity_rejection_threshold=0.99,
            chart_validity_rejection_scale=20.0,
            chart_validity_rejection_temperature=100.0,
        )
        d = cfg.as_dict()
        self.assertEqual(d["chart_validity_rejection_threshold"], 0.99)
        self.assertEqual(d["chart_validity_rejection_scale"], 20.0)
        self.assertEqual(d["chart_validity_rejection_temperature"], 100.0)
        loaded = config.ArrowRunConfig.from_dict(d)
        self.assertEqual(loaded.chart_validity_rejection_threshold, 0.99)
        self.assertEqual(loaded.chart_validity_rejection_scale, 20.0)
        self.assertEqual(loaded.chart_validity_rejection_temperature, 100.0)

    def test_negative_warmup_epochs_raises(self):
        """warmup_epochs < 0 raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            config.ArrowRunConfig(
                epoch=5,
                take_count=1,
                model_output_dir="out",
                warmup_epochs=-1,
            )
        self.assertIn("warmup_epochs", str(ctx.exception))

    def test_warmup_epochs_ge_epoch_raises(self):
        """warmup_epochs >= epoch raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            config.ArrowRunConfig(
                epoch=5,
                take_count=1,
                model_output_dir="out",
                warmup_epochs=5,
            )
        self.assertIn("warmup_epochs", str(ctx.exception))

    def test_lr_peak_non_positive_raises(self):
        """lr_peak <= 0 raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            config.ArrowRunConfig(
                epoch=5,
                take_count=1,
                model_output_dir="out",
                lr_peak=0.0,
            )
        self.assertIn("lr_peak", str(ctx.exception))

    def test_negative_lr_min_raises(self):
        """lr_min < 0 raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            config.ArrowRunConfig(
                epoch=5,
                take_count=1,
                model_output_dir="out",
                lr_min=-0.1,
            )
        self.assertIn("lr_min", str(ctx.exception))

    def test_lr_min_ge_lr_peak_raises(self):
        """lr_min >= lr_peak raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            config.ArrowRunConfig(
                epoch=5,
                take_count=1,
                model_output_dir="out",
                lr_peak=1e-3,
                lr_min=1e-3,
            )
        self.assertIn("lr_min", str(ctx.exception))

    def test_round_trip_loss_type_focal_gamma_label_smoothing_aux_interval_weight(self):
        """Round-trip and validation for loss_type, focal_gamma, label_smoothing, aux_interval_weight."""
        data = {
            "epoch": 2,
            "take_count": 1,
            "model_output_dir": "out",
            "loss_type": "focal",
            "focal_gamma": 2.5,
            "label_smoothing": 0.1,
            "aux_interval_weight": 0.5,
        }
        cfg = config.ArrowRunConfig.from_dict(data)
        self.assertEqual(cfg.loss_type, "focal")
        self.assertEqual(cfg.focal_gamma, 2.5)
        self.assertEqual(cfg.label_smoothing, 0.1)
        self.assertEqual(cfg.aux_interval_weight, 0.5)
        d = cfg.as_dict()
        self.assertEqual(d["loss_type"], "focal")
        self.assertEqual(d["focal_gamma"], 2.5)
        self.assertEqual(d["label_smoothing"], 0.1)
        self.assertEqual(d["aux_interval_weight"], 0.5)
        cfg2 = config.ArrowRunConfig.from_dict(d)
        self.assertEqual(cfg2.loss_type, cfg.loss_type)
        self.assertEqual(cfg2.aux_interval_weight, cfg.aux_interval_weight)

    def test_loss_type_invalid_raises(self):
        """loss_type other than crossentropy or focal raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            config.ArrowRunConfig(
                epoch=1,
                take_count=1,
                model_output_dir="out",
                loss_type="mse",
            )
        self.assertIn("loss_type", str(ctx.exception))

    def test_focal_gamma_negative_raises(self):
        """focal_gamma < 0 raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            config.ArrowRunConfig(
                epoch=1,
                take_count=1,
                model_output_dir="out",
                focal_gamma=-0.1,
            )
        self.assertIn("focal_gamma", str(ctx.exception))

    def test_label_smoothing_invalid_raises(self):
        """label_smoothing outside [0, 1) raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            config.ArrowRunConfig(
                epoch=1,
                take_count=1,
                model_output_dir="out",
                label_smoothing=1.0,
            )
        self.assertIn("label_smoothing", str(ctx.exception))

    def test_aux_interval_weight_negative_raises(self):
        """aux_interval_weight < 0 raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            config.ArrowRunConfig(
                epoch=1,
                take_count=1,
                model_output_dir="out",
                aux_interval_weight=-0.1,
            )
        self.assertIn("aux_interval_weight", str(ctx.exception))


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
            config_path = pathlib.Path(temp_dir) / "test_config.json"
            exp_cfg.to_json(config_path)

            # Verify file exists
            self.assertTrue(pathlib.Path(config_path).exists())

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
            config_path = pathlib.Path(temp_dir) / "subdir" / "config.json"
            exp_cfg.to_json(config_path)
            self.assertTrue(pathlib.Path(config_path).exists())

    def test_from_json_file_not_found(self):
        """Test that loading non-existent file raises FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            config.OnsetExperimentConfig.from_json("nonexistent.json")

    def test_from_json_invalid_json(self):
        """Test that invalid JSON raises JSONDecodeError."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = pathlib.Path(temp_dir) / "invalid.json"
            with pathlib.Path(config_path).open("w") as f:
                f.write("invalid json content {")
            with self.assertRaises(json.JSONDecodeError):
                config.OnsetExperimentConfig.from_json(config_path)


class ArrowExperimentConfigTest(unittest.TestCase):
    def test_create_experiment_config(self):
        """Test creating complete experiment config; run is ArrowRunConfig."""
        dataset_cfg = config.ArrowDatasetConfig(
            data_dir="data/train", val_data_dir="data/val"
        )
        model_cfg = config.ArrowModelConfig()
        run_cfg = config.ArrowRunConfig(epoch=10, take_count=-1, model_output_dir="out")
        exp_cfg = config.ArrowExperimentConfig(
            dataset=dataset_cfg, model=model_cfg, run=run_cfg
        )
        self.assertEqual(exp_cfg.dataset, dataset_cfg)
        self.assertEqual(exp_cfg.model, model_cfg)
        self.assertEqual(exp_cfg.run, run_cfg)
        self.assertIsInstance(exp_cfg.run, config.ArrowRunConfig)

    def test_as_dict(self):
        """Test converting experiment config to dictionary; input options only under dataset."""
        dataset_cfg = config.ArrowDatasetConfig(
            data_dir="data/train",
            val_data_dir="data/val",
            batch_size=2,
            use_interval=True,
            interval_encoding=config.IntervalEncoding.LOG,
        )
        model_cfg = config.ArrowModelConfig.from_dict(
            {"transformer": {"num_layers": 2}}
        )
        run_cfg = config.ArrowRunConfig(epoch=10, take_count=-1, model_output_dir="out")
        exp_cfg = config.ArrowExperimentConfig(
            dataset=dataset_cfg, model=model_cfg, run=run_cfg
        )
        d = exp_cfg.as_dict()
        self.assertEqual(d["dataset"]["batch_size"], 2)
        self.assertEqual(d["dataset"]["use_interval"], True)
        self.assertEqual(d["dataset"]["interval_encoding"], "log")
        self.assertEqual(d["model"]["transformer"]["num_layers"], 2)

    def test_to_json_and_from_json(self):
        """Test saving and loading config from JSON file."""
        dataset_cfg = config.ArrowDatasetConfig(
            data_dir="data/train", val_data_dir="data/val"
        )
        model_cfg = config.ArrowModelConfig.from_dict(
            {"transformer": {"num_layers": 3, "d_model": 256}}
        )
        run_cfg = config.ArrowRunConfig(
            epoch=15, take_count=-1, model_output_dir="out", seed=99
        )
        exp_cfg = config.ArrowExperimentConfig(
            dataset=dataset_cfg, model=model_cfg, run=run_cfg
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = pathlib.Path(temp_dir) / "arrow_config.json"
            exp_cfg.to_json(config_path)

            loaded_cfg = config.ArrowExperimentConfig.from_json(config_path)
            self.assertEqual(loaded_cfg.dataset.data_dir, "data/train")
            assert loaded_cfg.model.transformer is not None
            self.assertEqual(loaded_cfg.model.transformer.num_layers, 3)
            self.assertEqual(loaded_cfg.model.transformer.d_model, 256)
            self.assertEqual(loaded_cfg.run.seed, 99)

    def test_run_is_arrow_run_config_round_trip_includes_aux_weights(self):
        """ArrowExperimentConfig.run is ArrowRunConfig; as_dict/from_dict round-trip includes aux weights."""
        run_cfg = config.ArrowRunConfig(
            epoch=1,
            take_count=1,
            model_output_dir="out",
            chart_validity_aux_weight=0.3,
            diversity_aux_weight=0.1,
        )
        exp_cfg = config.ArrowExperimentConfig(
            dataset=config.ArrowDatasetConfig(data_dir="d", val_data_dir="v"),
            model=config.ArrowModelConfig(),
            run=run_cfg,
        )
        d = exp_cfg.as_dict()
        self.assertIn("chart_validity_aux_weight", d["run"])
        self.assertIn("diversity_aux_weight", d["run"])
        self.assertEqual(d["run"]["chart_validity_aux_weight"], 0.3)
        self.assertEqual(d["run"]["diversity_aux_weight"], 0.1)
        loaded = config.ArrowExperimentConfig.from_dict(d)
        self.assertIsInstance(loaded.run, config.ArrowRunConfig)
        self.assertEqual(loaded.run.chart_validity_aux_weight, 0.3)
        self.assertEqual(loaded.run.diversity_aux_weight, 0.1)

    def test_arrow_experiment_round_trip_includes_rejection_params(self):
        """ArrowExperimentConfig as_dict/from_dict round-trip includes chart_validity_rejection_*."""
        run_cfg = config.ArrowRunConfig(
            epoch=1,
            take_count=1,
            model_output_dir="out",
            chart_validity_rejection_threshold=0.99,
            chart_validity_rejection_scale=15.0,
        )
        exp_cfg = config.ArrowExperimentConfig(
            dataset=config.ArrowDatasetConfig(data_dir="d", val_data_dir="v"),
            model=config.ArrowModelConfig(),
            run=run_cfg,
        )
        d = exp_cfg.as_dict()
        self.assertEqual(d["run"]["chart_validity_rejection_threshold"], 0.99)
        self.assertEqual(d["run"]["chart_validity_rejection_scale"], 15.0)
        loaded = config.ArrowExperimentConfig.from_dict(d)
        self.assertEqual(loaded.run.chart_validity_rejection_threshold, 0.99)
        self.assertEqual(loaded.run.chart_validity_rejection_scale, 15.0)

    def test_from_dict_input_options_only_under_dataset(self):
        """Loading with input options only under dataset populates both configs."""
        data = {
            "dataset": {
                "data_dir": "d",
                "val_data_dir": "v",
                "snippet_half_frames": 3,
                "use_interval": True,
                "interval_encoding": "log",
            },
            "model": {"model_type": "transformer", "transformer": {"num_layers": 1}},
            "run": {"epoch": 1, "take_count": 1, "model_output_dir": "out"},
        }
        loaded = config.ArrowExperimentConfig.from_dict(data)
        self.assertEqual(loaded.dataset.snippet_half_frames, 3)
        self.assertEqual(loaded.dataset.use_interval, True)
        self.assertEqual(loaded.dataset.interval_encoding, config.IntervalEncoding.LOG)


if __name__ == "__main__":
    unittest.main()
