import unittest

import keras
import numpy as np

from stepcovnet import config, models


class ModelTest(unittest.TestCase):
    def test_build_unet_wavenet_model(self):
        model_instance = models.build_unet_wavenet_model()

        self.assertIsInstance(model_instance, keras.Model)

        self.assertEqual(model_instance.input_shape, (None, None, 128))
        self.assertEqual(model_instance.output_shape, (None, None, 1))

        # Call the model and check the output shape
        dummy_input = np.random.random((1, 100, 128)).astype(np.float32)
        prediction = model_instance.predict(dummy_input)
        self.assertEqual(prediction.shape, (1, 100, 1))

    def test_build_arrow_model_model(self):
        input_opts = models.ArrowInputOptions()
        output_opts = models.ArrowOutputOptions()
        params = config.TransformerArrowParams()
        model_instance = models.build_arrow_model(input_opts, output_opts, params)

        self.assertIsInstance(model_instance, keras.Model)

        self.assertEqual(model_instance.input_shape, (None, None, 1))
        self.assertEqual(model_instance.output_shape, (None, None, 256))

        # Call the model and check the output shape
        dummy_input = np.random.random((1, 100, 1)).astype(np.float32)
        prediction = model_instance.predict(dummy_input)
        self.assertEqual(prediction.shape, (1, 100, 256))

    def test_build_unet_wavenet_model_default_name(self):
        """Model has default name stepcovnet_ONSET when model_name is empty."""
        model = models.build_unet_wavenet_model(model_name="")
        self.assertEqual(model.name, "stepcovnet_ONSET")

    def test_build_unet_wavenet_model_custom_name(self):
        """Model name includes custom model_name suffix."""
        model = models.build_unet_wavenet_model(model_name="my_experiment")
        self.assertEqual(model.name, "stepcovnet_ONSET-my_experiment")

    def test_build_arrow_model_default_name(self):
        """Arrow model has default name stepcovnet_ARROW when model_name is empty."""
        input_opts = models.ArrowInputOptions()
        output_opts = models.ArrowOutputOptions(model_name="")
        params = config.TransformerArrowParams()
        model = models.build_arrow_model(input_opts, output_opts, params)
        self.assertEqual(model.name, "stepcovnet_ARROW")

    def test_build_arrow_model_custom_name(self):
        """Arrow model name includes custom model_name suffix."""
        input_opts = models.ArrowInputOptions()
        output_opts = models.ArrowOutputOptions(model_name="my_arrow_run")
        params = config.TransformerArrowParams()
        model = models.build_arrow_model(input_opts, output_opts, params)
        self.assertEqual(model.name, "stepcovnet_ARROW-my_arrow_run")

    def test_build_arrow_model_with_audio_snippets(self):
        """Arrow model with snippet_half_frames > 0 has two inputs and runs forward pass."""
        input_opts = models.ArrowInputOptions(snippet_half_frames=5)
        output_opts = models.ArrowOutputOptions()
        params = config.TransformerArrowParams()
        model = models.build_arrow_model(input_opts, output_opts, params)
        self.assertIsInstance(model, keras.Model)
        self.assertEqual(len(model.inputs), 2)
        timing_input = np.random.random((1, 100, 1)).astype(np.float32)
        snippet_input = np.random.random((1, 100, 11, 128)).astype(np.float32)
        out = model.predict([timing_input, snippet_input])
        self.assertEqual(out.shape, (1, 100, 256))

    def test_build_arrow_model_with_interval(self):
        """Arrow model with use_interval=True has two inputs (timing, interval) and runs forward pass."""
        input_opts = models.ArrowInputOptions(use_interval=True)
        output_opts = models.ArrowOutputOptions()
        params = config.TransformerArrowParams()
        model = models.build_arrow_model(input_opts, output_opts, params)
        self.assertIsInstance(model, keras.Model)
        self.assertEqual(len(model.inputs), 2)
        input_names = [inp.name for inp in model.inputs]
        self.assertIn("timing_input", input_names)
        self.assertIn("interval_input", input_names)
        timing_input = np.random.random((1, 100, 1)).astype(np.float32)
        interval_input = np.random.random((1, 100, 1)).astype(np.float32)
        out = model.predict([timing_input, interval_input])
        self.assertEqual(out.shape, (1, 100, 256))

    def test_build_arrow_model_with_interval_and_snippets(self):
        """Arrow model with use_interval and snippet_half_frames has three inputs."""
        input_opts = models.ArrowInputOptions(
            snippet_half_frames=5,
            use_interval=True,
        )
        output_opts = models.ArrowOutputOptions()
        params = config.TransformerArrowParams()
        model = models.build_arrow_model(input_opts, output_opts, params)
        self.assertEqual(len(model.inputs), 3)
        timing = np.random.random((1, 50, 1)).astype(np.float32)
        interval = np.random.random((1, 50, 1)).astype(np.float32)
        snippet = np.random.random((1, 50, 11, 128)).astype(np.float32)
        out = model.predict([timing, interval, snippet])
        self.assertEqual(out.shape, (1, 50, 256))

    def test_build_arrow_model_from_config_transformer(self):
        """build_arrow_model_from_config with model_type transformer matches build_arrow_model."""
        model_config = config.ArrowModelConfig.from_dict(
            {
                "model_type": "transformer",
                "transformer": {"num_layers": 1, "d_model": 128},
            }
        )
        input_opts = models.ArrowInputOptions()
        output_opts = models.ArrowOutputOptions(model_name="")
        model = models.build_arrow_model_from_config(
            model_config, input_opts, output_opts
        )
        self.assertIsInstance(model, keras.Model)
        self.assertEqual(model.input_shape, (None, None, 1))
        self.assertEqual(model.output_shape, (None, None, 256))
        dummy_input = np.random.random((1, 100, 1)).astype(np.float32)
        prediction = model.predict(dummy_input)
        self.assertEqual(prediction.shape, (1, 100, 256))
        self.assertEqual(model.name, "stepcovnet_ARROW")

    def test_build_arrow_model_from_config_mlp(self):
        """build_arrow_model_from_config with model_type mlp produces valid model."""
        model_config = config.ArrowModelConfig.from_dict(
            {
                "model_type": "mlp",
                "mlp": {"hidden_dims": [256, 128], "dropout_rate": 0.0},
            }
        )
        input_opts = models.ArrowInputOptions()
        output_opts = models.ArrowOutputOptions(model_name="mlp_run")
        model = models.build_arrow_model_from_config(
            model_config, input_opts, output_opts
        )
        self.assertIsInstance(model, keras.Model)
        self.assertEqual(len(model.inputs), 1)
        self.assertEqual(model.output_shape, (None, None, 256))
        dummy_input = np.random.random((1, 100, 1)).astype(np.float32)
        prediction = model.predict(dummy_input)
        self.assertEqual(prediction.shape, (1, 100, 256))
        self.assertIn("mlp_run", model.name)

    def test_build_arrow_model_from_config_mlp_with_snippets(self):
        """build_arrow_model_from_config mlp with snippet_half_frames has two inputs."""
        model_config = config.ArrowModelConfig.from_dict(
            {
                "model_type": "mlp",
                "mlp": {"hidden_dims": [128], "dropout_rate": 0.0},
            }
        )
        input_opts = models.ArrowInputOptions(snippet_half_frames=5)
        output_opts = models.ArrowOutputOptions(model_name="")
        model = models.build_arrow_model_from_config(
            model_config, input_opts, output_opts
        )
        self.assertIsInstance(model, keras.Model)
        self.assertEqual(len(model.inputs), 2)
        timing_input = np.random.random((1, 100, 1)).astype(np.float32)
        snippet_input = np.random.random((1, 100, 11, 128)).astype(np.float32)
        out = model.predict([timing_input, snippet_input])
        self.assertEqual(out.shape, (1, 100, 256))

    def test_build_arrow_model_from_config_mlp_with_interval(self):
        """build_arrow_model_from_config mlp with use_interval=True has two inputs and runs forward pass."""
        model_config = config.ArrowModelConfig.from_dict(
            {
                "model_type": "mlp",
                "mlp": {"hidden_dims": [256, 128], "dropout_rate": 0.0},
            }
        )
        input_opts = models.ArrowInputOptions(use_interval=True)
        output_opts = models.ArrowOutputOptions(model_name="")
        model = models.build_arrow_model_from_config(
            model_config, input_opts, output_opts
        )
        self.assertIsInstance(model, keras.Model)
        self.assertEqual(len(model.inputs), 2)
        input_names = [inp.name for inp in model.inputs]
        self.assertIn("timing_input", input_names)
        self.assertIn("interval_input", input_names)
        timing_input = np.random.random((1, 100, 1)).astype(np.float32)
        interval_input = np.random.random((1, 100, 1)).astype(np.float32)
        out = model.predict([timing_input, interval_input])
        self.assertEqual(out.shape, (1, 100, 256))

    def test_build_arrow_model_from_config_lstm(self):
        """build_arrow_model_from_config with model_type lstm produces valid model."""
        model_config = config.ArrowModelConfig.from_dict(
            {
                "model_type": "lstm",
                "lstm": {"units": 64, "num_layers": 1, "dropout_rate": 0.0},
            }
        )
        input_opts = models.ArrowInputOptions()
        output_opts = models.ArrowOutputOptions(model_name="lstm_run")
        model = models.build_arrow_model_from_config(
            model_config, input_opts, output_opts
        )
        self.assertIsInstance(model, keras.Model)
        self.assertEqual(len(model.inputs), 1)
        self.assertEqual(model.output_shape, (None, None, 256))
        dummy_input = np.random.random((1, 100, 1)).astype(np.float32)
        prediction = model.predict(dummy_input)
        self.assertEqual(prediction.shape, (1, 100, 256))
        self.assertIn("lstm_run", model.name)

    def test_build_arrow_model_from_config_lstm_bidirectional(self):
        """build_arrow_model_from_config with lstm and bidirectional=True builds and contains Bidirectional layer."""
        model_config = config.ArrowModelConfig.from_dict(
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
        input_opts = models.ArrowInputOptions()
        output_opts = models.ArrowOutputOptions(model_name="lstm_bidir_run")
        model = models.build_arrow_model_from_config(
            model_config, input_opts, output_opts
        )
        self.assertIsInstance(model, keras.Model)
        self.assertEqual(len(model.inputs), 1)
        self.assertEqual(model.output_shape, (None, None, 256))
        bidirectional_layers = [
            layer
            for layer in model.layers
            if isinstance(layer, keras.layers.Bidirectional)
        ]
        self.assertGreater(
            len(bidirectional_layers),
            0,
            "model should contain at least one Bidirectional layer",
        )
        dummy_input = np.random.random((1, 100, 1)).astype(np.float32)
        prediction = model.predict(dummy_input)
        self.assertEqual(prediction.shape, (1, 100, 256))

    def test_build_arrow_model_from_config_lstm_with_interval(self):
        """build_arrow_model_from_config lstm with use_interval=True has two inputs and runs forward pass."""
        model_config = config.ArrowModelConfig.from_dict(
            {
                "model_type": "lstm",
                "lstm": {"units": 64, "num_layers": 1, "dropout_rate": 0.0},
            }
        )
        input_opts = models.ArrowInputOptions(use_interval=True)
        output_opts = models.ArrowOutputOptions(model_name="")
        model = models.build_arrow_model_from_config(
            model_config, input_opts, output_opts
        )
        self.assertIsInstance(model, keras.Model)
        self.assertEqual(len(model.inputs), 2)
        input_names = [inp.name for inp in model.inputs]
        self.assertIn("timing_input", input_names)
        self.assertIn("interval_input", input_names)
        timing_input = np.random.random((1, 100, 1)).astype(np.float32)
        interval_input = np.random.random((1, 100, 1)).astype(np.float32)
        out = model.predict([timing_input, interval_input])
        self.assertEqual(out.shape, (1, 100, 256))

    def test_build_arrow_model_from_config_gru(self):
        """build_arrow_model_from_config with model_type gru produces valid model."""
        model_config = config.ArrowModelConfig.from_dict(
            {
                "model_type": "gru",
                "gru": {"units": 64, "num_layers": 1, "dropout_rate": 0.0},
            }
        )
        input_opts = models.ArrowInputOptions()
        output_opts = models.ArrowOutputOptions(model_name="gru_run")
        model = models.build_arrow_model_from_config(
            model_config, input_opts, output_opts
        )
        self.assertIsInstance(model, keras.Model)
        self.assertEqual(len(model.inputs), 1)
        self.assertEqual(model.output_shape, (None, None, 256))
        dummy_input = np.random.random((1, 100, 1)).astype(np.float32)
        prediction = model.predict(dummy_input)
        self.assertEqual(prediction.shape, (1, 100, 256))
        self.assertIn("gru_run", model.name)

    def test_build_arrow_model_from_config_gru_bidirectional(self):
        """build_arrow_model_from_config with gru and bidirectional=True builds and contains Bidirectional layer."""
        model_config = config.ArrowModelConfig.from_dict(
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
        input_opts = models.ArrowInputOptions()
        output_opts = models.ArrowOutputOptions(model_name="gru_bidir_run")
        model = models.build_arrow_model_from_config(
            model_config, input_opts, output_opts
        )
        self.assertIsInstance(model, keras.Model)
        self.assertEqual(len(model.inputs), 1)
        self.assertEqual(model.output_shape, (None, None, 256))
        bidirectional_layers = [
            layer
            for layer in model.layers
            if isinstance(layer, keras.layers.Bidirectional)
        ]
        self.assertGreater(
            len(bidirectional_layers),
            0,
            "model should contain at least one Bidirectional layer",
        )
        dummy_input = np.random.random((1, 100, 1)).astype(np.float32)
        prediction = model.predict(dummy_input)
        self.assertEqual(prediction.shape, (1, 100, 256))

    def test_build_arrow_model_from_config_gru_with_snippets(self):
        """build_arrow_model_from_config gru with snippet_half_frames has two inputs."""
        model_config = config.ArrowModelConfig.from_dict(
            {
                "model_type": "gru",
                "gru": {"units": 64, "num_layers": 1, "dropout_rate": 0.0},
            }
        )
        input_opts = models.ArrowInputOptions(snippet_half_frames=5)
        output_opts = models.ArrowOutputOptions(model_name="")
        model = models.build_arrow_model_from_config(
            model_config, input_opts, output_opts
        )
        self.assertIsInstance(model, keras.Model)
        self.assertEqual(len(model.inputs), 2)
        timing_input = np.random.random((1, 100, 1)).astype(np.float32)
        snippet_input = np.random.random((1, 100, 11, 128)).astype(np.float32)
        out = model.predict([timing_input, snippet_input])
        self.assertEqual(out.shape, (1, 100, 256))

    def test_build_arrow_model_from_config_gru_with_interval(self):
        """build_arrow_model_from_config gru with use_interval=True has two inputs and runs forward pass."""
        model_config = config.ArrowModelConfig.from_dict(
            {
                "model_type": "gru",
                "gru": {"units": 64, "num_layers": 1, "dropout_rate": 0.0},
            }
        )
        input_opts = models.ArrowInputOptions(use_interval=True)
        output_opts = models.ArrowOutputOptions(model_name="")
        model = models.build_arrow_model_from_config(
            model_config, input_opts, output_opts
        )
        self.assertIsInstance(model, keras.Model)
        self.assertEqual(len(model.inputs), 2)
        input_names = [inp.name for inp in model.inputs]
        self.assertIn("timing_input", input_names)
        self.assertIn("interval_input", input_names)
        timing_input = np.random.random((1, 100, 1)).astype(np.float32)
        interval_input = np.random.random((1, 100, 1)).astype(np.float32)
        out = model.predict([timing_input, interval_input])
        self.assertEqual(out.shape, (1, 100, 256))

    def test_build_arrow_model_from_config_tcn(self):
        """build_arrow_model_from_config with model_type tcn produces valid model."""
        model_config = config.ArrowModelConfig.from_dict(
            {
                "model_type": "tcn",
                "tcn": {
                    "filters": 32,
                    "kernel_size": 3,
                    "num_layers": 2,
                    "dilation_base": 2,
                    "dropout_rate": 0.0,
                },
            }
        )
        input_opts = models.ArrowInputOptions()
        output_opts = models.ArrowOutputOptions(model_name="tcn_run")
        model = models.build_arrow_model_from_config(
            model_config, input_opts, output_opts
        )
        self.assertIsInstance(model, keras.Model)
        self.assertEqual(len(model.inputs), 1)
        self.assertEqual(model.output_shape, (None, None, 256))
        dummy_input = np.random.random((1, 50, 1)).astype(np.float32)
        prediction = model.predict(dummy_input)
        self.assertEqual(prediction.shape, (1, 50, 256))
        self.assertIn("tcn_run", model.name)

    def test_build_arrow_model_from_config_cnn1d(self):
        """build_arrow_model_from_config with model_type cnn1d produces valid model."""
        model_config = config.ArrowModelConfig.from_dict(
            {
                "model_type": "cnn1d",
                "cnn1d": {
                    "filters": 32,
                    "kernel_sizes": [3, 3],
                    "dropout_rate": 0.0,
                },
            }
        )
        input_opts = models.ArrowInputOptions()
        output_opts = models.ArrowOutputOptions(model_name="cnn1d_run")
        model = models.build_arrow_model_from_config(
            model_config, input_opts, output_opts
        )
        self.assertIsInstance(model, keras.Model)
        self.assertEqual(len(model.inputs), 1)
        self.assertEqual(model.output_shape, (None, None, 256))
        dummy_input = np.random.random((1, 50, 1)).astype(np.float32)
        prediction = model.predict(dummy_input)
        self.assertEqual(prediction.shape, (1, 50, 256))
        self.assertIn("cnn1d_run", model.name)

    def test_build_arrow_model_from_config_unknown_model_type_raises(self):
        """build_arrow_model_from_config raises ValueError for unknown model_type."""
        with self.assertRaises(ValueError) as ctx:
            config.ArrowModelConfig.from_dict({"model_type": "unknown_arch"})
        self.assertIn("Invalid model_type: unknown_arch", str(ctx.exception))

    def test_build_arrow_model_from_config_tcn_with_interval_encoding_and_step_index(
        self,
    ):
        """build_arrow_model_from_config tcn with interval_encoding=log and use_step_index has correct inputs and output shape."""
        model_config = config.ArrowModelConfig.from_dict(
            {
                "model_type": "tcn",
                "tcn": {"filters": 32, "num_layers": 2, "dropout_rate": 0.0},
            }
        )
        input_opts = models.ArrowInputOptions(
            use_interval=True,
            interval_encoding=config.IntervalEncoding.LOG,
            use_step_index=True,
        )
        output_opts = models.ArrowOutputOptions(model_name="")
        model = models.build_arrow_model_from_config(
            model_config, input_opts, output_opts
        )
        self.assertIsInstance(model, keras.Model)
        input_names = [inp.name for inp in model.inputs]
        self.assertIn("timing_input", input_names)
        self.assertIn("interval_log_input", input_names)
        self.assertIn("step_index_input", input_names)
        batch_size, seq_len = 2, 40
        timing = np.random.random((batch_size, seq_len, 1)).astype(np.float32)
        interval_log = np.random.random((batch_size, seq_len, 1)).astype(np.float32)
        step_idx = np.random.random((batch_size, seq_len, 1)).astype(np.float32)
        out = model.predict([timing, interval_log, step_idx])
        self.assertEqual(out.shape, (batch_size, seq_len, 256))

    def test_build_arrow_model_from_config_cnn1d_with_interval_encoding_and_beat_phase(
        self,
    ):
        """build_arrow_model_from_config cnn1d with interval_encoding=multi and use_beat_phase has correct inputs and output shape. Model 'multi' uses both interval_log_input and interval_next_input."""
        model_config = config.ArrowModelConfig.from_dict(
            {
                "model_type": "cnn1d",
                "cnn1d": {"filters": 32, "kernel_sizes": [3, 3], "dropout_rate": 0.0},
            }
        )
        input_opts = models.ArrowInputOptions(
            use_interval=True,
            interval_encoding=config.IntervalEncoding.MULTI,
            use_beat_phase=True,
        )
        output_opts = models.ArrowOutputOptions(model_name="")
        model = models.build_arrow_model_from_config(
            model_config, input_opts, output_opts
        )
        self.assertIsInstance(model, keras.Model)
        input_names = [inp.name for inp in model.inputs]
        self.assertIn("timing_input", input_names)
        self.assertIn("interval_log_input", input_names)
        self.assertIn("interval_next_input", input_names)
        self.assertIn("beat_phase_input", input_names)
        batch_size, seq_len = 2, 30
        timing = np.random.random((batch_size, seq_len, 1)).astype(np.float32)
        interval_log = np.random.random((batch_size, seq_len, 1)).astype(np.float32)
        interval_next = np.random.random((batch_size, seq_len, 1)).astype(np.float32)
        beat_phase = np.random.random((batch_size, seq_len, 1)).astype(np.float32)
        out = model.predict([timing, interval_log, interval_next, beat_phase])
        self.assertEqual(out.shape, (batch_size, seq_len, 256))

    def test_build_arrow_model_from_config_gru_use_aux_interval_output_shapes(self):
        """build_arrow_model_from_config with use_aux_interval=True (GRU) returns dict with output_probabilities and aux_interval and correct shapes."""
        model_config = config.ArrowModelConfig.from_dict(
            {
                "model_type": "gru",
                "gru": {"units": 32, "num_layers": 1, "dropout_rate": 0.0},
            }
        )
        input_opts = models.ArrowInputOptions()
        output_opts = models.ArrowOutputOptions(model_name="", use_aux_interval=True)
        model = models.build_arrow_model_from_config(
            model_config, input_opts, output_opts
        )
        self.assertIsInstance(model, keras.Model)
        dummy_input = np.random.random((1, 20, 1)).astype(np.float32)
        outputs = model.predict(dummy_input)
        self.assertIsInstance(outputs, dict)
        self.assertIn("output_probabilities", outputs)
        self.assertIn("aux_interval", outputs)
        logits = outputs["output_probabilities"]
        aux_interval = outputs["aux_interval"]
        self.assertEqual(logits.shape, (1, 20, 256))
        self.assertEqual(aux_interval.shape, (1, 20, 1))

    def test_build_arrow_model_from_config_transformer_use_timing_position(self):
        """build_arrow_model_from_config transformer with use_timing_position=True builds and runs."""
        model_config = config.ArrowModelConfig.from_dict(
            {
                "model_type": "transformer",
                "transformer": {
                    "num_layers": 1,
                    "d_model": 64,
                    "num_heads": 2,
                    "ff_dim": 128,
                    "dropout_rate": 0.0,
                    "use_timing_position": True,
                },
            }
        )
        input_opts = models.ArrowInputOptions()
        output_opts = models.ArrowOutputOptions(model_name="")
        model = models.build_arrow_model_from_config(
            model_config, input_opts, output_opts
        )
        self.assertIsInstance(model, keras.Model)
        dummy_input = np.random.random((1, 25, 1)).astype(np.float32)
        out = model.predict(dummy_input)
        self.assertEqual(out.shape, (1, 25, 256))

    def test_build_arrow_model_from_config_gru_add_attention_layer(self):
        """build_arrow_model_from_config gru with add_attention_layer=True builds and runs."""
        model_config = config.ArrowModelConfig.from_dict(
            {
                "model_type": "gru",
                "gru": {
                    "units": 32,
                    "num_layers": 1,
                    "dropout_rate": 0.0,
                    "add_attention_layer": True,
                    "attention_heads": 2,
                    "attention_dim": 16,
                },
            }
        )
        input_opts = models.ArrowInputOptions()
        output_opts = models.ArrowOutputOptions(model_name="")
        model = models.build_arrow_model_from_config(
            model_config, input_opts, output_opts
        )
        self.assertIsInstance(model, keras.Model)
        dummy_input = np.random.random((1, 20, 1)).astype(np.float32)
        out = model.predict(dummy_input)
        self.assertEqual(out.shape, (1, 20, 256))


class PositionalEncodingTest(unittest.TestCase):
    def test_positional_encoding_raises_on_odd_d_model(self):
        with self.assertRaises(ValueError) as ctx:
            models.PositionalEncoding(position=100, d_model=127)
        self.assertIn("even d_model", str(ctx.exception))
        self.assertIn("sine and cosine", str(ctx.exception))

    def test_positional_encoding_accepts_even_d_model(self):
        layer = models.PositionalEncoding(position=100, d_model=128)
        dummy = np.random.random((2, 50, 128)).astype(np.float32)
        out = layer(dummy)
        self.assertEqual(out.shape, dummy.shape)


if __name__ == "__main__":
    unittest.main()
