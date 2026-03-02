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
        model_instance = models.build_arrow_model()

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
        model = models.build_arrow_model(model_name="")
        self.assertEqual(model.name, "stepcovnet_ARROW")

    def test_build_arrow_model_custom_name(self):
        """Arrow model name includes custom model_name suffix."""
        model = models.build_arrow_model(model_name="my_arrow_run")
        self.assertEqual(model.name, "stepcovnet_ARROW-my_arrow_run")

    def test_build_arrow_model_with_audio_snippets(self):
        """Arrow model with snippet_half_frames > 0 has two inputs and runs forward pass."""
        model = models.build_arrow_model(
            snippet_half_frames=5,
        )
        self.assertIsInstance(model, keras.Model)
        self.assertEqual(len(model.inputs), 2)
        timing_input = np.random.random((1, 100, 1)).astype(np.float32)
        snippet_input = np.random.random((1, 100, 11, 128)).astype(np.float32)
        out = model.predict([timing_input, snippet_input])
        self.assertEqual(out.shape, (1, 100, 256))

    def test_build_arrow_model_from_config_transformer(self):
        """build_arrow_model_from_config with model_type transformer matches build_arrow_model."""
        model_config = config.ArrowModelConfig.from_dict(
            {
                "model_type": "transformer",
                "transformer": {"num_layers": 1, "d_model": 128},
            }
        )
        model = models.build_arrow_model_from_config(model_config, model_name="")
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
        model = models.build_arrow_model_from_config(model_config, model_name="mlp_run")
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
                "snippet_half_frames": 5,
                "mlp": {"hidden_dims": [128], "dropout_rate": 0.0},
            }
        )
        model = models.build_arrow_model_from_config(model_config, model_name="")
        self.assertIsInstance(model, keras.Model)
        self.assertEqual(len(model.inputs), 2)
        timing_input = np.random.random((1, 100, 1)).astype(np.float32)
        snippet_input = np.random.random((1, 100, 11, 128)).astype(np.float32)
        out = model.predict([timing_input, snippet_input])
        self.assertEqual(out.shape, (1, 100, 256))

    def test_build_arrow_model_from_config_lstm(self):
        """build_arrow_model_from_config with model_type lstm produces valid model."""
        model_config = config.ArrowModelConfig.from_dict(
            {
                "model_type": "lstm",
                "lstm": {"units": 64, "num_layers": 1, "dropout_rate": 0.0},
            }
        )
        model = models.build_arrow_model_from_config(
            model_config, model_name="lstm_run"
        )
        self.assertIsInstance(model, keras.Model)
        self.assertEqual(len(model.inputs), 1)
        self.assertEqual(model.output_shape, (None, None, 256))
        dummy_input = np.random.random((1, 100, 1)).astype(np.float32)
        prediction = model.predict(dummy_input)
        self.assertEqual(prediction.shape, (1, 100, 256))
        self.assertIn("lstm_run", model.name)

    def test_build_arrow_model_from_config_unknown_model_type_raises(self):
        """build_arrow_model_from_config raises ValueError for unknown model_type."""
        model_config = config.ArrowModelConfig.from_dict({"model_type": "unknown_arch"})
        with self.assertRaises(ValueError) as ctx:
            models.build_arrow_model_from_config(model_config, model_name="")
        self.assertIn("unknown_arch", str(ctx.exception))
        self.assertIn("transformer", str(ctx.exception))
        self.assertIn("mlp", str(ctx.exception))
        self.assertIn("lstm", str(ctx.exception))


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
