import unittest

import keras
import numpy as np

from stepcovnet import models


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
