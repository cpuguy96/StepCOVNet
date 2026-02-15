import unittest

import keras
import numpy as np

from stepcovnet import model


class TestModel(unittest.TestCase):
    def test_build_unet_wavenet_model(self):
        input_shape = (100, 128)
        model_instance = model.build_unet_wavenet_model(input_shape=input_shape)
        self.assertIsInstance(model_instance, keras.Model)

        # Check input shape
        self.assertEqual(model_instance.input_shape, (None, 100, 128))

        # Check output shape
        # The output should have the same time steps as input and 1 channel
        self.assertEqual(model_instance.output_shape, (None, 100, 1))

        # Check prediction shape
        dummy_input = np.random.random((1, 100, 128)).astype(np.float32)
        prediction = model_instance.predict(dummy_input)
        self.assertEqual(prediction.shape, (1, 100, 1))

    def test_model_prediction_shape(self):
        input_shape = (100, 128)
        model_instance = model.build_unet_wavenet_model(input_shape=input_shape)

        # Create dummy input data
        dummy_input = np.random.random((1, 100, 128)).astype(np.float32)

        # Get prediction
        prediction = model_instance.predict(dummy_input)

        # Check prediction shape
        self.assertEqual(prediction.shape, (1, 100, 1))


if __name__ == '__main__':
    unittest.main()
