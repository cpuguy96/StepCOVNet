import unittest

import keras
import numpy as np

from stepcovnet import model


class TestModel(unittest.TestCase):
    def test_build_unet_wavenet_model(self):
        input_shape = (100, 128)

        model_instance = model.build_unet_wavenet_model(input_shape=input_shape)

        self.assertIsInstance(model_instance, keras.Model)

        self.assertEqual(model_instance.input_shape, (None, 100, 128))
        self.assertEqual(model_instance.output_shape, (None, 100, 1))

        # Call the model and check the output shape
        dummy_input = np.random.random((1, 100, 128)).astype(np.float32)
        prediction = model_instance.predict(dummy_input)
        self.assertEqual(prediction.shape, (1, 100, 1))


if __name__ == '__main__':
    unittest.main()
