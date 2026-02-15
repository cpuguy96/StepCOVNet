import unittest

import keras
import numpy as np

from stepcovnet import models


class TestModel(unittest.TestCase):
    def test_build_unet_wavenet_model(self):
        model_instance = models.build_unet_wavenet_model()

        self.assertIsInstance(model_instance, keras.Model)

        self.assertEqual(model_instance.input_shape, (None, None, 128))
        self.assertEqual(model_instance.output_shape, (None, None, 1))

        # Call the model and check the output shape
        dummy_input = np.random.random((1, 100, 128)).astype(np.float32)
        prediction = model_instance.predict(dummy_input)
        self.assertEqual(prediction.shape, (1, 100, 1))


if __name__ == '__main__':
    unittest.main()
