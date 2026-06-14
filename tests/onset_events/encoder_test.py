import unittest

import keras
import numpy as np

from stepcovnet.onset_events import encoder

_PLAN_ENCODER_CONFIG = {
    "initial_filters": 16,
    "depth": 2,
    "dilation_rates": [1, 2, 4, 8],
    "kernel_size": 3,
    "dropout_rate": 0.0,
}


class EncoderTest(unittest.TestCase):
    def test_build_temporal_encoder_forward_shape(self):
        input_features = 32
        t_steps = 64
        batch = 2
        model = encoder.build_temporal_encoder(input_features, _PLAN_ENCODER_CONFIG)
        self.assertIsInstance(model, keras.Model)
        x = np.random.randn(batch, t_steps, input_features).astype(np.float32)
        y = model.predict(x, verbose=0)
        self.assertEqual(y.shape, (batch, t_steps, 16))

    def test_build_temporal_encoder_preserves_time_length(self):
        input_features = 8
        for t_steps in (16, 100, 257):
            with self.subTest(t_steps=t_steps):
                model = encoder.build_temporal_encoder(
                    input_features, _PLAN_ENCODER_CONFIG
                )
                x = np.random.randn(1, t_steps, input_features).astype(np.float32)
                y = model.predict(x, verbose=0)
                self.assertEqual(y.shape[1], t_steps)
