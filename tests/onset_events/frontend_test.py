import unittest

import keras
import numpy as np
import tensorflow as tf

from stepcovnet import constants
from stepcovnet.onset_events import frontend


class FrontendTest(unittest.TestCase):
    def test_module_defaults(self):
        self.assertEqual(frontend.TARGET_SAMPLE_RATE, 44100)
        self.assertEqual(frontend.MAX_AUDIO_SECONDS, 300.0)
        self.assertEqual(
            frontend.max_waveform_samples(),
            int(round(300.0 * 44100)),
        )
        self.assertEqual(frontend.target_encoder_frames(300.0), 30000)

    def test_build_cached_feature_frontend_returns_model(self):
        model = frontend.build_cached_feature_frontend(
            input_features=constants.N_MELS,
            output_features=32,
            max_frames=25,
            name="test_cached_frontend",
        )
        self.assertEqual(model.name, "test_cached_frontend")
        out = model(
            np.zeros((1, 25, constants.N_MELS), dtype=np.float32), training=False
        )
        self.assertEqual(out.shape, (1, 25, 32))

    def test_build_audio_frontend_returns_model(self):
        model = frontend.build_audio_frontend(
            max_audio_seconds=1.0,
            base_filters=16,
            name="test_frontend",
        )
        self.assertIsInstance(model, keras.Model)
        self.assertEqual(model.name, "test_frontend")
        num_samples = frontend.max_waveform_samples(1.0, frontend.TARGET_SAMPLE_RATE)
        self.assertEqual(model.input_shape, (None, num_samples))

    def test_forward_pass_rank_and_dtype(self):
        max_sec = 0.5
        base_filters = 24
        model = frontend.build_audio_frontend(
            max_audio_seconds=max_sec,
            base_filters=base_filters,
        )
        num_samples = frontend.max_waveform_samples(
            max_sec, frontend.TARGET_SAMPLE_RATE
        )
        batch = 2
        audio = np.random.randn(batch, num_samples).astype(np.float32)
        out = model(audio, training=False)
        self.assertEqual(out.ndim, 3)
        self.assertEqual(out.shape[0], batch)
        self.assertEqual(out.shape[-1], base_filters)
        self.assertEqual(out.dtype, tf.float32)

    def test_default_build_matches_plan_cap(self):
        model = frontend.build_audio_frontend()
        self.assertEqual(
            model.input_shape[1],
            frontend.max_waveform_samples(),
        )
        audio = np.zeros((1, model.input_shape[1]), dtype=np.float32)
        out = model.predict(audio, verbose=0)
        self.assertEqual(out.shape[0], 1)
        self.assertEqual(out.shape[-1], frontend.DEFAULT_BASE_FILTERS)
        expected_t = frontend.target_encoder_frames(frontend.MAX_AUDIO_SECONDS)
        self.assertGreaterEqual(out.shape[1], expected_t // 2)
        self.assertLessEqual(out.shape[1], expected_t * 2)
