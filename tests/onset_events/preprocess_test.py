import os
import tempfile
import unittest
from unittest import mock

import numpy as np
import scipy.io.wavfile

from stepcovnet import constants
from stepcovnet.onset_events import frontend
from stepcovnet.onset_events import preprocess
from stepcovnet import datasets


def _write_wav(path: str, samples: np.ndarray, sample_rate: int) -> None:
    int_samples = np.clip(samples * 32767.0, -32768, 32767).astype(np.int16)
    scipy.io.wavfile.write(path, sample_rate, int_samples)


class PreprocessTest(unittest.TestCase):
    def test_validate_frontend_rejects_unknown(self):
        with self.assertRaises(ValueError):
            preprocess.validate_frontend("unknown")

    def test_encoder_feature_dim(self):
        self.assertEqual(preprocess.encoder_feature_dim("conv1d"), 1)
        self.assertEqual(preprocess.encoder_feature_dim("mel"), constants.N_MELS)
        self.assertEqual(
            preprocess.encoder_feature_dim("mert"), constants.MERT_HIDDEN_SIZE
        )

    def test_load_conv1d_returns_padded_waveform(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path = os.path.join(tmpdir, "clip.wav")
            sr = constants.TARGET_SR
            samples = np.zeros(int(0.2 * sr), dtype=np.float32)
            _write_wav(audio_path, samples, sr)
            max_samples = int(0.25 * sr)
            encoder_input, audio_length, duration = (
                preprocess.load_preprocessed_encoder_input(
                    audio_path,
                    frontend_name="conv1d",
                    target_sample_rate=sr,
                    max_samples=max_samples,
                    max_audio_seconds=0.25,
                )
            )
            self.assertEqual(encoder_input.shape, (max_samples,))
            self.assertEqual(int(audio_length), int(0.2 * sr))
            self.assertAlmostEqual(float(duration), 0.2, places=3)

    def test_load_mel_returns_padded_features(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path = os.path.join(tmpdir, "clip.wav")
            sr = constants.TARGET_SR
            samples = np.zeros(int(0.2 * sr), dtype=np.float32)
            _write_wav(audio_path, samples, sr)
            max_samples = int(0.25 * sr)
            fake_features = np.ones((20, constants.N_MELS), dtype=np.float32)
            with (
                mock.patch.object(
                    datasets,
                    "load_onset_features",
                    return_value=fake_features,
                    autospec=True,
                ),
                mock.patch.object(
                    datasets,
                    "normalize_onset_spectrogram",
                    side_effect=lambda x: x,
                ),
            ):
                encoder_input, _audio_length, duration = (
                    preprocess.load_preprocessed_encoder_input(
                        audio_path,
                        frontend_name="mel",
                        target_sample_rate=sr,
                        max_samples=max_samples,
                        max_audio_seconds=0.25,
                    )
                )
            max_frames = frontend.target_encoder_frames(0.25)
            self.assertEqual(encoder_input.shape, (max_frames, constants.N_MELS))
            self.assertAlmostEqual(float(duration), 0.2, places=3)


if __name__ == "__main__":
    unittest.main()
