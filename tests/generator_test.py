import pathlib
import unittest
from unittest import mock

import numpy as np
import torch

from stepcovnet import config, constants, generator, models

TEST_DATA_DIR = pathlib.Path(__file__).resolve().parent / "testdata"


class _StubOnsetModel(torch.nn.Module):
    def __init__(self, fill_value: float = 0.8) -> None:
        super().__init__()
        self.fill_value = fill_value
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.full(
            (x.shape[0], x.shape[1], 1),
            self.fill_value,
            dtype=torch.float32,
            device=x.device,
        )
        # Alternate peaks so post-processing peak picker finds peaks cleanly.
        out[:, ::2, :] = 0.0
        return out


class GeneratorHelpersTest(unittest.TestCase):
    def test_int_to_base4_string(self):
        self.assertEqual(generator._int_to_base4_string(0, min_digits=4), "0000")
        self.assertEqual(generator._int_to_base4_string(1, min_digits=4), "0001")
        self.assertEqual(generator._int_to_base4_string(64, min_digits=4), "1000")
        with self.assertRaises(ValueError):
            generator._int_to_base4_string(-1)

    def test_create_txt_mapping_skips_padding(self):
        mapping = generator._create_txt_mapping(
            onsets=[0.5, 1.0, 1.5],
            arrows=[constants.ARROW_PADDING_CLASS, 1, 64],
        )
        self.assertEqual(mapping, [("1.0", "0001"), ("1.5", "1000")])

    def test_output_data_generate_txt_output(self):
        out = generator.OutputData(
            title="Test Song",
            bpm=120,
            notes={"Challenge": [("0.5", "0001"), ("1.0", "1000")]},
        )
        text = out.generate_txt_output()
        self.assertIn("TITLE Test Song\n", text)
        self.assertIn("BPM 120\n", text)
        self.assertIn("DIFFICULTY Challenge\n", text)
        self.assertIn("0001 0.5\n", text)
        self.assertIn("1000 1.0\n", text)

    def test_post_process_predictions(self):
        probs = np.zeros(100, dtype=np.float32)
        probs[10] = 0.9
        probs[50] = 0.8
        times = generator._post_process_predictions(probs, threshold=0.5)
        self.assertEqual(len(times), 2)
        self.assertAlmostEqual(float(times[0]), 10 * constants.HOP_COEFF, places=5)

    def test_estimate_bpm_from_audio_error_when_zero(self):
        with (
            mock.patch.object(
                generator.librosa,
                "load",
                return_value=(np.zeros(22050, dtype=np.float32), constants.TARGET_SR),
                autospec=True,
            ),
            mock.patch.object(
                generator.librosa.beat,
                "beat_track",
                return_value=(np.array([0.0]), np.array([])),
                autospec=True,
            ),
            self.assertRaises(ValueError),
        ):
            generator._estimate_bpm_from_audio("dummy.wav")


class GeneratorEndToEndTest(unittest.TestCase):
    def test_generate_output_data_with_pytorch_models_and_snippets(self):
        onset_model = _StubOnsetModel(fill_value=0.9)
        arrow_model = models.build_arrow_model(
            models.ArrowInputOptions(
                snippet_half_frames=2,
                use_interval=True,
                interval_encoding=config.IntervalEncoding.MULTI,
                use_step_index=True,
                use_beat_phase=True,
            ),
            models.ArrowOutputOptions(use_aux_interval=False),
            config.TransformerArrowParams(
                num_layers=1, d_model=16, num_heads=2, ff_dim=32
            ),
        )
        fake_spec = np.random.randn(constants.N_MELS, 120).astype(np.float32)
        with mock.patch.object(
            generator.datasets,
            "audio_to_spectrogram",
            return_value=fake_spec,
            autospec=True,
        ):
            for use_pp in (False, True):
                with self.subTest(use_post_processing=use_pp):
                    output_data = generator.generate_output_data(
                        audio_path="dummy.ogg",
                        song_title="PyTorch Song",
                        bpm=128,
                        onset_model=onset_model,
                        arrow_model=arrow_model,
                        use_post_processing=use_pp,
                    )
                    self.assertEqual(output_data.title, "PyTorch Song")
                    self.assertEqual(output_data.bpm, 128)
                    self.assertIn("Challenge", output_data.notes)
                    for _t, arrow in output_data.notes["Challenge"]:
                        self.assertEqual(len(arrow), 4)
                        self.assertNotEqual(arrow, "0000")

    def test_generate_output_data_raises_when_no_onsets(self):
        onset_model = _StubOnsetModel(fill_value=0.0)
        arrow_model = models.build_arrow_model()
        fake_spec = np.random.randn(constants.N_MELS, 40).astype(np.float32)
        with (
            mock.patch.object(
                generator.datasets,
                "audio_to_spectrogram",
                return_value=fake_spec,
                autospec=True,
            ),
            self.assertRaises(ValueError),
        ):
            generator.generate_output_data(
                audio_path="dummy.ogg",
                song_title="Empty",
                bpm=120,
                onset_model=onset_model,
                arrow_model=arrow_model,
            )


if __name__ == "__main__":
    unittest.main()
