import unittest

import torch

from stepcovnet import config, constants, models


class OnsetModelsTest(unittest.TestCase):
    """Tests for PyTorch dense onset models exposed via stepcovnet.models."""

    def test_build_unet_wavenet_model_forward(self):
        model = models.build_unet_wavenet_model(
            model_name="test_unet",
            initial_filters=8,
            depth=2,
            dilation_rates=[1, 2],
            input_features=32,
        )
        self.assertIsInstance(model, torch.nn.Module)
        self.assertEqual(model.name, "test_unet")
        x = torch.randn(2, 40, 32)
        out = model(x)
        self.assertEqual(tuple(out.shape), (2, 40, 1))
        self.assertTrue(bool((out >= 0.0).all() and (out <= 1.0).all()))

    def test_build_unet_wavenet_from_waveform_model_forward(self):
        model = models.build_unet_wavenet_from_waveform_model(
            model_name="test_wave",
            initial_filters=8,
            depth=1,
            dilation_rates=[1, 2],
            frontend_filters=16,
        )
        self.assertIsInstance(model, torch.nn.Module)
        n_frames = 12
        x = torch.randn(1, n_frames * constants.WAVEFORM_SAMPLES_PER_FRAME)
        out = model(x)
        self.assertEqual(tuple(out.shape), (1, n_frames, 1))

    def test_build_tcn_bilstm_transformer_onset_models(self):
        x = torch.randn(2, 24, 32)
        for arch in (
            config.OnsetArchitecture.UNET_WAVENET,
            config.OnsetArchitecture.TCN,
            config.OnsetArchitecture.BILSTM,
            config.OnsetArchitecture.TRANSFORMER,
        ):
            with self.subTest(arch=arch):
                cfg = config.OnsetModelConfig(
                    onset_architecture=arch,
                    initial_filters=16,
                    depth=1,
                    dilation_rates=[1, 2],
                    tcn_blocks=1,
                    recurrent_units=16,
                    transformer_layers=1,
                    transformer_heads=4,
                    input_features=32,
                )
                model = models.build_onset_dense_model(
                    cfg, model_name=f"m_{arch.value}"
                )
                out = model(x)
                self.assertEqual(tuple(out.shape), (2, 24, 1))
                self.assertTrue(bool((out >= 0.0).all() and (out <= 1.0).all()))


class ArrowModelsTest(unittest.TestCase):
    """Tests for PyTorch arrow models exposed via stepcovnet.models."""

    def test_positional_encoding_rejects_odd_d_model(self):
        with self.assertRaises(ValueError):
            models.PositionalEncoding(position=16, d_model=15)

    def test_snippet_cnn_shape(self):
        cnn = models.SnippetCNN(n_frames=5, n_mels=16, filters=12)
        x = torch.randn(2, 7, 5, 16)
        out = cnn(x)
        self.assertEqual(tuple(out.shape), (2, 7, 12))

    def test_all_arrow_architectures_from_config(self):
        in_opts = models.ArrowInputOptions(
            snippet_half_frames=0,
            use_interval=True,
            interval_encoding=config.IntervalEncoding.MULTI,
            use_step_index=True,
            use_beat_phase=True,
        )
        out_opts = models.ArrowOutputOptions(use_aux_interval=True, model_name="arr")
        feed = {
            "timing_input": torch.rand(2, 10, 1),
            "interval_log_input": torch.rand(2, 10, 1),
            "interval_next_input": torch.rand(2, 10, 1),
            "step_index_input": torch.rand(2, 10, 1),
            "beat_phase_input": torch.rand(2, 10, 1),
        }
        configs = [
            config.ArrowModelConfig.from_dict(
                {
                    "model_type": "transformer",
                    "transformer": {
                        "num_layers": 1,
                        "d_model": 16,
                        "num_heads": 2,
                        "ff_dim": 32,
                    },
                }
            ),
            config.ArrowModelConfig.from_dict(
                {"model_type": "mlp", "mlp": {"hidden_dims": [16]}}
            ),
            config.ArrowModelConfig.from_dict(
                {"model_type": "lstm", "lstm": {"units": 16, "num_layers": 1}}
            ),
            config.ArrowModelConfig.from_dict(
                {"model_type": "gru", "gru": {"units": 16, "num_layers": 1}}
            ),
            config.ArrowModelConfig.from_dict(
                {
                    "model_type": "tcn",
                    "tcn": {"filters": 16, "num_layers": 2, "dilation_base": 2},
                }
            ),
            config.ArrowModelConfig.from_dict(
                {"model_type": "cnn1d", "cnn1d": {"filters": 16, "kernel_sizes": [3]}}
            ),
        ]
        for cfg in configs:
            with self.subTest(model_type=cfg.model_type):
                model = models.build_arrow_model_from_config(cfg, in_opts, out_opts)
                self.assertIsInstance(model, torch.nn.Module)
                out = model(feed)
                self.assertIsInstance(out, dict)
                self.assertEqual(
                    tuple(out["output_probabilities"].shape),
                    (2, 10, constants.N_ARROW_TYPES),
                )
                self.assertEqual(tuple(out["aux_interval"].shape), (2, 10, 1))
                row_sums = out["output_probabilities"].sum(dim=-1)
                self.assertTrue(
                    torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-4)
                )


if __name__ == "__main__":
    unittest.main()
