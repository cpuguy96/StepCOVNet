"""Tests for Keras -> PyTorch ITGPT weight transfer and numerical parity."""

from __future__ import annotations

import importlib.util
import types
import unittest

import numpy as np
import torch

from stepcovnet.pytorch.itgpt import models, parity


class _FakeVar:
    def __init__(self, array: np.ndarray) -> None:
        self._arr = np.asarray(array, dtype=np.float32)

    def numpy(self) -> np.ndarray:
        return self._arr


def _fake_dense(in_dim: int, out_dim: int, rng: np.random.Generator) -> object:
    return types.SimpleNamespace(
        kernel=_FakeVar(rng.standard_normal((in_dim, out_dim)) * 0.1),
        bias=_FakeVar(rng.standard_normal((out_dim,)) * 0.05),
    )


def _fake_conv1d(
    k: int, in_c_div_g: int, out_c: int, rng: np.random.Generator
) -> object:
    return types.SimpleNamespace(
        kernel=_FakeVar(rng.standard_normal((k, in_c_div_g, out_c)) * 0.1),
        bias=_FakeVar(rng.standard_normal((out_c,)) * 0.05),
    )


def _fake_rmsnorm(d: int, rng: np.random.Generator) -> object:
    return types.SimpleNamespace(scale=_FakeVar(1.0 + rng.standard_normal((d,)) * 0.1))


def _fake_emb(vocab: int, d: int, rng: np.random.Generator) -> object:
    return types.SimpleNamespace(
        embeddings=_FakeVar(rng.standard_normal((vocab, d)) * 0.02)
    )


def _fake_mha(d_model: int, n_heads: int, rng: np.random.Generator) -> object:
    head_dim = d_model // n_heads
    return types.SimpleNamespace(
        _query_dense=types.SimpleNamespace(
            kernel=_FakeVar(rng.standard_normal((d_model, n_heads, head_dim)) * 0.1),
            bias=_FakeVar(rng.standard_normal((n_heads, head_dim)) * 0.05),
        ),
        _key_dense=types.SimpleNamespace(
            kernel=_FakeVar(rng.standard_normal((d_model, n_heads, head_dim)) * 0.1),
            bias=_FakeVar(rng.standard_normal((n_heads, head_dim)) * 0.05),
        ),
        _value_dense=types.SimpleNamespace(
            kernel=_FakeVar(rng.standard_normal((d_model, n_heads, head_dim)) * 0.1),
            bias=_FakeVar(rng.standard_normal((n_heads, head_dim)) * 0.05),
        ),
        _output_dense=types.SimpleNamespace(
            kernel=_FakeVar(rng.standard_normal((n_heads, head_dim, d_model)) * 0.1),
            bias=_FakeVar(rng.standard_normal((d_model,)) * 0.05),
        ),
    )


def _fake_ffn(d_model: int, rng: np.random.Generator) -> object:
    return types.SimpleNamespace(
        dense1=_fake_dense(d_model, 4 * d_model, rng),
        dense2=_fake_dense(4 * d_model, d_model, rng),
    )


def _fake_hier_block(
    d_model: int,
    n_heads: int,
    frames_per_beat: int,
    beats_per_chunk: int,
    rng: np.random.Generator,
) -> object:
    return types.SimpleNamespace(
        pos_emb=_FakeVar(
            rng.standard_normal((1, frames_per_beat * beats_per_chunk, d_model)) * 0.02
        ),
        chunk_pos_emb=_fake_emb(2048, d_model, rng),
        attn_block=types.SimpleNamespace(
            attn=_fake_mha(d_model, n_heads, rng),
            ffn=_fake_ffn(d_model, rng),
            norm=_fake_rmsnorm(d_model, rng),
        ),
        norm1=_fake_rmsnorm(d_model, rng),
        norm2=_fake_rmsnorm(d_model, rng),
    )


class PyTorchItgptParityTest(unittest.TestCase):
    def test_copy_keras_weights_to_torch_shapes_and_values(self):
        rng = np.random.default_rng(42)
        d_model, n_heads, feature_dim, n_mert_layers = 32, 4, 16, 3
        torch_model = models.build_itgpt_placement_model(
            d_model=d_model,
            n_heads=n_heads,
            n_enc_layers=1,
            dropout_rate=0.0,
            max_beats=64,
            audio_feature_type="mert",
            n_frames_per_beat=24,
            feature_dim=feature_dim,
            n_mert_layers=n_mert_layers,
            early_film=True,
            cyclic_measure_pos=True,
        )
        fake_keras = types.SimpleNamespace(
            beat_cnn=None,
            beat_mert=types.SimpleNamespace(
                layer_logits=_FakeVar(np.array([0.1, -0.2, 0.3])),
                norm=_fake_rmsnorm(feature_dim, rng),
                proj0=_fake_dense(feature_dim, d_model, rng),
                conv=_fake_conv1d(3, d_model, d_model, rng),
                proj1=_fake_dense(d_model, d_model, rng),
            ),
            diff_proj=types.SimpleNamespace(
                dense0=_fake_dense(1, d_model // 2, rng),
                dense1=_fake_dense(d_model // 2, d_model, rng),
            ),
            bpm_proj=types.SimpleNamespace(
                dense0=_fake_dense(1, d_model // 2, rng),
                dense1=_fake_dense(d_model // 2, d_model, rng),
            ),
            early_film_subbeat=types.SimpleNamespace(
                hidden=_fake_dense(2 * d_model, d_model, rng),
                gamma_beta=_fake_dense(d_model, 2 * d_model, rng),
            ),
            early_film_beat=types.SimpleNamespace(
                hidden=_fake_dense(2 * d_model, d_model, rng),
                gamma_beta=_fake_dense(d_model, 2 * d_model, rng),
            ),
            level1=_fake_hier_block(d_model, n_heads, 24, 1, rng),
            level2=_fake_hier_block(d_model, n_heads, 24, 4, rng),
            level3=_fake_hier_block(d_model, n_heads, 24, 16, rng),
            level4=_fake_hier_block(d_model, n_heads, 24, 32, rng),
            frame_compress=types.SimpleNamespace(
                proj=_fake_dense(4 * d_model, 2 * d_model, rng),
                norm=_fake_rmsnorm(d_model, rng),
            ),
            level5=_fake_hier_block(d_model, n_heads, 6, 64, rng),
            beat_compress=types.SimpleNamespace(
                proj=_fake_dense(6 * d_model, 2 * d_model, rng),
                norm=_fake_rmsnorm(d_model, rng),
            ),
            level6_pos=_fake_emb(64 + 200, d_model, rng),
            bar_pos_emb=_fake_emb(4, d_model, rng),
            phrase_pos_emb=_fake_emb(16, d_model, rng),
            level6=[
                types.SimpleNamespace(
                    ln1=_fake_rmsnorm(d_model, rng),
                    attn=_fake_mha(d_model, n_heads, rng),
                    ln2=_fake_rmsnorm(d_model, rng),
                    ffn=_fake_ffn(d_model, rng),
                )
            ],
            conv_head=types.SimpleNamespace(
                norm=_fake_rmsnorm(d_model, rng),
                dw=_fake_conv1d(3, 1, d_model, rng),
                pw=_fake_conv1d(1, d_model, d_model, rng),
            ),
            onset_head=_fake_dense(d_model, 48, rng),
        )
        parity.copy_keras_weights_to_torch(fake_keras, torch_model)
        assert torch_model.beat_mert is not None
        assert torch_model.beat_mert.layer_logits is not None
        self.assertTrue(
            np.allclose(
                torch_model.beat_mert.layer_logits.detach().numpy(),
                np.array([0.1, -0.2, 0.3], dtype=np.float32),
            )
        )
        self.assertTrue(
            np.allclose(
                torch_model.onset_head.weight.detach().numpy(),
                fake_keras.onset_head.kernel.numpy().T,
            )
        )

    @unittest.skipUnless(
        importlib.util.find_spec("keras") is not None
        and importlib.util.find_spec("tensorflow") is not None,
        "Requires Keras + TensorFlow for live cross-framework numerical parity test",
    )
    def test_live_keras_torch_numerical_parity(self):
        from stepcovnet.itgpt import models as keras_models

        rng = np.random.default_rng(123)
        for feature_type, audio_shape, n_frames in (
            ("logmel", (1, 64, 32, 80, 3), 32),
            ("mert", (1, 64, 24, 16), 24),
        ):
            k_model = keras_models.build_itgpt_placement_model(
                d_model=32,
                n_heads=4,
                n_enc_layers=2,
                cnn_hidden=8,
                dropout_rate=0.0,
                max_beats=64,
                audio_feature_type=feature_type,
                n_frames_per_beat=n_frames,
                feature_dim=16,
                early_film=True,
                cyclic_measure_pos=True,
            )
            t_model = models.build_itgpt_placement_model(
                d_model=32,
                n_heads=4,
                n_enc_layers=2,
                cnn_hidden=8,
                dropout_rate=0.0,
                max_beats=64,
                audio_feature_type=feature_type,
                n_frames_per_beat=n_frames,
                feature_dim=16,
                early_film=True,
                cyclic_measure_pos=True,
            )
            inputs_np = {
                "audio": rng.standard_normal(audio_shape).astype(np.float32),
                "bpm": np.array([[140.0]], dtype=np.float32),
                "difficulty": np.array([[9.0]], dtype=np.float32),
            }
            k_out = np.asarray(k_model(inputs_np, training=False), dtype=np.float32)
            parity.copy_keras_weights_to_torch(k_model, t_model)
            t_model.eval()
            with torch.no_grad():
                t_out = (
                    t_model({k: torch.from_numpy(v) for k, v in inputs_np.items()})
                    .cpu()
                    .numpy()
                )
            max_diff = float(np.max(np.abs(k_out - t_out)))
            self.assertLess(max_diff, 1e-5, f"{feature_type} max_diff={max_diff}")
