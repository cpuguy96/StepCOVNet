"""Unit tests for the PyTorch core dense onset, arrow, generator, and ONNX stack."""

from __future__ import annotations

import importlib.util
import pathlib
import subprocess
import sys

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from stepcovnet import config, constants
from stepcovnet.pytorch.core import (
    arrow_models,
    export_onnx,
    generator,
    losses,
    onset_models,
    trainers,
)


@pytest.mark.parametrize(
    "arch",
    [
        config.OnsetArchitecture.TCN,
        config.OnsetArchitecture.BILSTM,
        config.OnsetArchitecture.TRANSFORMER,
        config.OnsetArchitecture.UNET_WAVENET,
    ],
)
def test_dense_onset_architectures_forward_backward_and_round_trip(
    arch: config.OnsetArchitecture,
    tmp_path: pathlib.Path,
) -> None:
    """All dense onset architectures run on odd lengths, backprop, and reload cleanly."""
    mc = config.OnsetModelConfig(
        onset_architecture=arch,
        initial_filters=16,
        depth=2,
        dilation_rates=[1, 2],
        kernel_size=3,
        dropout_rate=0.0,
        tcn_blocks=2,
        recurrent_units=12,
        transformer_layers=1,
        transformer_heads=4,
        input_features=24,
    )
    model = onset_models.build_onset_dense_model(mc)
    x = torch.randn(2, 37, 24)
    logits = model.forward_logits(x)
    probs = model(x)
    assert logits.shape == (2, 37, 1)
    assert probs.shape == (2, 37, 1)
    assert torch.all((probs >= 0.0) & (probs <= 1.0))
    logits.pow(2).mean().backward()
    assert any(
        p.grad is not None and p.grad.abs().sum().item() > 0 for p in model.parameters()
    )

    model.eval()
    ckpt = trainers.save_core_checkpoint(model, tmp_path / f"{arch.value}.pt")
    loaded = trainers.load_onset_model(ckpt, device="cpu")
    with torch.no_grad():
        torch.testing.assert_close(loaded.forward_logits(x), model.forward_logits(x))


def test_waveform_onset_model_forward_and_round_trip(tmp_path: pathlib.Path) -> None:
    """UNetWaveNetWaveformModel maps raw mono audio to frame probabilities."""
    model = onset_models.build_unet_wavenet_from_waveform_model(
        initial_filters=8,
        depth=1,
        dilation_rates=[1, 2],
        frontend_filters=12,
    )
    n_frames = 15
    wav = torch.randn(2, n_frames * constants.WAVEFORM_SAMPLES_PER_FRAME)
    probs = model(wav)
    assert probs.shape == (2, n_frames, 1)
    probs.mean().backward()
    assert any(
        p.grad is not None and p.grad.abs().sum().item() > 0 for p in model.parameters()
    )

    model.eval()
    ckpt = trainers.save_core_checkpoint(model, tmp_path / "wav.pt")
    loaded = trainers.load_onset_model(ckpt, device="cpu")
    with torch.no_grad():
        torch.testing.assert_close(
            loaded.forward_logits(wav), model.forward_logits(wav)
        )


@pytest.mark.parametrize(
    ("model_type", "params"),
    [
        (
            "transformer",
            config.TransformerArrowParams(
                num_layers=1, d_model=32, num_heads=4, ff_dim=64, dropout_rate=0.0
            ),
        ),
        (
            "mlp",
            config.MLPArrowParams(hidden_dims=[32, 16], dropout_rate=0.0),
        ),
        (
            "lstm",
            config.LSTMArrowParams(
                units=16, num_layers=1, bidirectional=True, dropout_rate=0.0
            ),
        ),
        (
            "gru",
            config.GRUArrowParams(
                units=16,
                num_layers=1,
                bidirectional=False,
                dropout_rate=0.0,
                add_attention_layer=True,
                attention_heads=4,
                attention_dim=32,
            ),
        ),
        (
            "tcn",
            config.TCNArrowParams(
                filters=16,
                kernel_size=3,
                num_layers=2,
                dilation_base=2,
                dropout_rate=0.0,
            ),
        ),
        (
            "cnn1d",
            config.CNN1DArrowParams(filters=16, kernel_sizes=[3, 5], dropout_rate=0.0),
        ),
    ],
)
def test_all_arrow_backbones_and_checkpoint_round_trip(
    model_type: str,
    params: object,
    tmp_path: pathlib.Path,
) -> None:
    """All 6 arrow backbones run with multi-input features, aux_interval, and reload."""
    in_opts = arrow_models.ArrowInputOptions(
        snippet_half_frames=2,
        use_interval=True,
        interval_encoding=config.IntervalEncoding.MULTI,
        use_step_index=True,
        use_beat_phase=True,
    )
    out_opts = arrow_models.ArrowOutputOptions(use_aux_interval=True, model_name="t")
    model = arrow_models.ArrowSequenceModel(
        model_type=model_type,
        input_options=in_opts,
        output_options=out_opts,
        params=params,
    )
    feed = {
        "timing_input": torch.rand(2, 10, 1),
        "interval_log_input": torch.rand(2, 10, 1),
        "interval_next_input": torch.rand(2, 10, 1),
        "step_index_input": torch.rand(2, 10, 1),
        "beat_phase_input": torch.rand(2, 10, 1),
        "snippet_input": torch.randn(2, 10, 5, constants.N_MELS),
    }
    out = model(feed)
    assert isinstance(out, dict)
    probs = out["output_probabilities"]
    aux = out["aux_interval"]
    assert probs.shape == (2, 10, constants.N_ARROW_TYPES)
    assert aux.shape == (2, 10, 1)
    torch.testing.assert_close(
        probs.sum(dim=-1), torch.ones(2, 10), atol=1e-5, rtol=1e-5
    )

    loss = probs.pow(2).mean() + aux.pow(2).mean()
    loss.backward()
    assert any(
        p.grad is not None and p.grad.abs().sum().item() > 0 for p in model.parameters()
    )

    model.eval()
    ckpt = trainers.save_core_checkpoint(model, tmp_path / f"arrow_{model_type}.pt")
    loaded = trainers.load_arrow_model(ckpt, device="cpu")
    with torch.no_grad():
        out_a = model(feed)
        out_b = loaded(feed)
        assert isinstance(out_a, dict) and isinstance(out_b, dict)
        torch.testing.assert_close(
            out_b["output_probabilities"], out_a["output_probabilities"]
        )


def test_chart_validity_and_arrow_combined_losses() -> None:
    """Validity loss is ~0 for valid holds and positive for orphaned tails/unterminated holds."""
    # Base-4 column codes: col0 hold start = 2*64 = 128, col0 hold tail = 3*64 = 192, tap = 64.
    valid_seq = torch.tensor([[128, 192, 64, constants.ARROW_PADDING_CLASS]])
    valid_probs = torch.nn.functional.one_hot(
        valid_seq.clamp(0, constants.N_ARROW_TYPES - 1),
        num_classes=constants.N_ARROW_TYPES,
    ).float()
    loss_valid = losses.chart_validity_auxiliary_loss(valid_seq, valid_probs)
    assert loss_valid.item() < 0.02

    invalid_seq = torch.tensor([[192, 128, 64, constants.ARROW_PADDING_CLASS]])
    invalid_probs = torch.nn.functional.one_hot(
        invalid_seq.clamp(0, constants.N_ARROW_TYPES - 1),
        num_classes=constants.N_ARROW_TYPES,
    ).float()
    loss_invalid = losses.chart_validity_auxiliary_loss(invalid_seq, invalid_probs)
    assert loss_invalid.item() > loss_valid.item() + 0.05

    for loss_type, rej_thr in (("focal", None), ("crossentropy", 0.8)):
        rc = config.ArrowRunConfig(
            epoch=1,
            take_count=-1,
            model_output_dir="",
            loss_type=loss_type,
            label_smoothing=0.05,
            focal_gamma=2.0,
            chart_validity_aux_weight=0.2,
            diversity_aux_weight=0.1,
            chart_validity_rejection_threshold=rej_thr,
        )
        comb = losses.build_arrow_combined_loss(rc)
        val = comb(valid_seq, valid_probs)
        assert torch.isfinite(val)


def test_core_train_loops_and_generator_smoke(tmp_path: pathlib.Path) -> None:
    """Dense onset & arrow trainers run on CPU and feed the PyTorch chart generator."""
    onset_model = onset_models.build_tcn_onset_model(
        initial_filters=8,
        dilation_rates=[1, 2],
        tcn_blocks=1,
        input_features=constants.N_MELS,
    )
    x_onset = torch.randn(4, 30, constants.N_MELS)
    y_onset = torch.zeros(4, 30, 1)
    y_onset[:, 5::6, 0] = 1.0
    onset_loader = DataLoader(TensorDataset(x_onset, y_onset), batch_size=2)
    h_onset = trainers.train_dense_onset(
        onset_model,
        onset_loader,
        val_loader=onset_loader,
        epochs=2,
        learning_rate=1e-2,
        output_dir=tmp_path / "onset_out",
        device="cpu",
    )
    assert len(h_onset["loss"]) == 2
    assert (tmp_path / "onset_out" / "best.pt").is_file()

    in_opts = arrow_models.ArrowInputOptions(
        snippet_half_frames=1,
        use_interval=True,
        interval_encoding=config.IntervalEncoding.DEFAULT,
        use_step_index=True,
        use_beat_phase=True,
    )
    arrow_model = arrow_models.build_arrow_model(
        in_opts,
        arrow_models.ArrowOutputOptions(use_aux_interval=False),
        config.TransformerArrowParams(
            num_layers=1, d_model=16, num_heads=2, ff_dim=32, dropout_rate=0.0
        ),
    )
    arrow_batch_in = {
        "timing_input": torch.rand(2, 8, 1),
        "interval_input": torch.rand(2, 8, 1),
        "step_index_input": torch.rand(2, 8, 1),
        "beat_phase_input": torch.rand(2, 8, 1),
        "snippet_input": torch.randn(2, 8, 3, constants.N_MELS),
    }
    y_arrow = torch.randint(1, 20, (2, 8))
    h_arrow = trainers.train_arrow(
        arrow_model,
        [(arrow_batch_in, y_arrow)],  # type: ignore[arg-type]
        val_loader=[(arrow_batch_in, y_arrow)],  # type: ignore[arg-type]
        epochs=2,
        learning_rate=1e-2,
        output_dir=tmp_path / "arrow_out",
        device="cpu",
    )
    assert len(h_arrow["loss"]) == 2
    assert (tmp_path / "arrow_out" / "best.pt").is_file()

    # Force high onset probabilities so generator always emits onsets in smoke test.
    with torch.no_grad():
        onset_model.head.proj2.bias.fill_(5.0)
    spec = (
        np.random.default_rng(0)
        .standard_normal((60, constants.N_MELS))
        .astype(np.float32)
    )
    out = generator.generate_output_data_from_spec(
        normalized_spec=spec,
        song_title="TestSong",
        bpm=128,
        onset_model=onset_model,
        arrow_model=arrow_model,
        use_post_processing=False,
        threshold=0.5,
        device=torch.device("cpu"),
    )
    txt = out.generate_txt_output()
    assert "TITLE TestSong" in txt
    assert "BPM 128" in txt
    assert "DIFFICULTY Challenge" in txt


def test_onnx_export_if_available(tmp_path: pathlib.Path) -> None:
    """Export dense onset and arrow models to ONNX when onnx is installed."""
    if importlib.util.find_spec("onnx") is None:
        pytest.skip("onnx package not installed in local venv")
    onset_m = onset_models.build_tcn_onset_model(
        initial_filters=8, dilation_rates=[1, 2], tcn_blocks=1, input_features=16
    )
    p1 = export_onnx.export_onset_onnx(
        onset_m, tmp_path / "onset.onnx", sample_frames=16
    )
    assert p1.is_file() and p1.stat().st_size > 0

    arrow_m = arrow_models.build_arrow_model(
        arrow_models.ArrowInputOptions(),
        arrow_models.ArrowOutputOptions(),
        config.TransformerArrowParams(num_layers=1, d_model=16, num_heads=2, ff_dim=32),
    )
    p2 = export_onnx.export_arrow_onnx(arrow_m, tmp_path / "arrow.onnx", sample_steps=8)
    assert p2.is_file() and p2.stat().st_size > 0


def test_core_no_tensorflow_imported() -> None:
    """Importing stepcovnet.pytorch.core never imports TensorFlow or Keras."""
    code = (
        "import sys; "
        "import stepcovnet.pytorch.core; "
        "assert 'tensorflow' not in sys.modules; "
        "assert 'keras' not in sys.modules"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
