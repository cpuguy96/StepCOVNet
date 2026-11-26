"""ONNX export helpers for lightweight CPU deployment of PyTorch StepCOVNet models."""

from __future__ import annotations

import pathlib
from typing import Any

import torch
from torch import nn

from stepcovnet import constants
from stepcovnet.pytorch.core import arrow_models, onset_models
from stepcovnet.pytorch.ddc import models as ddc_models
from stepcovnet.pytorch.ddcl import models as ddcl_models
from stepcovnet.pytorch.itgpt import models as itgpt_models


class _DictInputWrapper(nn.Module):
    """Adapts a dict-input PyTorch model to positional tensor arguments for ONNX export.

    Args:
        model: Target PyTorch module accepting a dict of tensors.
        input_keys: Ordered input keys matching positional arguments.
        output_key: Optional key to extract if the module returns a dict.
    """

    def __init__(
        self,
        model: nn.Module,
        input_keys: tuple[str, ...],
        *,
        output_key: str | None = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.input_keys = input_keys
        self.output_key = output_key

    def forward(self, *args: torch.Tensor) -> torch.Tensor:
        """Pack positional args into a dict and return the primary output tensor.

        Args:
            *args: Positional tensors matching :attr:`input_keys`.

        Returns:
            Output tensor.
        """
        feed = dict(zip(self.input_keys, args, strict=True))
        out = self.model(feed)
        if isinstance(out, dict):
            key = self.output_key or next(iter(out.keys()))
            return out[key]
        return out


def export_model_onnx(
    module: nn.Module,
    example_inputs: tuple[torch.Tensor, ...],
    output_path: str | pathlib.Path,
    *,
    input_names: list[str],
    output_names: list[str],
    dynamic_axes: dict[str, dict[int, str]] | None = None,
    opset_version: int = 17,
) -> pathlib.Path:
    """Export any PyTorch module to an ONNX file on CPU.

    Args:
        module: PyTorch module in eval mode.
        example_inputs: Tuple of representative CPU tensors.
        output_path: Destination ``.onnx`` path.
        input_names: Ordered ONNX graph input names.
        output_names: Ordered ONNX graph output names.
        dynamic_axes: Optional dynamic axis mapping.
        opset_version: ONNX opset version.

    Returns:
        Written :class:`pathlib.Path`.
    """
    target = pathlib.Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    module_cpu = module.to("cpu").eval()
    inputs_cpu = tuple(t.detach().to("cpu") for t in example_inputs)
    torch.onnx.export(
        module_cpu,
        inputs_cpu,
        str(target),
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=opset_version,
        dynamo=False,
    )
    return target


def export_onset_onnx(
    model: onset_models.DenseOnsetTorchModel,
    output_path: str | pathlib.Path,
    *,
    sample_frames: int = 128,
    opset_version: int = 17,
) -> pathlib.Path:
    """Export a PyTorch dense onset model to ONNX with dynamic batch and time axes.

    Args:
        model: Trained dense onset model.
        output_path: Destination ``.onnx`` path.
        sample_frames: Dummy sequence length for tracing.
        opset_version: ONNX opset version.

    Returns:
        Written :class:`pathlib.Path`.
    """
    if isinstance(model, onset_models.UNetWaveNetWaveformModel):
        dummy = torch.randn(1, sample_frames * constants.WAVEFORM_SAMPLES_PER_FRAME)
        return export_model_onnx(
            model,
            (dummy,),
            output_path,
            input_names=["waveform"],
            output_names=["onset_probs"],
            dynamic_axes={
                "waveform": {0: "batch", 1: "samples"},
                "onset_probs": {0: "batch", 1: "frames"},
            },
            opset_version=opset_version,
        )
    in_dim = int(model.model_config()["input_features"])
    dummy = torch.randn(1, sample_frames, in_dim)
    return export_model_onnx(
        model,
        (dummy,),
        output_path,
        input_names=["input_features"],
        output_names=["onset_probs"],
        dynamic_axes={
            "input_features": {0: "batch", 1: "frames"},
            "onset_probs": {0: "batch", 1: "frames"},
        },
        opset_version=opset_version,
    )


def export_arrow_onnx(
    model: arrow_models.ArrowSequenceModel,
    output_path: str | pathlib.Path,
    *,
    sample_steps: int = 32,
    opset_version: int = 17,
) -> pathlib.Path:
    """Export a PyTorch arrow sequence model to ONNX with dynamic step length.

    Args:
        model: Trained arrow sequence model.
        output_path: Destination ``.onnx`` path.
        sample_steps: Dummy step count for tracing.
        opset_version: ONNX opset version.

    Returns:
        Written :class:`pathlib.Path`.
    """
    keys = tuple(model.required_input_names())
    dummies: list[torch.Tensor] = []
    dyn_axes: dict[str, dict[int, str]] = {
        "output_probabilities": {0: "batch", 1: "steps"}
    }
    for key in keys:
        if key == "snippet_input":
            width = 2 * int(model.input_options.snippet_half_frames) + 1
            dummies.append(torch.randn(1, sample_steps, width, constants.N_MELS))
        else:
            dummies.append(torch.randn(1, sample_steps, 1))
        dyn_axes[key] = {0: "batch", 1: "steps"}

    wrapper = _DictInputWrapper(model, keys, output_key="output_probabilities")
    return export_model_onnx(
        wrapper,
        tuple(dummies),
        output_path,
        input_names=list(keys),
        output_names=["output_probabilities"],
        dynamic_axes=dyn_axes,
        opset_version=opset_version,
    )


def export_itgpt_onnx(
    model: itgpt_models.ItgptPlacementModel,
    output_path: str | pathlib.Path,
    *,
    sample_beats: int = 64,
    opset_version: int = 17,
) -> pathlib.Path:
    """Export a PyTorch ITGPT placement model to ONNX with dynamic beat length.

    Args:
        model: Trained :class:`itgpt_models.ItgptPlacementModel`.
        output_path: Destination ``.onnx`` path.
        sample_beats: Dummy beat count for tracing.
        opset_version: ONNX opset version.

    Returns:
        Written :class:`pathlib.Path`.
    """
    cfg: dict[str, Any] = dict(model.model_config())
    if cfg["audio_feature_type"] == "mert":
        n_layers = int(cfg.get("n_mert_layers", 1))
        if n_layers > 1:
            audio = torch.randn(
                1,
                sample_beats,
                int(cfg["n_frames"]),
                n_layers,
                int(cfg["mert_feature_dim"]),
            )
        else:
            audio = torch.randn(
                1, sample_beats, int(cfg["n_frames"]), int(cfg["mert_feature_dim"])
            )
    else:
        audio = torch.randn(1, sample_beats, int(cfg["n_frames"]), 80, 3)
    difficulty = torch.zeros(1, sample_beats, 5)
    difficulty[:, :, 0] = 1.0
    meter = torch.full((1, sample_beats, 1), 0.5)
    mask = torch.ones(1, sample_beats)
    keys = ("audio", "difficulty", "meter", "mask")
    wrapper = _DictInputWrapper(model, keys)
    return export_model_onnx(
        wrapper,
        (audio, difficulty, meter, mask),
        output_path,
        input_names=list(keys),
        output_names=["slot_probs"],
        dynamic_axes={
            "audio": {0: "batch", 1: "beats"},
            "difficulty": {0: "batch", 1: "beats"},
            "meter": {0: "batch", 1: "beats"},
            "mask": {0: "batch", 1: "beats"},
            "slot_probs": {0: "batch", 1: "beats"},
        },
        opset_version=opset_version,
    )


def export_ddc_onnx(
    model: ddc_models.DdcClstmPlacementModel,
    output_path: str | pathlib.Path,
    *,
    sample_frames: int = 64,
    opset_version: int = 17,
) -> pathlib.Path:
    """Export a PyTorch DDC C-LSTM placement model to ONNX.

    Args:
        model: Trained :class:`ddc_models.DdcClstmPlacementModel`.
        output_path: Destination ``.onnx`` path.
        sample_frames: Dummy frame sequence length.
        opset_version: ONNX opset version.

    Returns:
        Written :class:`pathlib.Path`.
    """
    audio = torch.randn(1, sample_frames, 15, 80, 3)
    difficulty = torch.zeros(1, sample_frames, 5)
    difficulty[:, :, 0] = 1.0
    keys = ("audio", "difficulty")
    wrapper = _DictInputWrapper(model, keys)
    return export_model_onnx(
        wrapper,
        (audio, difficulty),
        output_path,
        input_names=list(keys),
        output_names=["onset_salience"],
        dynamic_axes={
            "audio": {0: "batch", 1: "frames"},
            "difficulty": {0: "batch", 1: "frames"},
            "onset_salience": {0: "batch", 1: "frames"},
        },
        opset_version=opset_version,
    )


def export_ddcl_onnx(
    model: ddcl_models.DdclConvLstmPlacementModel | ddcl_models.DdclMertPlacementModel,
    output_path: str | pathlib.Path,
    *,
    opset_version: int = 17,
) -> pathlib.Path:
    """Export a PyTorch DDCL ConvLSTM or MERT placement model to ONNX.

    Args:
        model: Trained DDCL placement model.
        output_path: Destination ``.onnx`` path.
        opset_version: ONNX opset version.

    Returns:
        Written :class:`pathlib.Path`.
    """
    cfg: dict[str, Any] = dict(model.model_config())
    memlen = int(cfg["memlen"])
    n_frames = int(cfg["n_frames"])
    if cfg.get("arch") == "mert":
        feat_dim = int(cfg["feature_dim"])
        audio_fwd = torch.randn(1, memlen, n_frames, feat_dim)
        audio_bwd = torch.randn(1, memlen, n_frames, feat_dim)
    else:
        audio_fwd = torch.randn(1, memlen, n_frames, 80, 3)
        audio_bwd = torch.randn(1, memlen, n_frames, 80, 3)
    stream_fwd = torch.zeros(1, memlen, 7)
    stream_bwd = torch.zeros(1, memlen, 7)
    keys = ("audio_fwd", "audio_bwd", "stream_fwd", "stream_bwd")
    wrapper = _DictInputWrapper(model, keys)
    return export_model_onnx(
        wrapper,
        (audio_fwd, audio_bwd, stream_fwd, stream_bwd),
        output_path,
        input_names=list(keys),
        output_names=["slot_probs"],
        dynamic_axes={
            "audio_fwd": {0: "batch"},
            "audio_bwd": {0: "batch"},
            "stream_fwd": {0: "batch"},
            "stream_bwd": {0: "batch"},
            "slot_probs": {0: "batch"},
        },
        opset_version=opset_version,
    )
