"""Shared PyTorch device, seeding, mixed-precision, EMA, and LR utilities."""

from __future__ import annotations

import contextlib
import math
import random
from collections.abc import Iterator

import numpy as np
import torch
from torch import nn


def resolve_device(device: str | torch.device | None = None) -> torch.device:
    """Resolve the target compute device.

    Prefers CUDA, then Apple MPS, then CPU when ``device`` is ``None`` or
    ``"auto"``.

    Args:
        device: Explicit device string/object or ``None`` / ``"auto"``.

    Returns:
        Resolved :class:`torch.device`.
    """
    if isinstance(device, torch.device):
        return device
    if device is not None and str(device).strip().lower() not in ("", "auto"):
        return torch.device(str(device).strip())
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch RNGs.

    Args:
        seed: Reproducibility seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_amp_dtype(
    precision: str,
    device: torch.device,
) -> torch.dtype | None:
    """Map a precision setting and device to an autocast dtype.

    Args:
        precision: One of ``"fp32"``, ``"bf16"``, ``"fp16"``, or ``"auto"``.
        device: Target compute device.

    Returns:
        ``torch.bfloat16``, ``torch.float16``, or ``None`` (fp32).

    Raises:
        ValueError: If ``precision`` is not recognized.
    """
    mode = precision.strip().lower()
    if mode not in ("fp32", "bf16", "fp16", "auto"):
        raise ValueError(
            f"precision must be one of ('fp32', 'bf16', 'fp16', 'auto'), "
            f"got {precision!r}"
        )
    if mode == "fp32":
        return None
    if device.type == "cuda":
        major, _ = torch.cuda.get_device_capability(device)
        native_bf16 = major >= 8 and torch.cuda.is_bf16_supported()
        if mode == "bf16":
            return torch.bfloat16 if native_bf16 else torch.float16
        if mode == "fp16":
            return torch.float16
        return torch.bfloat16 if native_bf16 else torch.float16
    if mode == "bf16" and device.type == "cpu":
        return torch.bfloat16
    return None


def configure_cuda_performance(
    device: torch.device,
    *,
    allow_tf32: bool = True,
) -> bool:
    """Enable TF32 matmul and cuDNN kernels on Ampere+ (SM 8.0+) CUDA GPUs.

    Args:
        device: Target compute device.
        allow_tf32: Desired TF32 setting on SM 8.0+ GPUs.

    Returns:
        ``True`` if running on an SM 8.0+ CUDA GPU and TF32 flags were applied,
        else ``False``.
    """
    if device.type != "cuda" or not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability(device)
    if major < 8:
        return False
    torch.backends.cuda.matmul.allow_tf32 = bool(allow_tf32)
    torch.backends.cudnn.allow_tf32 = bool(allow_tf32)
    return bool(allow_tf32)


@contextlib.contextmanager
def autocast_context(
    device: torch.device,
    amp_dtype: torch.dtype | None,
) -> Iterator[None]:
    """Context manager enabling autocast when ``amp_dtype`` is set.

    Args:
        device: Compute device.
        amp_dtype: Autocast dtype or ``None`` to run in full fp32.

    Yields:
        None: Control inside the autocast (or no-op) context.
    """
    if amp_dtype is None:
        yield
        return
    device_type = device.type if device.type in ("cuda", "cpu") else "cpu"
    with torch.amp.autocast(device_type=device_type, dtype=amp_dtype):  # type: ignore[attr-defined]
        yield


class ModelEMA:
    """Exponential moving average tracker for trainable model parameters.

    Maintains shadow weights initialized from the model parameters and updated
    on every optimizer step as ``shadow = momentum * shadow + (1 - momentum) * param``;
    swap shadow weights in for validation and checkpointing.

    Args:
        model: Model whose trainable parameters are tracked.
        momentum: EMA decay factor in ``(0, 1)``.

    Raises:
        ValueError: If ``momentum`` is not in ``(0, 1)``.
    """

    def __init__(self, model: nn.Module, momentum: float = 0.99) -> None:
        if not 0.0 < momentum < 1.0:
            raise ValueError(f"momentum must be in (0, 1), got {momentum}")
        self.momentum = float(momentum)
        self.shadow: dict[str, torch.Tensor] = {
            name: param.detach().clone().float()
            for name, param in model.named_parameters()
            if param.requires_grad
        }
        self._backup: dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """Update shadow weights from ``model`` parameters.

        Args:
            model: Model after an optimizer step.
        """
        one_minus = 1.0 - self.momentum
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            src = param.detach().float()
            if name not in self.shadow:
                self.shadow[name] = src.clone()
            else:
                self.shadow[name].mul_(self.momentum).add_(src, alpha=one_minus)

    @torch.no_grad()
    def apply_shadow(self, model: nn.Module) -> None:
        """Swap EMA shadow weights into ``model`` and stash live weights.

        Args:
            model: Model to swap in place.
        """
        self._backup = {}
        for name, param in model.named_parameters():
            if name in self.shadow:
                self._backup[name] = param.detach().clone()
                param.copy_(
                    self.shadow[name].to(device=param.device, dtype=param.dtype)
                )

    @torch.no_grad()
    def restore(self, model: nn.Module) -> None:
        """Restore live training weights stashed by :meth:`apply_shadow`.

        Args:
            model: Model to restore in place.
        """
        for name, param in model.named_parameters():
            if name in self._backup:
                param.copy_(self._backup[name])
        self._backup = {}

    @contextlib.contextmanager
    def average_parameters(self, model: nn.Module) -> Iterator[None]:
        """Temporarily evaluate or save ``model`` using its EMA shadow weights.

        Args:
            model: Model to swap during the context.

        Yields:
            None: Control while ``model`` holds EMA weights.
        """
        self.apply_shadow(model)
        try:
            yield
        finally:
            self.restore(model)

    def state_dict(self) -> dict[str, object]:
        """Serialize EMA momentum and shadow tensors.

        Returns:
            Serializable state dictionary.
        """
        return {
            "momentum": self.momentum,
            "shadow": {name: tensor.cpu() for name, tensor in self.shadow.items()},
        }

    def load_state_dict(
        self,
        state: dict[str, object],
        *,
        device: torch.device | None = None,
    ) -> None:
        """Load serialized EMA momentum and shadow tensors.

        Args:
            state: Dictionary produced by :meth:`state_dict`.
            device: Optional target device for shadow tensors.
        """
        momentum = state.get("momentum")
        if isinstance(momentum, (int, float)):
            self.momentum = float(momentum)
        raw_shadow = state.get("shadow")
        if isinstance(raw_shadow, dict):
            self.shadow = {
                str(name): (
                    tensor.detach().clone().float().to(device)
                    if device is not None
                    else tensor.detach().clone().float()
                )
                for name, tensor in raw_shadow.items()
                if isinstance(tensor, torch.Tensor)
            }


def compute_warmup_cosine_lr(
    step: int,
    *,
    peak_lr: float,
    min_lr: float,
    warmup_steps: int,
    total_steps: int,
) -> float:
    """Compute step-level linear warmup + cosine decay learning rate.

    Linearly ramps from ``min_lr`` to ``peak_lr`` over ``warmup_steps`` and
    then decays along a cosine curve back to ``min_lr`` over the remaining steps.

    Args:
        step: Zero-based global optimizer step index.
        peak_lr: Peak learning rate at end of warmup.
        min_lr: Floor learning rate at step 0 (when warming up) and cosine end.
        warmup_steps: Linear warmup step count (0 disables warmup).
        total_steps: Total training steps across all epochs.

    Returns:
        Scalar learning rate for ``step``.
    """
    current = max(0, int(step))
    if warmup_steps > 0 and current < warmup_steps:
        progress = current / float(warmup_steps)
        return float(min_lr + (peak_lr - min_lr) * progress)
    decay_steps = max(1, int(total_steps) - int(warmup_steps))
    decay_step = min(max(0, current - int(warmup_steps)), decay_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * decay_step / float(decay_steps)))
    return float(min_lr + (peak_lr - min_lr) * cosine)
