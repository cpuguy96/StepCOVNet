"""Checkpoint I/O and training loops for PyTorch core dense onset & arrow models."""

from __future__ import annotations

import dataclasses
import pathlib
from typing import TYPE_CHECKING, Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from stepcovnet import config, runtime
from stepcovnet.core import arrow_models, losses, onset_models

if TYPE_CHECKING:
    from torch.utils.tensorboard import SummaryWriter


def save_core_checkpoint(
    model: nn.Module,
    path: str | pathlib.Path,
    *,
    extra: dict[str, Any] | None = None,
) -> pathlib.Path:
    """Atomically save a PyTorch core onset or arrow checkpoint.

    Args:
        model: Model exposing ``model_config()``.
        path: Destination ``.pt`` file path.
        extra: Optional metadata dictionary.

    Returns:
        Saved :class:`pathlib.Path`.
    """
    target = pathlib.Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    cfg_fn = getattr(model, "model_config", None)
    payload: dict[str, Any] = {
        "model_config": cfg_fn() if callable(cfg_fn) else {},
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
    }
    if extra:
        payload.update(extra)
    tmp = target.with_suffix(target.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(target)
    return target


def load_onset_model(
    path: str | pathlib.Path,
    *,
    device: str | torch.device | None = None,
) -> onset_models.DenseOnsetTorchModel:
    """Load a saved PyTorch dense onset checkpoint.

    Args:
        path: Path to ``.pt`` checkpoint.
        device: Optional target device.

    Returns:
        Loaded dense onset model in eval mode.

    Raises:
        ValueError: If checkpoint architecture is unrecognized.
    """
    target_device = runtime.resolve_device(device)
    ckpt = torch.load(pathlib.Path(path), map_location="cpu", weights_only=False)
    cfg = dict(ckpt["model_config"])
    arch = str(cfg.pop("arch", "unet_wavenet"))
    if "dilation_rates" in cfg:
        cfg["dilation_rates"] = tuple(cfg["dilation_rates"])
    model: onset_models.DenseOnsetTorchModel
    if arch == "tcn":
        model = onset_models.TcnOnsetModel(**cfg)
    elif arch == "bilstm":
        model = onset_models.BiLstmOnsetModel(**cfg)
    elif arch == "transformer":
        model = onset_models.TransformerOnsetModel(**cfg)
    elif arch == "unet_wavenet":
        model = onset_models.UNetWaveNetOnsetModel(**cfg)
    elif arch == "unet_wavenet_waveform":
        model = onset_models.UNetWaveNetWaveformModel(**cfg)
    else:
        raise ValueError(f"Unsupported dense onset checkpoint arch: {arch!r}")
    model.load_state_dict(ckpt["state_dict"])
    model.to(target_device)
    model.eval()
    return model


def _params_from_dict(model_type: str, data: dict[str, Any]) -> Any:
    mapping = {
        "transformer": config.TransformerArrowParams,
        "mlp": config.MLPArrowParams,
        "lstm": config.LSTMArrowParams,
        "gru": config.GRUArrowParams,
        "tcn": config.TCNArrowParams,
        "cnn1d": config.CNN1DArrowParams,
    }
    cls = mapping.get(model_type)
    if cls is None:
        raise ValueError(f"Unsupported arrow model_type: {model_type!r}")
    valid = {f.name for f in dataclasses.fields(cls)}
    return cls(**{k: v for k, v in data.items() if k in valid})


def load_arrow_model(
    path: str | pathlib.Path,
    *,
    device: str | torch.device | None = None,
) -> arrow_models.ArrowSequenceModel:
    """Load a saved PyTorch arrow sequence checkpoint.

    Args:
        path: Path to ``.pt`` checkpoint.
        device: Optional target device.

    Returns:
        Loaded :class:`arrow_models.ArrowSequenceModel` in eval mode.
    """
    target_device = runtime.resolve_device(device)
    ckpt = torch.load(pathlib.Path(path), map_location="cpu", weights_only=False)
    cfg = dict(ckpt["model_config"])
    model_type = str(cfg["model_type"])
    in_opts = arrow_models.ArrowInputOptions.from_dict(cfg["input_options"])
    out_opts = arrow_models.ArrowOutputOptions(**cfg["output_options"])
    params = _params_from_dict(model_type, cfg["params"])
    model = arrow_models.ArrowSequenceModel(
        model_type=model_type,
        input_options=in_opts,
        output_options=out_opts,
        params=params,
    )
    model.load_state_dict(ckpt["state_dict"])
    model.to(target_device)
    model.eval()
    return model


def train_dense_onset(
    model: onset_models.DenseOnsetTorchModel,
    train_loader: DataLoader[Any],
    *,
    val_loader: DataLoader[Any] | None = None,
    epochs: int = 10,
    learning_rate: float = 1e-3,
    pos_weight: float = 1.0,
    clipnorm: float = 1.0,
    output_dir: str | pathlib.Path | None = None,
    log_dir: str | pathlib.Path | None = None,
    device: str | torch.device | None = None,
    precision: str = "fp32",
    seed: int = 42,
) -> dict[str, list[float]]:
    """Train a PyTorch dense onset model on ``(features, targets[, mask])`` batches.

    Args:
        model: PyTorch dense onset model.
        train_loader: Training dataloader yielding ``(x, y)`` or ``(x, y, mask)``.
        val_loader: Optional validation dataloader.
        epochs: Number of epochs.
        learning_rate: AdamW learning rate.
        pos_weight: Positive class weight for BCE-with-logits.
        clipnorm: Gradient clip norm (``<= 0`` disables).
        output_dir: Optional directory for ``best.pt`` and ``last.pt``.
        log_dir: Optional TensorBoard log directory.
        device: Target compute device.
        precision: Precision policy (``fp32``, ``bf16``, ``fp16``, ``auto``).
        seed: Random seed.

    Returns:
        Epoch history dict with ``loss`` and optional ``val_loss``.
    """
    runtime.set_seed(seed)
    target_device = runtime.resolve_device(device)
    runtime.configure_cuda_performance(target_device)
    amp_dtype = runtime.resolve_amp_dtype(precision, target_device)
    use_scaler = amp_dtype == torch.float16 and target_device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    model.to(target_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(learning_rate), eps=1e-7)
    writer: SummaryWriter | None = (
        runtime.create_summary_writer(log_dir) if log_dir is not None else None
    )
    out_path = pathlib.Path(output_dir) if output_dir is not None else None
    if out_path is not None:
        out_path.mkdir(parents=True, exist_ok=True)

    history: dict[str, list[float]] = {"loss": [], "val_loss": []}
    best_val = float("inf")
    try:
        for epoch in range(int(epochs)):
            model.train()
            tr_sum = 0.0
            tr_n = 0
            for batch in train_loader:
                x = batch[0].to(target_device)
                y = batch[1].to(target_device)
                mask = batch[2].to(target_device) if len(batch) > 2 else None
                optimizer.zero_grad(set_to_none=True)
                with runtime.autocast_context(target_device, amp_dtype):
                    logits = model.forward_logits(x)
                    loss = losses.dense_onset_bce_with_logits_loss(
                        logits, y, pos_weight=pos_weight, mask=mask
                    )
                if use_scaler:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    if clipnorm > 0:
                        nn.utils.clip_grad_norm_(model.parameters(), clipnorm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if clipnorm > 0:
                        nn.utils.clip_grad_norm_(model.parameters(), clipnorm)
                    optimizer.step()
                tr_sum += float(loss.detach().cpu().item())
                tr_n += 1
            tr_loss = tr_sum / max(1, tr_n)
            history["loss"].append(tr_loss)
            if writer is not None:
                writer.add_scalar("loss/train", tr_loss, epoch + 1)

            if val_loader is not None:
                model.eval()
                va_sum = 0.0
                va_n = 0
                with torch.no_grad():
                    for batch in val_loader:
                        x = batch[0].to(target_device)
                        y = batch[1].to(target_device)
                        mask = batch[2].to(target_device) if len(batch) > 2 else None
                        with runtime.autocast_context(target_device, amp_dtype):
                            logits = model.forward_logits(x)
                            v_loss = losses.dense_onset_bce_with_logits_loss(
                                logits, y, pos_weight=pos_weight, mask=mask
                            )
                        va_sum += float(v_loss.detach().cpu().item())
                        va_n += 1
                va_loss = va_sum / max(1, va_n)
                history["val_loss"].append(va_loss)
                if writer is not None:
                    writer.add_scalar("loss/val", va_loss, epoch + 1)
                if out_path is not None and va_loss < best_val:
                    best_val = va_loss
                    save_core_checkpoint(
                        model,
                        out_path / "best.pt",
                        extra={"epoch": epoch + 1, "val_loss": va_loss},
                    )
        if out_path is not None:
            save_core_checkpoint(
                model, out_path / "last.pt", extra={"epoch": int(epochs)}
            )
        return history
    finally:
        if writer is not None:
            writer.close()


def train_arrow(
    model: arrow_models.ArrowSequenceModel,
    train_loader: DataLoader[Any],
    *,
    run_config: config.ArrowRunConfig | None = None,
    val_loader: DataLoader[Any] | None = None,
    epochs: int = 10,
    learning_rate: float = 1e-3,
    clipnorm: float = 1.0,
    output_dir: str | pathlib.Path | None = None,
    log_dir: str | pathlib.Path | None = None,
    device: str | torch.device | None = None,
    precision: str = "fp32",
    seed: int = 42,
) -> dict[str, list[float]]:
    """Train a PyTorch arrow sequence model using combined StepMania validity losses.

    Args:
        model: PyTorch arrow model.
        train_loader: Dataloader yielding ``(inputs, y_arrow)`` or
            ``(inputs, y_arrow, y_aux_interval, aux_mask)``.
        run_config: Optional :class:`config.ArrowRunConfig` configuring loss terms.
        val_loader: Optional validation dataloader.
        epochs: Number of epochs.
        learning_rate: AdamW learning rate.
        clipnorm: Gradient clipping norm.
        output_dir: Optional directory for ``best.pt`` and ``last.pt``.
        log_dir: Optional TensorBoard directory.
        device: Target compute device.
        precision: Mixed precision mode.
        seed: Random seed.

    Returns:
        Epoch history dict with ``loss`` and optional ``val_loss``.
    """
    runtime.set_seed(seed)
    target_device = runtime.resolve_device(device)
    runtime.configure_cuda_performance(target_device)
    amp_dtype = runtime.resolve_amp_dtype(precision, target_device)
    use_scaler = amp_dtype == torch.float16 and target_device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    cfg = run_config or config.ArrowRunConfig(
        epoch=int(epochs),
        take_count=-1,
        model_output_dir=str(output_dir or ""),
    )
    loss_fn = losses.build_arrow_combined_loss(cfg)
    aux_w = float(cfg.aux_interval_weight)

    model.to(target_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(learning_rate), eps=1e-7)
    writer: SummaryWriter | None = (
        runtime.create_summary_writer(log_dir) if log_dir is not None else None
    )
    out_path = pathlib.Path(output_dir) if output_dir is not None else None
    if out_path is not None:
        out_path.mkdir(parents=True, exist_ok=True)

    def _to_device(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: v.to(target_device) for k, v in obj.items()}
        return obj.to(target_device)

    def _step_loss(batch: Any) -> torch.Tensor:
        inputs = _to_device(batch[0])
        y_arrow = batch[1].to(target_device)
        out = model(inputs)
        if isinstance(out, dict):
            total = loss_fn(y_arrow, out["output_probabilities"])
            if len(batch) >= 3 and aux_w > 0.0:
                y_aux = batch[2].to(target_device)
                aux_mask = batch[3].to(target_device) if len(batch) >= 4 else None
                total = total + aux_w * losses.masked_mse_aux_interval(
                    y_aux, out["aux_interval"], aux_mask
                )
            return total
        return loss_fn(y_arrow, out)

    history: dict[str, list[float]] = {"loss": [], "val_loss": []}
    best_val = float("inf")
    try:
        for epoch in range(int(epochs)):
            model.train()
            tr_sum = 0.0
            tr_n = 0
            for batch in train_loader:
                optimizer.zero_grad(set_to_none=True)
                with runtime.autocast_context(target_device, amp_dtype):
                    loss = _step_loss(batch)
                if use_scaler:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    if clipnorm > 0:
                        nn.utils.clip_grad_norm_(model.parameters(), clipnorm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if clipnorm > 0:
                        nn.utils.clip_grad_norm_(model.parameters(), clipnorm)
                    optimizer.step()
                tr_sum += float(loss.detach().cpu().item())
                tr_n += 1
            tr_loss = tr_sum / max(1, tr_n)
            history["loss"].append(tr_loss)
            if writer is not None:
                writer.add_scalar("loss/train", tr_loss, epoch + 1)

            if val_loader is not None:
                model.eval()
                va_sum = 0.0
                va_n = 0
                with torch.no_grad():
                    for batch in val_loader:
                        with runtime.autocast_context(target_device, amp_dtype):
                            v_loss = _step_loss(batch)
                        va_sum += float(v_loss.detach().cpu().item())
                        va_n += 1
                va_loss = va_sum / max(1, va_n)
                history["val_loss"].append(va_loss)
                if writer is not None:
                    writer.add_scalar("loss/val", va_loss, epoch + 1)
                if out_path is not None and va_loss < best_val:
                    best_val = va_loss
                    save_core_checkpoint(
                        model,
                        out_path / "best.pt",
                        extra={"epoch": epoch + 1, "val_loss": va_loss},
                    )
        if out_path is not None:
            save_core_checkpoint(
                model, out_path / "last.pt", extra={"epoch": int(epochs)}
            )
        return history
    finally:
        if writer is not None:
            writer.close()
