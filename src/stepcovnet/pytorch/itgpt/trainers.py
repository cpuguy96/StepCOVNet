"""Train a PyTorch ITGPT hierarchical placement model (`omalley2026itgpt`)."""

from __future__ import annotations

import contextlib
import datetime
import json
import pathlib
import shutil
import time
from typing import Any

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from stepcovnet.ddcl import constants as ddcl_constants
from stepcovnet.itgpt import config, constants
from stepcovnet.pytorch import runtime
from stepcovnet.pytorch.itgpt import datasets, evaluation, models

BEST_CHECKPOINT_FILENAME = "best.pt"
BEST_SLOT_F1_CHECKPOINT_FILENAME = "best_slot_f1.pt"
LAST_CHECKPOINT_FILENAME = "last.pt"
BACKUP_DIRNAME = "backup"
BACKUP_FILENAME = "resume.pt"
LAST_EVAL_FILENAME = "eval_val_slot48.json"
BEST_EVAL_FILENAME = "eval_val_slot48_best.json"
BEST_SLOT_F1_EVAL_FILENAME = "eval_val_slot48_best_f1.json"


def _unwrap_model(model: nn.Module) -> models.ItgptPlacementModel:
    """Return the underlying :class:`ItgptPlacementModel` if wrapped by ``torch.compile``.

    Args:
        model: Raw or compiled model.

    Returns:
        Unwrapped :class:`ItgptPlacementModel`.
    """
    orig = getattr(model, "_orig_mod", model)
    return orig  # type: ignore[return-value]


def save_checkpoint(
    model: nn.Module,
    path: str | pathlib.Path,
    *,
    extra: dict[str, Any] | None = None,
) -> pathlib.Path:
    """Save a PyTorch ITGPT checkpoint with constructor config and weights.

    Args:
        model: Trained (or partially trained) :class:`ItgptPlacementModel`.
        path: Destination ``.pt`` file path.
        extra: Optional metadata (epoch, optimizer state, EMA state, metrics).

    Returns:
        Saved :class:`pathlib.Path`.
    """
    unwrapped = _unwrap_model(model)
    target = pathlib.Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "model_config": unwrapped.model_config(),
        "state_dict": {
            key: tensor.detach().cpu() for key, tensor in unwrapped.state_dict().items()
        },
    }
    if extra:
        payload.update(extra)
    tmp_path = target.with_suffix(target.suffix + ".tmp")
    torch.save(payload, tmp_path)
    tmp_path.replace(target)
    return target


def load_placement_model(
    path: str | pathlib.Path,
    *,
    device: str | torch.device | None = None,
) -> models.ItgptPlacementModel:
    """Load a saved PyTorch ITGPT placement model from a ``.pt`` checkpoint.

    Args:
        path: Path to a ``.pt`` checkpoint written by :func:`save_checkpoint`.
        device: Optional target device (defaults to auto-detected device).

    Returns:
        Loaded :class:`ItgptPlacementModel` in eval mode.
    """
    target_device = runtime.resolve_device(device)
    checkpoint = torch.load(
        pathlib.Path(path),
        map_location="cpu",
        weights_only=False,
    )
    model_cfg = dict(checkpoint["model_config"])
    model = models.build_itgpt_placement_model(**model_cfg)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(target_device)
    model.eval()
    return model


def build_optimizer(
    model: nn.Module,
    run_config: config.ItgptRunConfig,
) -> torch.optim.Optimizer:
    """Build AdamW (when ``weight_decay > 0``) or Adam matching ITGPT defaults.

    Uses ``eps=1e-7`` for numerical stability in ``Adam`` / ``AdamW``.

    Args:
        model: Model whose parameters will be optimized.
        run_config: Optimizer settings.

    Returns:
        Configured PyTorch optimizer.
    """
    params = [param for param in model.parameters() if param.requires_grad]
    initial_lr = (
        float(run_config.min_learning_rate)
        if run_config.lr_schedule == constants.LR_SCHEDULE_COSINE
        and run_config.warmup_epochs > 0
        else float(run_config.learning_rate)
    )
    if run_config.weight_decay > 0.0:
        return torch.optim.AdamW(
            params,
            lr=initial_lr,
            weight_decay=float(run_config.weight_decay),
            eps=1e-7,
        )
    return torch.optim.Adam(
        params,
        lr=initial_lr,
        eps=1e-7,
    )


def learning_rate_for_step(
    run_config: config.ItgptRunConfig,
    step: int,
) -> float:
    """Return the learning rate for optimizer step ``step``.

    Args:
        run_config: Optimizer and schedule config.
        step: Zero-based global step index.

    Returns:
        Scalar learning rate.
    """
    if run_config.lr_schedule == constants.LR_SCHEDULE_CONSTANT:
        return float(run_config.learning_rate)
    steps_per_epoch = max(1, int(run_config.steps_per_epoch))
    total_steps = int(run_config.epoch) * steps_per_epoch
    warmup_steps = int(run_config.warmup_epochs) * steps_per_epoch
    return runtime.compute_warmup_cosine_lr(
        step,
        peak_lr=float(run_config.learning_rate),
        min_lr=float(run_config.min_learning_rate),
        warmup_steps=warmup_steps,
        total_steps=total_steps,
    )


def _set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    """Update learning rate across all parameter groups.

    Args:
        optimizer: PyTorch optimizer.
        lr: New scalar learning rate.
    """
    for group in optimizer.param_groups:
        group["lr"] = float(lr)


def tensorboard_run_log_dir(callback_root_dir: str, model_name: str) -> pathlib.Path:
    """Return a per-run TensorBoard directory under the stage ``logs/`` tree.

    Args:
        callback_root_dir: Stage root shared by ITGPT placement runs.
        model_name: Run label.

    Returns:
        ``{callback_root_dir}/logs/{YYYYMMDD-HHMMSS}-{model_name}``.
    """
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    safe_name = str(model_name).strip() or "itgpt_placement"
    log_dir = pathlib.Path(callback_root_dir) / "logs" / f"{stamp}-{safe_name}"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def _move_batch(
    batch: tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor],
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    """Move a packed chart batch onto ``device``.

    Args:
        batch: ``(inputs, slots, mask)`` CPU tensors.
        device: Target compute device.

    Returns:
        Same structure on ``device``.
    """
    non_blocking = device.type == "cuda"
    inputs, slots, mask = batch
    inputs_dev = {
        key: tensor.to(device, non_blocking=non_blocking)
        for key, tensor in inputs.items()
    }
    return (
        inputs_dev,
        slots.to(device, non_blocking=non_blocking),
        mask.to(device, non_blocking=non_blocking),
    )


def _write_eval_report(
    report: evaluation.Slot48EvalReport,
    path: pathlib.Path,
    *,
    weights: str,
) -> None:
    """Write ``M-slot48`` JSON report and echo it to stdout.

    Args:
        report: Pooled scores.
        path: Destination JSON file.
        weights: Checkpoint tag (``last``, ``best``, ``best_slot_f1``).
    """
    payload = evaluation.report_as_dict(report, weights=weights)
    text = json.dumps(payload, indent=2)
    path.write_text(text + "\n", encoding="utf-8")
    print(text)


def build_model(
    experiment: config.ItgptExperimentConfig,
    *,
    feature_dim: int = ddcl_constants.MERT_FEATURE_DIM,
    n_mert_layers: int = 1,
) -> models.ItgptPlacementModel:
    """Construct an :class:`ItgptPlacementModel` from an experiment config.

    Args:
        experiment: Placement experiment config.
        feature_dim: Per-frame feature width when ``audio_feature_type == "mert"``.
        n_mert_layers: Stacked MERT layer count (``1`` disables layer mixing).

    Returns:
        Uncompiled :class:`ItgptPlacementModel`.
    """
    return models.build_itgpt_placement_model(
        d_model=experiment.model.d_model,
        n_heads=experiment.model.n_heads,
        n_enc_layers=experiment.model.n_enc_layers,
        cnn_hidden=experiment.model.cnn_hidden,
        dropout_rate=experiment.model.dropout_rate,
        max_beats=experiment.dataset.max_beats,
        audio_feature_type=experiment.dataset.audio_feature_type,
        n_frames_per_beat=experiment.dataset.n_frames_per_beat,
        feature_dim=feature_dim,
        n_mert_layers=n_mert_layers,
        early_film=experiment.model.early_film,
        cyclic_measure_pos=experiment.model.cyclic_measure_pos,
        continuous_beat_conv=experiment.model.continuous_beat_conv,
    )


def compile_placement_blocks(model: models.ItgptPlacementModel) -> None:
    """Compile repeated ITGPT transformer blocks in-place with dynamic shapes.

    Compiling ``beat_mert`` / ``beat_cnn``, ``level1``–``level5``, and each
    ``level6`` encoder block individually avoids unrolling all 13 blocks into a
    single monolithic FX graph (~4.5x faster first-step compile with zero
    beat-length recompiles) while preserving ``state_dict()`` keys for
    checkpointing and EMA.

    Args:
        model: Placement model whose submodules will be compiled in-place.
    """
    if model.beat_mert is not None:
        model.beat_mert.compile(dynamic=True)
    if model.beat_cnn is not None:
        model.beat_cnn.compile(dynamic=True)
    for level_name in ("level1", "level2", "level3", "level4", "level5"):
        getattr(model, level_name).compile(dynamic=True)
    for block in model.level6:
        block.compile(dynamic=True)


def train_placement(
    experiment: config.ItgptExperimentConfig,
    *,
    device: str | torch.device | None = None,
    precision: str | None = None,
    compile_model: bool = False,
) -> models.ItgptPlacementModel:
    """Load beat-grid charts, train PyTorch ITGPT placement, and save artifacts.

    Args:
        experiment: Placement experiment config (same JSON schema as TF ITGPT).
        device: Optional device override (defaults to CUDA -> MPS -> CPU).
        precision: Optional precision override (``"fp32"``, ``"bf16"``,
            ``"fp16"``, ``"auto"``). Defaults to ``"auto"`` when
            ``experiment.run.mixed_precision`` is True, else ``"fp32"``.
        compile_model: If True (or ``experiment.run.jit_compile`` is True),
            compile the model's transformer and beat-encoder blocks in-place
            with :func:`compile_placement_blocks`.

    Returns:
        Trained :class:`ItgptPlacementModel` (with EMA weights applied if
        ``experiment.run.use_ema`` is True).

    Raises:
        ValueError: If the train or val split is empty.
    """
    runtime.set_seed(experiment.run.seed)
    target_device = runtime.resolve_device(device)
    tf32_enabled = runtime.configure_cuda_performance(target_device)
    effective_precision = (
        precision
        if precision is not None
        else ("auto" if experiment.run.mixed_precision else "fp32")
    )
    amp_dtype = runtime.resolve_amp_dtype(effective_precision, target_device)
    use_grad_scaler = target_device.type == "cuda" and amp_dtype == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_grad_scaler)  # type: ignore[attr-defined]
    jit_enabled = bool(compile_model or experiment.run.jit_compile)

    print(
        f"PyTorch ITGPT training: device={target_device}, "
        f"precision={effective_precision} (amp_dtype={amp_dtype}), "
        f"tf32={tf32_enabled}, jit_compile={jit_enabled}, "
        f"use_ema={experiment.run.use_ema}",
        flush=True,
    )

    train_charts = datasets.load_split_charts(experiment.dataset, "train")
    val_charts = datasets.load_split_charts(experiment.dataset, "val")
    if not train_charts:
        raise ValueError("train split is empty")
    if not val_charts:
        raise ValueError("val split is empty")

    beat_shape = train_charts[0].beat_audio.shape
    if experiment.dataset.audio_feature_type == ddcl_constants.AUDIO_FEATURE_MERT:
        feature_dim = int(beat_shape[-1])
        n_mert_layers = int(beat_shape[-2]) if len(beat_shape) == 4 else 1
    else:
        feature_dim = ddcl_constants.MERT_FEATURE_DIM
        n_mert_layers = 1

    raw_model = build_model(
        experiment,
        feature_dim=feature_dim,
        n_mert_layers=n_mert_layers,
    ).to(target_device)

    if jit_enabled:
        compile_placement_blocks(raw_model)
    forward_logits_fn: Any = raw_model.forward_logits

    criterion = models.ItgptPlacementLoss(
        focal_gamma=experiment.run.focal_gamma,
        grid_weight_24th=experiment.run.grid_weight_24th,
        label_smoothing=experiment.run.label_smoothing,
    ).to(target_device)
    optimizer = build_optimizer(raw_model, experiment.run)
    ema = (
        runtime.ModelEMA(raw_model, momentum=experiment.run.ema_momentum)
        if experiment.run.use_ema
        else None
    )

    output_dir = pathlib.Path(experiment.run.model_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    backup_dir = output_dir / BACKUP_DIRNAME
    backup_path = backup_dir / BACKUP_FILENAME
    if not experiment.run.resume and backup_dir.exists():
        shutil.rmtree(backup_dir)

    best_path = output_dir / BEST_CHECKPOINT_FILENAME
    best_slot_f1_path = output_dir / BEST_SLOT_F1_CHECKPOINT_FILENAME
    last_path = output_dir / LAST_CHECKPOINT_FILENAME

    start_epoch = 0
    global_step = 0
    best_val_loss: float | None = None
    best_slot_f1: float | None = None
    best_slot_f1_epoch: int | None = None

    if experiment.run.resume and backup_path.is_file():
        resume_state = torch.load(backup_path, map_location="cpu", weights_only=False)
        raw_model.load_state_dict(resume_state["state_dict"])
        optimizer.load_state_dict(resume_state["optimizer_state_dict"])
        if ema is not None and "ema_state_dict" in resume_state:
            ema.load_state_dict(resume_state["ema_state_dict"], device=target_device)
        start_epoch = int(resume_state.get("completed_epoch", 0))
        global_step = int(resume_state.get("global_step", 0))
        best_val_loss = resume_state.get("best_val_loss")
        best_slot_f1 = resume_state.get("best_slot_f1")
        best_slot_f1_epoch = resume_state.get("best_slot_f1_epoch")
        print(
            f"Resumed PyTorch ITGPT training from epoch {start_epoch} "
            f"(global_step={global_step})",
            flush=True,
        )

    writer: SummaryWriter | None = None
    if experiment.run.callback_root_dir:
        log_dir = tensorboard_run_log_dir(
            experiment.run.callback_root_dir,
            experiment.run.model_name,
        )
        writer = SummaryWriter(log_dir=str(log_dir))

    steps_per_epoch = max(1, int(experiment.run.steps_per_epoch))
    validation_steps = max(1, int(experiment.run.validation_steps))
    total_epochs = int(experiment.run.epoch)
    grad_accum = max(1, int(experiment.run.gradient_accumulation_steps))
    clipnorm = float(experiment.run.clipnorm)
    pin_memory = target_device.type == "cuda"

    train_iter = iter(
        datasets.build_dataloader(
            train_charts,
            max_beats=experiment.dataset.max_beats,
            seed=experiment.run.seed + start_epoch,
            max_bar_shift=experiment.run.max_bar_shift,
            pin_memory=pin_memory,
        )
    )
    val_loader = datasets.build_dataloader(
        val_charts,
        max_beats=experiment.dataset.max_beats,
        seed=experiment.run.seed + 1,
        pin_memory=pin_memory,
    )

    try:
        for epoch_idx in range(start_epoch, total_epochs):
            epoch_num = epoch_idx + 1
            t0 = time.perf_counter()
            raw_model.train()
            optimizer.zero_grad(set_to_none=True)
            train_loss_sum = 0.0
            accum_count = 0

            for step_in_epoch in range(steps_per_epoch):
                current_lr = learning_rate_for_step(experiment.run, global_step)
                _set_optimizer_lr(optimizer, current_lr)
                inputs, slots, mask = _move_batch(next(train_iter), target_device)

                with runtime.autocast_context(target_device, amp_dtype):
                    logits = forward_logits_fn(inputs)
                    loss = criterion(logits, slots, mask)

                train_loss_sum += float(loss.detach().item())
                scaled_loss = loss / float(grad_accum)
                scaler.scale(scaled_loss).backward()
                accum_count += 1
                global_step += 1

                is_last_step = step_in_epoch + 1 == steps_per_epoch
                if accum_count >= grad_accum or is_last_step:
                    if clipnorm > 0.0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(raw_model.parameters(), clipnorm)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    if ema is not None:
                        ema.update(raw_model)
                    accum_count = 0

            mean_train_loss = train_loss_sum / float(steps_per_epoch)

            eval_ctx = (
                ema.average_parameters(raw_model)
                if ema is not None
                else contextlib.nullcontext()
            )
            with eval_ctx:
                raw_model.eval()
                val_loss_sum = 0.0
                val_iter = iter(val_loader)
                with torch.no_grad():
                    for _ in range(validation_steps):
                        val_inputs, val_slots, val_mask = _move_batch(
                            next(val_iter), target_device
                        )
                        with runtime.autocast_context(target_device, amp_dtype):
                            val_logits = raw_model.forward_logits(val_inputs)
                            val_loss = criterion(val_logits, val_slots, val_mask)
                        val_loss_sum += float(val_loss.detach().item())
                mean_val_loss = val_loss_sum / float(validation_steps)

                elapsed = time.perf_counter() - t0
                current_lr = float(optimizer.param_groups[0]["lr"])
                print(
                    f"Epoch {epoch_num}/{total_epochs} "
                    f"- {elapsed:.1f}s "
                    f"- loss: {mean_train_loss:.5f} "
                    f"- val_loss: {mean_val_loss:.5f} "
                    f"- lr: {current_lr:.2e}",
                    flush=True,
                )

                save_checkpoint(
                    raw_model,
                    last_path,
                    extra={"epoch": epoch_num, "val_loss": mean_val_loss},
                )
                if best_val_loss is None or mean_val_loss < best_val_loss:
                    prev_str = (
                        "inf" if best_val_loss is None else f"{best_val_loss:.5f}"
                    )
                    best_val_loss = mean_val_loss
                    save_checkpoint(
                        raw_model,
                        best_path,
                        extra={"epoch": epoch_num, "val_loss": mean_val_loss},
                    )
                    print(
                        f"Epoch {epoch_num}: val_loss improved from {prev_str} "
                        f"to {mean_val_loss:.5f}, saving model to {best_path}",
                        flush=True,
                    )

                if writer is not None:
                    writer.add_scalar("loss/train", mean_train_loss, epoch_num)
                    writer.add_scalar("loss/val", mean_val_loss, epoch_num)
                    writer.add_scalar("lr", current_lr, epoch_num)

                eval_every = int(experiment.run.eval_slot_f1_every)
                if eval_every > 0 and (
                    epoch_num % eval_every == 0 or epoch_num == total_epochs
                ):
                    slot_report = evaluation.evaluate_slot48(
                        raw_model,
                        val_charts,
                        seed=experiment.run.seed,
                        max_beats=experiment.dataset.max_beats,
                        device=target_device,
                    )
                    if writer is not None:
                        writer.add_scalar(
                            "val_slot_f1_at_05",
                            float(slot_report.f1_at_05),
                            epoch_num,
                        )
                        writer.add_scalar(
                            "val_slot_f1_max",
                            float(slot_report.f1_max),
                            epoch_num,
                        )
                    if best_slot_f1 is None or slot_report.f1_at_05 > best_slot_f1:
                        prev_f1 = (
                            "None" if best_slot_f1 is None else f"{best_slot_f1:.5f}"
                        )
                        best_slot_f1 = float(slot_report.f1_at_05)
                        best_slot_f1_epoch = epoch_num
                        save_checkpoint(
                            raw_model,
                            best_slot_f1_path,
                            extra={
                                "epoch": epoch_num,
                                "val_slot_f1_at_05": best_slot_f1,
                                "val_slot_f1_max": float(slot_report.f1_max),
                            },
                        )
                        print(
                            f"Epoch {epoch_num}: val_slot_f1_at_05 improved from "
                            f"{prev_f1} to {slot_report.f1_at_05:.5f} "
                            f"(f1_max={slot_report.f1_max:.5f} @ "
                            f"{slot_report.best_threshold:.2f}), saving model to "
                            f"{best_slot_f1_path}",
                            flush=True,
                        )
                    else:
                        print(
                            f"Epoch {epoch_num}: "
                            f"val_slot_f1_at_05={slot_report.f1_at_05:.5f} "
                            f"(f1_max={slot_report.f1_max:.5f}; "
                            f"best={best_slot_f1:.5f} @ ep {best_slot_f1_epoch})",
                            flush=True,
                        )

            backup_extra: dict[str, Any] = {
                "completed_epoch": epoch_num,
                "global_step": global_step,
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_loss": best_val_loss,
                "best_slot_f1": best_slot_f1,
                "best_slot_f1_epoch": best_slot_f1_epoch,
            }
            if ema is not None:
                backup_extra["ema_state_dict"] = ema.state_dict()
            save_checkpoint(raw_model, backup_path, extra=backup_extra)
    finally:
        if writer is not None:
            writer.flush()
            writer.close()

    if backup_dir.exists():
        shutil.rmtree(backup_dir)

    if ema is not None:
        ema.apply_shadow(raw_model)

    save_path = output_dir / f"{experiment.run.model_name}.pt"
    save_checkpoint(raw_model, save_path, extra={"epoch": total_epochs})
    print(f"saved {save_path}", flush=True)

    last_report = evaluation.evaluate_slot48(
        raw_model,
        val_charts,
        seed=experiment.run.seed,
        max_beats=experiment.dataset.max_beats,
        device=target_device,
    )
    _write_eval_report(last_report, output_dir / LAST_EVAL_FILENAME, weights="last")

    if best_path.is_file():
        best_model = load_placement_model(best_path, device=target_device)
        best_report = evaluation.evaluate_slot48(
            best_model,
            val_charts,
            seed=experiment.run.seed,
            max_beats=experiment.dataset.max_beats,
            device=target_device,
        )
        _write_eval_report(best_report, output_dir / BEST_EVAL_FILENAME, weights="best")
        print(f"saved {best_path}", flush=True)

    if best_slot_f1_path.is_file():
        best_f1_model = load_placement_model(best_slot_f1_path, device=target_device)
        best_f1_report = evaluation.evaluate_slot48(
            best_f1_model,
            val_charts,
            seed=experiment.run.seed,
            max_beats=experiment.dataset.max_beats,
            device=target_device,
        )
        _write_eval_report(
            best_f1_report,
            output_dir / BEST_SLOT_F1_EVAL_FILENAME,
            weights="best_slot_f1",
        )
        print(f"saved {best_slot_f1_path}", flush=True)

    return raw_model
