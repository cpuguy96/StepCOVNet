"""PyTorch training loop, checkpointing, and eval for AR onset models."""

from __future__ import annotations

import dataclasses
import datetime
import json
import pathlib
import time
from typing import Any

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from stepcovnet.pytorch import runtime
from stepcovnet.pytorch.onset_ar import config, datasets, inference, losses, models

BEST_CHECKPOINT_FILENAME = "best.pt"
LAST_CHECKPOINT_FILENAME = "last.pt"
EVAL_REPORT_FILENAME = "eval_val_ar.json"


@dataclasses.dataclass
class ArTrainResult:
    """Summary returned by :func:`train_ar_onset`.

    Attributes:
        model: Trained :class:`models.ArOnsetModel`.
        history: Epoch-indexed dictionary of scalar metric lists.
        best_checkpoint: Path to best saved ``.pt`` checkpoint (or None).
        val_report: Optional final free-run :class:`inference.ArEvalReport`.
    """

    model: models.ArOnsetModel
    history: dict[str, list[float]]
    best_checkpoint: pathlib.Path | None = None
    val_report: inference.ArEvalReport | None = None


def save_checkpoint(
    model: models.ArOnsetModel,
    path: str | pathlib.Path,
    *,
    extra: dict[str, Any] | None = None,
) -> pathlib.Path:
    """Save an AR onset PyTorch checkpoint atomically.

    Args:
        model: AR onset model.
        path: Destination ``.pt`` path.
        extra: Optional extra metadata.

    Returns:
        Saved :class:`pathlib.Path`.
    """
    target = pathlib.Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "experiment_config": model.model_config(),
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
    }
    if extra:
        payload.update(extra)
    tmp = target.with_suffix(target.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(target)
    return target


def load_ar_onset_model(
    path: str | pathlib.Path,
    *,
    device: str | torch.device | None = None,
) -> models.ArOnsetModel:
    """Load a saved PyTorch AR onset checkpoint.

    Args:
        path: Path to ``.pt`` checkpoint.
        device: Optional compute device.

    Returns:
        Loaded :class:`models.ArOnsetModel` in eval mode.
    """
    target_device = runtime.resolve_device(device)
    ckpt = torch.load(pathlib.Path(path), map_location="cpu", weights_only=False)
    exp_cfg = config.ArExperimentConfig.from_dict(ckpt["experiment_config"])
    model = models.build_ar_onset_model(exp_cfg)
    model.load_state_dict(ckpt["state_dict"])
    model.to(target_device)
    model.eval()
    return model


def _soft_alpha_for_epoch(run_cfg: config.ArRunConfig, epoch_idx: int) -> float:
    base = float(run_cfg.pointer_soft_distance_alpha)
    anneal = int(run_cfg.pointer_soft_distance_alpha_anneal_epochs)
    if base <= 0.0 or anneal <= 0:
        return base
    hold = int(run_cfg.pointer_soft_distance_alpha_hold_epochs)
    if epoch_idx < hold:
        return base
    prog = min(1.0, float(epoch_idx - hold) / float(max(1, anneal)))
    return base * (1.0 - prog)


def _lambda_time_for_epoch(run_cfg: config.ArRunConfig, epoch_idx: int) -> float:
    target = float(run_cfg.lambda_time)
    ramp = int(run_cfg.lambda_time_ramp_epochs)
    if target <= 0.0 or ramp <= 0:
        return target
    return target * min(1.0, float(epoch_idx + 1) / float(ramp))


def _tensorboard_log_dir(callback_root_dir: str, run_label: str) -> pathlib.Path:
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    safe = str(run_label).strip() or "ar_onset"
    log_dir = pathlib.Path(callback_root_dir) / "logs" / f"{stamp}-{safe}"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def train_ar_onset(
    experiment_config: config.ArExperimentConfig,
    *,
    train_samples: list[datasets.ArSample] | None = None,
    val_samples: list[datasets.ArSample] | None = None,
    device: str | torch.device | None = None,
    precision: str = "fp32",
    eval_val: bool = True,
) -> ArTrainResult:
    """Train a PyTorch AR onset encoder-decoder Transformer.

    Args:
        experiment_config: Complete AR experiment config.
        train_samples: Optional preloaded training samples (loaded from config if None).
        val_samples: Optional preloaded validation samples.
        device: Target device (``auto``, ``cuda``, ``cpu``, etc.).
        precision: Precision policy (``fp32``, ``bf16``, ``fp16``, ``auto``).
        eval_val: Whether to run free-run KV-cached evaluation after training.

    Returns:
        :class:`ArTrainResult` containing model, loss history, and eval report.
    """
    run_cfg = experiment_config.run
    ds_cfg = experiment_config.dataset
    mc = experiment_config.model

    runtime.set_seed(run_cfg.seed)
    target_device = runtime.resolve_device(device)
    runtime.configure_cuda_performance(target_device)
    amp_dtype = runtime.resolve_amp_dtype(
        "auto" if run_cfg.mixed_precision and precision == "fp32" else precision,
        target_device,
    )
    use_scaler = amp_dtype == torch.float16 and target_device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    if train_samples is None:
        train_samples = datasets.load_split_samples(experiment_config, "train")
    if val_samples is None:
        val_samples = (
            train_samples
            if run_cfg.overfit_one_song
            else datasets.load_split_samples(experiment_config, "val")
        )

    model = models.build_ar_onset_model(experiment_config).to(target_device)
    if run_cfg.init_model_path:
        ckpt = torch.load(
            pathlib.Path(run_cfg.init_model_path),
            map_location="cpu",
            weights_only=False,
        )
        model.load_state_dict(ckpt["state_dict"], strict=False)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(run_cfg.learning_rate),
        eps=1e-7,
    )

    train_loader = datasets.build_dataloader(
        train_samples,
        batch_size=ds_cfg.batch_size,
        shuffle=len(train_samples) > 1,
        dynamic_padding=True,
        seed=run_cfg.seed,
    )
    val_loader = datasets.build_dataloader(
        val_samples,
        batch_size=ds_cfg.batch_size,
        shuffle=False,
        dynamic_padding=True,
        seed=run_cfg.seed,
    )

    out_dir = (
        pathlib.Path(run_cfg.model_output_dir) if run_cfg.model_output_dir else None
    )
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
    writer: SummaryWriter | None = None
    if run_cfg.callback_root_dir:
        writer = SummaryWriter(
            log_dir=str(
                _tensorboard_log_dir(run_cfg.callback_root_dir, run_cfg.run_label)
            )
        )

    history: dict[str, list[float]] = {
        "loss": [],
        "val_loss": [],
        "token_acc": [],
        "val_token_acc": [],
    }
    best_val_loss = float("inf")
    best_ckpt: pathlib.Path | None = None

    try:
        for epoch in range(int(run_cfg.epochs)):
            t0 = time.perf_counter()
            alpha = _soft_alpha_for_epoch(run_cfg, epoch)
            lam_time = _lambda_time_for_epoch(run_cfg, epoch)

            model.train()
            tr_loss_sum = 0.0
            tr_acc_sum = 0.0
            tr_steps = 0
            for batch in train_loader:
                batch_dev = {k: v.to(target_device) for k, v in batch.items()}
                optimizer.zero_grad(set_to_none=True)
                with runtime.autocast_context(target_device, amp_dtype):
                    outputs = model(batch_dev)
                    total_loss, _ = losses.compute_ar_onset_loss(
                        outputs,
                        batch_dev,
                        patch_frames=mc.patch_frames,
                        hop_sec=ds_cfg.hop_sec,
                        lambda_time=lam_time,
                        lambda_residual=run_cfg.lambda_residual,
                        pointer_loss_weight=run_cfg.pointer_loss_weight,
                        length_normalize_ce=run_cfg.length_normalize_ce,
                        use_soft_pointer_time=run_cfg.use_soft_pointer_time,
                        use_ste_pointer_time=run_cfg.use_ste_pointer_time,
                        time_loss_correct_patch_only=run_cfg.time_loss_correct_patch_only,
                        pointer_local_ce_radius=run_cfg.pointer_local_ce_radius,
                        pointer_local_ce_anchor=run_cfg.pointer_local_ce_anchor,
                        pointer_soft_distance_alpha=alpha,
                        monotonic_pointer=mc.monotonic_pointer,
                        gap_alignment=model.gap_alignment,
                        gap_loss_weight=run_cfg.gap_loss_weight,
                        gap_delta_lookup=model.gap_delta_lookup,
                    )
                if use_scaler:
                    scaler.scale(total_loss).backward()
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    total_loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    optimizer.step()

                tr_loss_sum += float(total_loss.detach().cpu().item())
                tr_acc_sum += float(
                    losses.masked_token_accuracy(
                        outputs["token_logits"].detach(),
                        batch_dev["decoder_target_ids"],
                        batch_dev["decoder_mask"],
                    )
                    .cpu()
                    .item()
                )
                tr_steps += 1

            model.eval()
            va_loss_sum = 0.0
            va_acc_sum = 0.0
            va_steps = 0
            with torch.no_grad():
                for batch in val_loader:
                    batch_dev = {k: v.to(target_device) for k, v in batch.items()}
                    with runtime.autocast_context(target_device, amp_dtype):
                        outputs = model(batch_dev)
                        val_loss, _ = losses.compute_ar_onset_loss(
                            outputs,
                            batch_dev,
                            patch_frames=mc.patch_frames,
                            hop_sec=ds_cfg.hop_sec,
                            lambda_time=lam_time,
                            lambda_residual=run_cfg.lambda_residual,
                            pointer_loss_weight=run_cfg.pointer_loss_weight,
                            length_normalize_ce=run_cfg.length_normalize_ce,
                            use_soft_pointer_time=run_cfg.use_soft_pointer_time,
                            use_ste_pointer_time=run_cfg.use_ste_pointer_time,
                            time_loss_correct_patch_only=run_cfg.time_loss_correct_patch_only,
                            pointer_local_ce_radius=run_cfg.pointer_local_ce_radius,
                            pointer_local_ce_anchor=run_cfg.pointer_local_ce_anchor,
                            pointer_soft_distance_alpha=alpha,
                            monotonic_pointer=mc.monotonic_pointer,
                            gap_alignment=model.gap_alignment,
                            gap_loss_weight=run_cfg.gap_loss_weight,
                            gap_delta_lookup=model.gap_delta_lookup,
                        )
                    va_loss_sum += float(val_loss.detach().cpu().item())
                    va_acc_sum += float(
                        losses.masked_token_accuracy(
                            outputs["token_logits"],
                            batch_dev["decoder_target_ids"],
                            batch_dev["decoder_mask"],
                        )
                        .cpu()
                        .item()
                    )
                    va_steps += 1

            tr_loss = tr_loss_sum / max(1, tr_steps)
            va_loss = va_loss_sum / max(1, va_steps)
            tr_acc = tr_acc_sum / max(1, tr_steps)
            va_acc = va_acc_sum / max(1, va_steps)
            history["loss"].append(tr_loss)
            history["val_loss"].append(va_loss)
            history["token_acc"].append(tr_acc)
            history["val_token_acc"].append(va_acc)

            if writer is not None:
                writer.add_scalar("loss/train", tr_loss, epoch + 1)
                writer.add_scalar("loss/val", va_loss, epoch + 1)
                writer.add_scalar("token_acc/train", tr_acc, epoch + 1)
                writer.add_scalar("token_acc/val", va_acc, epoch + 1)

            elapsed = time.perf_counter() - t0
            print(
                f"Epoch {epoch + 1}/{run_cfg.epochs} ({elapsed:.1f}s) "
                f"loss={tr_loss:.4f} val_loss={va_loss:.4f} "
                f"tok_acc={tr_acc:.4f} val_tok_acc={va_acc:.4f}"
            )
            if out_dir is not None and va_loss < best_val_loss:
                best_val_loss = va_loss
                best_ckpt = save_checkpoint(
                    model,
                    out_dir / BEST_CHECKPOINT_FILENAME,
                    extra={"epoch": epoch + 1, "val_loss": va_loss},
                )

        if out_dir is not None:
            save_checkpoint(
                model,
                out_dir / LAST_CHECKPOINT_FILENAME,
                extra={"epoch": int(run_cfg.epochs)},
            )

        val_report: inference.ArEvalReport | None = None
        if eval_val and val_samples:
            val_report = inference.evaluate_ar_samples(
                model, val_samples, device=target_device
            )
            if out_dir is not None:
                (out_dir / EVAL_REPORT_FILENAME).write_text(
                    json.dumps(val_report.as_dict(), indent=2) + "\n",
                    encoding="utf-8",
                )
        return ArTrainResult(
            model=model,
            history=history,
            best_checkpoint=best_ckpt,
            val_report=val_report,
        )
    finally:
        if writer is not None:
            writer.close()
