"""Train a PyTorch DDCL ConvLSTM or MERT placement model (`omalley2025ddcl`)."""

from __future__ import annotations

import datetime
import json
import pathlib
import shutil
import time
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from stepcovnet.dataset_prep import training_index
from stepcovnet.ddcl import config, constants, datasets
from stepcovnet.pytorch import runtime
from stepcovnet.pytorch.ddcl import evaluation, models
from stepcovnet.pytorch.itgpt import evaluation as slot_eval

BEST_CHECKPOINT_FILENAME = "best.pt"
BEST_SLOT_F1_CHECKPOINT_FILENAME = "best_slot_f1.pt"
LAST_CHECKPOINT_FILENAME = "last.pt"
BACKUP_DIRNAME = "backup"
BACKUP_FILENAME = "resume.pt"
LAST_EVAL_FILENAME = "eval_val_slot48.json"
BEST_EVAL_FILENAME = "eval_val_slot48_best.json"
BEST_SLOT_F1_EVAL_FILENAME = "eval_val_slot48_best_f1.json"

DdclTorchModel = models.DdclConvLstmPlacementModel | models.DdclMertPlacementModel


def save_checkpoint(
    model: DdclTorchModel,
    path: str | pathlib.Path,
    *,
    extra: dict[str, Any] | None = None,
) -> pathlib.Path:
    """Save a PyTorch DDCL checkpoint with architecture config and weights.

    Args:
        model: DDCL ConvLSTM or MERT model.
        path: Destination ``.pt`` path.
        extra: Optional metadata dictionary.

    Returns:
        Saved :class:`pathlib.Path`.
    """
    target = pathlib.Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "model_config": model.model_config(),
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
    }
    if extra:
        payload.update(extra)
    tmp = target.with_suffix(target.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(target)
    return target


def load_placement_model(
    path: str | pathlib.Path,
    *,
    device: str | torch.device | None = None,
) -> DdclTorchModel:
    """Load a saved PyTorch DDCL placement checkpoint.

    Args:
        path: Path to a ``.pt`` checkpoint.
        device: Optional target compute device.

    Returns:
        Loaded DDCL model in eval mode.
    """
    target_device = runtime.resolve_device(device)
    ckpt = torch.load(pathlib.Path(path), map_location="cpu", weights_only=False)
    cfg = dict(ckpt["model_config"])
    arch = cfg.pop("arch", "convlstm")
    cfg["dense_sizes"] = tuple(cfg["dense_sizes"])
    model: DdclTorchModel = (
        models.build_mert_placement_model(**cfg)
        if arch == "mert"
        else models.build_convlstm_placement_model(**cfg)
    )
    model.load_state_dict(ckpt["state_dict"])
    model.to(target_device)
    model.eval()
    return model


def tensorboard_run_log_dir(callback_root_dir: str, model_name: str) -> pathlib.Path:
    """Return a per-run TensorBoard directory under the stage ``logs/`` tree.

    Args:
        callback_root_dir: Stage root shared by DDCL placement runs.
        model_name: Run label.

    Returns:
        ``{callback_root_dir}/logs/{YYYYMMDD-HHMMSS}-{model_name}``.
    """
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    safe_name = str(model_name).strip() or "ddcl_placement"
    log_dir = pathlib.Path(callback_root_dir) / "logs" / f"{stamp}-{safe_name}"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def _write_eval_report(
    report: slot_eval.Slot48EvalReport,
    path: pathlib.Path,
    *,
    weights: str,
) -> None:
    """Write ``M-slot48`` JSON report and print to stdout.

    Args:
        report: Pooled slot-F1 report.
        path: Output JSON path.
        weights: Checkpoint label.
    """
    payload = evaluation.report_as_dict(report, weights=weights)
    text = json.dumps(payload, indent=2)
    path.write_text(text + "\n", encoding="utf-8")
    print(text)


def train_placement(
    experiment: config.DdclExperimentConfig,
    *,
    device: str | torch.device | None = None,
    precision: str = "fp32",
) -> DdclTorchModel:
    """Load Dataset A charts, train PyTorch DDCL, and save checkpoints + reports.

    Args:
        experiment: DDCL experiment config.
        device: Optional target compute device.
        precision: Precision mode (``"fp32"``, ``"bf16"``, ``"fp16"``, ``"auto"``).

    Returns:
        Trained DDCL model.

    Raises:
        ValueError: If train or val split is empty.
    """
    runtime.set_seed(experiment.run.seed)
    target_device = runtime.resolve_device(device)
    amp_dtype = runtime.resolve_amp_dtype(precision, target_device)
    use_scaler = target_device.type == "cuda" and amp_dtype == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)  # type: ignore[attr-defined]

    train_charts = datasets.load_split_charts(
        experiment.dataset, training_index.SPLIT_TRAIN
    )
    val_charts = datasets.load_split_charts(
        experiment.dataset, training_index.SPLIT_VAL
    )
    if not train_charts:
        raise ValueError("train split is empty")
    if not val_charts:
        raise ValueError("val split is empty")

    if experiment.dataset.audio_feature_type == constants.AUDIO_FEATURE_MERT:
        feature_dim = int(train_charts[0].beat_audio.shape[-1])
        model: DdclTorchModel = models.build_mert_placement_model(
            memlen=experiment.dataset.memlen,
            n_frames=experiment.dataset.n_frames_per_beat,
            feature_dim=feature_dim,
            proj_dim=experiment.model.mert_proj_dim,
            conv_filters=experiment.model.mert_conv_filters,
            noise_std=experiment.model.mert_noise_std,
            lstm_units=experiment.model.lstm_units,
            dropout_rate=experiment.model.dropout_rate,
            dense_sizes=tuple(experiment.model.dense_sizes),
        )
    else:
        model = models.build_convlstm_placement_model(
            memlen=experiment.dataset.memlen,
            n_frames=experiment.dataset.n_frames_per_beat,
            lstm_units=experiment.model.lstm_units,
            dropout_rate=experiment.model.dropout_rate,
            dense_sizes=tuple(experiment.model.dense_sizes),
        )
    model = model.to(target_device)

    criterion = models.BinaryFocalWithLogitsLoss().to(target_device)
    optimizer: torch.optim.Optimizer = (
        torch.optim.AdamW(
            model.parameters(),
            lr=float(experiment.run.learning_rate),
            weight_decay=float(experiment.run.weight_decay),
            eps=1e-7,
        )
        if experiment.run.weight_decay > 0.0
        else torch.optim.Adam(
            model.parameters(),
            lr=float(experiment.run.learning_rate),
            eps=1e-7,
        )
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
    best_val_loss: float | None = None
    best_slot_f1: float | None = None
    if experiment.run.resume and backup_path.is_file():
        state = torch.load(backup_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state["state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        start_epoch = int(state.get("completed_epoch", 0))
        best_val_loss = state.get("best_val_loss")
        best_slot_f1 = state.get("best_slot_f1")

    writer: SummaryWriter | None = None
    if experiment.run.callback_root_dir:
        log_dir = tensorboard_run_log_dir(
            experiment.run.callback_root_dir,
            experiment.run.model_name,
        )
        writer = SummaryWriter(log_dir=str(log_dir))

    train_rng = np.random.default_rng(experiment.run.seed + start_epoch)
    val_rng = np.random.default_rng(experiment.run.seed + 1 + start_epoch)
    steps_per_epoch = max(1, int(experiment.run.steps_per_epoch))
    val_steps = max(1, int(experiment.run.validation_steps))
    total_epochs = int(experiment.run.epoch)
    clipnorm = float(experiment.run.clipnorm)

    try:
        for epoch_idx in range(start_epoch, total_epochs):
            epoch_num = epoch_idx + 1
            t0 = time.perf_counter()
            model.train()
            train_loss_sum = 0.0
            for _ in range(steps_per_epoch):
                inputs_np, target_np = datasets.sample_train_batch(
                    train_charts,
                    batch_size=experiment.dataset.batch_size,
                    rng=train_rng,
                )
                inputs = {
                    k: torch.as_tensor(v, dtype=torch.float32, device=target_device)
                    for k, v in inputs_np.items()
                }
                targets = torch.as_tensor(
                    target_np, dtype=torch.float32, device=target_device
                )
                optimizer.zero_grad(set_to_none=True)
                with runtime.autocast_context(target_device, amp_dtype):
                    logits = model.forward_logits(inputs)
                    loss = criterion(logits, targets)
                scaler.scale(loss).backward()
                if clipnorm > 0.0:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), clipnorm)
                scaler.step(optimizer)
                scaler.update()
                train_loss_sum += float(loss.detach().item())

            model.eval()
            val_loss_sum = 0.0
            with torch.no_grad():
                for _ in range(val_steps):
                    v_inputs_np, v_target_np = datasets.sample_train_batch(
                        val_charts,
                        batch_size=experiment.dataset.batch_size,
                        rng=val_rng,
                    )
                    v_inputs = {
                        k: torch.as_tensor(v, dtype=torch.float32, device=target_device)
                        for k, v in v_inputs_np.items()
                    }
                    v_targets = torch.as_tensor(
                        v_target_np, dtype=torch.float32, device=target_device
                    )
                    with runtime.autocast_context(target_device, amp_dtype):
                        v_logits = model.forward_logits(v_inputs)
                        v_loss = criterion(v_logits, v_targets)
                    val_loss_sum += float(v_loss.detach().item())

            mean_train = train_loss_sum / float(steps_per_epoch)
            mean_val = val_loss_sum / float(val_steps)
            elapsed = time.perf_counter() - t0
            print(
                f"Epoch {epoch_num}/{total_epochs} - {elapsed:.1f}s "
                f"- loss: {mean_train:.5f} - val_loss: {mean_val:.5f}",
                flush=True,
            )

            save_checkpoint(model, last_path, extra={"epoch": epoch_num})
            if best_val_loss is None or mean_val < best_val_loss:
                best_val_loss = mean_val
                save_checkpoint(
                    model,
                    best_path,
                    extra={"epoch": epoch_num, "val_loss": mean_val},
                )
            if writer is not None:
                writer.add_scalar("loss/train", mean_train, epoch_num)
                writer.add_scalar("loss/val", mean_val, epoch_num)

            eval_every = int(experiment.run.eval_slot_f1_every)
            if eval_every > 0 and (
                epoch_num % eval_every == 0 or epoch_num == total_epochs
            ):
                slot_report = evaluation.evaluate_slot48(
                    model, val_charts, seed=experiment.run.seed, device=target_device
                )
                if writer is not None:
                    writer.add_scalar(
                        "val_slot_f1_at_05", float(slot_report.f1_at_05), epoch_num
                    )
                    writer.add_scalar(
                        "val_slot_f1_max", float(slot_report.f1_max), epoch_num
                    )
                if best_slot_f1 is None or slot_report.f1_at_05 > best_slot_f1:
                    best_slot_f1 = float(slot_report.f1_at_05)
                    save_checkpoint(
                        model,
                        best_slot_f1_path,
                        extra={"epoch": epoch_num, "val_slot_f1_at_05": best_slot_f1},
                    )

            save_checkpoint(
                model,
                backup_path,
                extra={
                    "completed_epoch": epoch_num,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_loss": best_val_loss,
                    "best_slot_f1": best_slot_f1,
                },
            )
    finally:
        if writer is not None:
            writer.flush()
            writer.close()

    if backup_dir.exists():
        shutil.rmtree(backup_dir)

    save_path = output_dir / f"{experiment.run.model_name}.pt"
    save_checkpoint(model, save_path, extra={"epoch": total_epochs})
    print(f"saved {save_path}")

    last_report = evaluation.evaluate_slot48(
        model, val_charts, seed=experiment.run.seed, device=target_device
    )
    _write_eval_report(last_report, output_dir / LAST_EVAL_FILENAME, weights="last")

    if best_path.is_file():
        best_model = load_placement_model(best_path, device=target_device)
        best_report = evaluation.evaluate_slot48(
            best_model, val_charts, seed=experiment.run.seed, device=target_device
        )
        _write_eval_report(best_report, output_dir / BEST_EVAL_FILENAME, weights="best")
        print(f"saved {best_path}")

    if best_slot_f1_path.is_file():
        best_f1_model = load_placement_model(best_slot_f1_path, device=target_device)
        best_f1_report = evaluation.evaluate_slot48(
            best_f1_model, val_charts, seed=experiment.run.seed, device=target_device
        )
        _write_eval_report(
            best_f1_report,
            output_dir / BEST_SLOT_F1_EVAL_FILENAME,
            weights="best_slot_f1",
        )
        print(f"saved {best_slot_f1_path}")
    return model
