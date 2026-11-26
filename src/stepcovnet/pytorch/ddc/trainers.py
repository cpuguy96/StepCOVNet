"""Train a PyTorch DDC C-LSTM placement model (`donahue2017ddc`)."""

from __future__ import annotations

import datetime
import json
import pathlib
import time
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as nn_func
from torch.utils.tensorboard import SummaryWriter

from stepcovnet.dataset_prep import training_index
from stepcovnet.ddc import config, datasets
from stepcovnet.pytorch import runtime
from stepcovnet.pytorch.ddc import evaluation, models

BEST_CHECKPOINT_FILENAME = "best.pt"
LAST_EVAL_FILENAME = "eval_val_ddc.json"
BEST_EVAL_FILENAME = "eval_val_ddc_best.json"


def save_checkpoint(
    model: models.DdcClstmPlacementModel,
    path: str | pathlib.Path,
    *,
    extra: dict[str, Any] | None = None,
) -> pathlib.Path:
    """Save a PyTorch DDC checkpoint with config and weights.

    Args:
        model: Trained DDC model.
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
) -> models.DdcClstmPlacementModel:
    """Load a saved PyTorch DDC placement checkpoint.

    Args:
        path: Path to a ``.pt`` checkpoint.
        device: Optional target compute device.

    Returns:
        Loaded :class:`DdcClstmPlacementModel` in eval mode.
    """
    target_device = runtime.resolve_device(device)
    ckpt = torch.load(pathlib.Path(path), map_location="cpu", weights_only=False)
    cfg = dict(ckpt["model_config"])
    cfg["dnn_sizes"] = tuple(cfg["dnn_sizes"])
    model = models.build_clstm_placement_model(**cfg)
    model.load_state_dict(ckpt["state_dict"])
    model.to(target_device)
    model.eval()
    return model


def tensorboard_run_log_dir(callback_root_dir: str, model_name: str) -> pathlib.Path:
    """Return a per-run TensorBoard directory under the stage ``logs/`` tree.

    Args:
        callback_root_dir: Stage root shared by DDC placement runs.
        model_name: Run label.

    Returns:
        ``{callback_root_dir}/logs/{YYYYMMDD-HHMMSS}-{model_name}``.
    """
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    safe_name = str(model_name).strip() or "ddc_placement"
    log_dir = pathlib.Path(callback_root_dir) / "logs" / f"{stamp}-{safe_name}"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def _write_eval_report(
    report: evaluation.PlacementEvalReport,
    path: pathlib.Path,
    *,
    weights: str,
) -> None:
    """Write a DDC placement eval JSON tagged with which weights were scored.

    Args:
        report: Val F-score report.
        path: Output JSON path.
        weights: ``last`` or ``best``.
    """
    payload = report.as_dict()
    payload["weights"] = weights
    text = json.dumps(payload, indent=2)
    path.write_text(text + "\n", encoding="utf-8")
    print(text)


def train_placement(
    experiment: config.PlacementExperimentConfig,
    *,
    device: str | torch.device | None = None,
    precision: str = "fp32",
) -> models.DdcClstmPlacementModel:
    """Load Dataset A charts, train the PyTorch C-LSTM, and save weights.

    Args:
        experiment: Placement experiment config.
        device: Optional target device override.
        precision: Precision mode (``"fp32"``, ``"bf16"``, ``"fp16"``, ``"auto"``).

    Returns:
        Trained :class:`DdcClstmPlacementModel`.

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

    model = models.build_clstm_placement_model(
        lstm_units=experiment.model.lstm_units,
        lstm_layers=experiment.model.lstm_layers,
        dropout_rate=experiment.model.dropout_rate,
        dnn_sizes=tuple(experiment.model.dnn_sizes),
    ).to(target_device)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=float(experiment.run.learning_rate),
    )

    output_dir = pathlib.Path(experiment.run.model_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / BEST_CHECKPOINT_FILENAME

    writer: SummaryWriter | None = None
    if experiment.run.callback_root_dir:
        log_dir = tensorboard_run_log_dir(
            experiment.run.callback_root_dir,
            experiment.run.model_name,
        )
        writer = SummaryWriter(log_dir=str(log_dir))

    train_rng = np.random.default_rng(experiment.run.seed)
    steps_per_epoch = max(1, int(experiment.run.steps_per_epoch))
    val_steps = max(1, int(experiment.run.validation_steps))
    clipnorm = float(experiment.run.clipnorm)
    best_val_loss: float | None = None

    try:
        for epoch_idx in range(int(experiment.run.epoch)):
            epoch_num = epoch_idx + 1
            t0 = time.perf_counter()
            model.train()
            train_loss_sum = 0.0
            for _ in range(steps_per_epoch):
                inputs_np, target_np = datasets.sample_train_batch(
                    train_charts,
                    batch_size=experiment.dataset.batch_size,
                    nunroll=experiment.dataset.nunroll,
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
                    loss = nn_func.binary_cross_entropy_with_logits(
                        logits.float(), targets.float()
                    )
                scaler.scale(loss).backward()
                if clipnorm > 0.0:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), clipnorm)
                scaler.step(optimizer)
                scaler.update()
                train_loss_sum += float(loss.detach().item())

            model.eval()
            val_loss_sum = 0.0
            val_rng = np.random.default_rng(experiment.run.seed + 1)
            with torch.no_grad():
                for _ in range(val_steps):
                    v_inputs_np, v_target_np = datasets.sample_train_batch(
                        val_charts,
                        batch_size=experiment.dataset.batch_size,
                        nunroll=experiment.dataset.nunroll,
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
                        v_loss = nn_func.binary_cross_entropy_with_logits(
                            v_logits.float(), v_targets.float()
                        )
                    val_loss_sum += float(v_loss.detach().item())

            mean_train = train_loss_sum / float(steps_per_epoch)
            mean_val = val_loss_sum / float(val_steps)
            elapsed = time.perf_counter() - t0
            print(
                f"Epoch {epoch_num}/{experiment.run.epoch} - {elapsed:.1f}s "
                f"- loss: {mean_train:.5f} - val_loss: {mean_val:.5f}",
                flush=True,
            )
            if writer is not None:
                writer.add_scalar("loss/train", mean_train, epoch_num)
                writer.add_scalar("loss/val", mean_val, epoch_num)
            if best_val_loss is None or mean_val < best_val_loss:
                best_val_loss = mean_val
                save_checkpoint(
                    model,
                    best_path,
                    extra={"epoch": epoch_num, "val_loss": mean_val},
                )
    finally:
        if writer is not None:
            writer.flush()
            writer.close()

    save_path = output_dir / f"{experiment.run.model_name}.pt"
    save_checkpoint(model, save_path, extra={"epoch": int(experiment.run.epoch)})
    print(f"saved {save_path}")

    last_report = evaluation.evaluate_placement(
        model, val_charts, seed=experiment.run.seed, device=target_device
    )
    _write_eval_report(last_report, output_dir / LAST_EVAL_FILENAME, weights="last")

    if best_path.is_file():
        best_model = load_placement_model(best_path, device=target_device)
        best_report = evaluation.evaluate_placement(
            best_model, val_charts, seed=experiment.run.seed, device=target_device
        )
        _write_eval_report(best_report, output_dir / BEST_EVAL_FILENAME, weights="best")
        print(f"saved {best_path}")
    return model
