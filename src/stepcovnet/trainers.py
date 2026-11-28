"""PyTorch training loops for core dense onset and arrow classification models."""

from __future__ import annotations

import copy
import dataclasses
import datetime
import itertools
import json
import logging
import math
import os
import pathlib
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import torch
from torch import nn
from torch.nn import functional as nn_func

from stepcovnet import (
    config,
    constants,
    datasets,
    dense_overfit_eval,
    losses,
    metrics,
    models,
    reproducibility,
    runtime,
    wsl_gpu,
)
from stepcovnet.core import trainers as core_trainers
from stepcovnet.dataset_prep import training_index

if TYPE_CHECKING:
    from torch.utils.tensorboard import SummaryWriter

ONSET_CHECKPOINT_MONITOR = "val_onset_f1_score"
POST_HOC_EVENT_F1_REPORT_NAME = "event_f1_sweep.json"

load_onset_model = core_trainers.load_onset_model
load_arrow_model = core_trainers.load_arrow_model
save_core_checkpoint = core_trainers.save_core_checkpoint


@dataclasses.dataclass
class TrainingHistory:
    """Epoch-level training and validation metrics container.

    Attributes:
        history: Mapping from metric name to list of per-epoch floats.
    """

    history: dict[str, list[float]] = dataclasses.field(default_factory=dict)


def _get_onset_experiment_name(
    take_count: int,
    apply_temporal_augment: bool,
    should_apply_spec_augment: bool,
    use_gaussian_target: bool,
    gaussian_sigma: float,
    model_params: config.OnsetModelConfig,
    feature_source: config.FeatureSource = config.FeatureSource.MEL,
) -> str:
    """Generate a descriptive experiment name from onset hyperparameters.

    Args:
        take_count: Number of batches per epoch (``-1`` for all).
        apply_temporal_augment: Whether temporal augmentation is enabled.
        should_apply_spec_augment: Whether SpecAugment is enabled.
        use_gaussian_target: Whether Gaussian targets are used.
        gaussian_sigma: Standard deviation for Gaussian targets.
        model_params: Dense onset architecture configuration.
        feature_source: Input feature source.

    Returns:
        Human-readable experiment identifier string.
    """
    parts = ["ONSET"]
    if feature_source == config.FeatureSource.MERT:
        parts.append("mert")
    elif feature_source == config.FeatureSource.WAVEFORM:
        parts.append("waveform")

    parts.append("take_all" if take_count == -1 else f"take_{take_count}")

    if use_gaussian_target:
        parts.append(f"sigma_{str(gaussian_sigma).replace('.', '_')}")
    if apply_temporal_augment:
        parts.append("temporal_augment")
    if should_apply_spec_augment:
        parts.append("spec_augment")

    arch = model_params.onset_architecture
    parts.append(arch.value)
    parts.append(f"filters_{model_params.initial_filters}")
    parts.append(f"dropout_{str(model_params.dropout_rate).replace('.', '_')}")

    dilation_rates = model_params.dilation_rates
    if dilation_rates is None:
        dilation_str = "N_A"
    elif isinstance(dilation_rates, list | tuple):
        dilation_str = "_".join(map(str, dilation_rates))
    else:
        dilation_str = str(dilation_rates)

    if arch == config.OnsetArchitecture.BILSTM:
        parts.append(f"bilstm_depth_{model_params.depth}")
        parts.append(f"bilstm_units_{model_params.recurrent_units}")
    elif arch == config.OnsetArchitecture.TRANSFORMER:
        parts.append(f"tfm_layers_{model_params.transformer_layers}")
        parts.append(f"tfm_heads_{model_params.transformer_heads}")
    elif arch == config.OnsetArchitecture.TCN:
        parts.append(f"tcn_blocks_{model_params.tcn_blocks}")
        parts.append(f"kernel_{model_params.kernel_size}")
        parts.append(f"dilations_{dilation_str}")
    else:
        parts.append(f"depth_{model_params.depth}")
        parts.append(f"kernel_{model_params.kernel_size}")
        parts.append(f"dilations_{dilation_str}")

    return "-".join(parts)


def _get_arrow_experiment_name(
    experiment_config: config.ArrowExperimentConfig,
) -> str:
    """Generate a descriptive experiment name from arrow hyperparameters.

    Args:
        experiment_config: Full arrow experiment configuration.

    Returns:
        Human-readable experiment identifier string.
    """
    parts = ["ARROW", experiment_config.model.model_type]
    parts.extend(experiment_config.model.get_experiment_name_parts())
    parts.extend(experiment_config.run.get_experiment_name_parts())
    parts.extend(experiment_config.dataset.get_experiment_name_parts())
    return "-".join(parts)


def _list_monitored_checkpoints(
    callback_root_dir: str,
    monitor_metric: str,
) -> list[str]:
    """Return all monitored ``.pt`` checkpoint paths under a callback root."""
    models_root = pathlib.Path(callback_root_dir) / "models"
    if not models_root.is_dir():
        return []
    prefix = f"{monitor_metric.upper()}-"
    paths: list[str] = []
    for root, _dirs, files in os.walk(models_root):
        for name in files:
            if not name.startswith(prefix) or not name.endswith(".pt"):
                continue
            value_text = name[len(prefix) : -len(".pt")]
            try:
                float(value_text)
            except ValueError:
                continue
            paths.append(str(pathlib.Path(root) / name))
    return sorted(paths)


def _monitored_checkpoint_value(path: str, monitor_metric: str) -> float:
    """Parse the monitored metric value encoded in a checkpoint filename."""
    prefix = f"{monitor_metric.upper()}-"
    name = pathlib.Path(path).name
    return float(name[len(prefix) : -len(".pt")])


def _latest_monitored_checkpoint(
    callback_root_dir: str,
    monitor_metric: str,
    *,
    monitor_mode: str = "max",
) -> str | None:
    """Return the checkpoint with the best monitored metric under a callback root."""
    paths = _list_monitored_checkpoints(callback_root_dir, monitor_metric)
    if not paths:
        return None
    selector = min if monitor_mode == "min" else max
    return selector(
        paths,
        key=lambda path: _monitored_checkpoint_value(path, monitor_metric),
    )


def _save_config(
    experiment_config: config.OnsetExperimentConfig | config.ArrowExperimentConfig,
    callback_root_dir: str,
    callback_name: str,
) -> None:
    """Save experiment config JSON into ``{callback_root_dir}/logs/{callback_name}/config.json``."""
    logdir = pathlib.Path(callback_root_dir) / "logs" / callback_name
    logdir.mkdir(parents=True, exist_ok=True)
    config_path = logdir / "config.json"
    experiment_config.to_json(str(config_path))
    logging.info("Saved experiment config to %s", config_path)


def _model_filename(model: nn.Module, fallback_name: str) -> str:
    name = str(getattr(model, "name", "") or fallback_name)
    return f"{name}.pt"


def _write_model(
    model: nn.Module,
    model_output_dir: str,
    *,
    callback_root_dir: str = "",
    monitor_metric: str = ONSET_CHECKPOINT_MONITOR,
    monitor_mode: str = "max",
    fallback_name: str = "model",
) -> pathlib.Path:
    """Save the trained PyTorch model (or best monitored checkpoint) to ``model_output_dir``."""
    filepath = pathlib.Path(model_output_dir) / _model_filename(model, fallback_name)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    best_path = _latest_monitored_checkpoint(
        callback_root_dir, monitor_metric, monitor_mode=monitor_mode
    )
    if best_path is not None:
        logging.info("Copying best checkpoint from %s to %s", best_path, filepath)
        ckpt = torch.load(
            pathlib.Path(best_path), map_location="cpu", weights_only=False
        )
        tmp = filepath.with_suffix(".pt.tmp")
        torch.save(ckpt, tmp)
        tmp.replace(filepath)
        return filepath
    logging.info("Saving trained model to %s", filepath)
    return core_trainers.save_core_checkpoint(model, filepath)


def _select_best_event_f1_checkpoint(
    dataset_config: config.OnsetDatasetConfig,
    model_config: config.OnsetModelConfig,
    run_config: config.RunConfig,
    *,
    monitor_metric: str = ONSET_CHECKPOINT_MONITOR,
) -> dict[str, Any] | None:
    """Sweep every saved ``.pt`` checkpoint and threshold for best validation event F1."""
    checkpoint_paths = _list_monitored_checkpoints(
        run_config.callback_root_dir,
        monitor_metric,
    )
    if not checkpoint_paths:
        return None
    thresholds = tuple(run_config.post_hoc_event_f1_thresholds)
    per_checkpoint: list[dict[str, Any]] = []
    best_entry: dict[str, Any] | None = None
    for checkpoint_path in checkpoint_paths:
        checkpoint_model = core_trainers.load_onset_model(checkpoint_path)
        sweep = dense_overfit_eval.sweep_thresholds_dense_val_event_f1(
            checkpoint_model,
            dataset_config,
            model_config,
            thresholds=thresholds,
            min_onset_distance_ms=run_config.min_onset_distance_ms,
            tolerance_sec=run_config.tolerance_sec,
        )
        entry: dict[str, Any] = {"checkpoint": checkpoint_path, **sweep}
        per_checkpoint.append(entry)
        if (
            best_entry is None
            or entry["best_micro_event_f1"] > best_entry["best_micro_event_f1"]
        ):
            best_entry = entry
    assert best_entry is not None
    return {
        "best_checkpoint": best_entry["checkpoint"],
        "best_threshold": best_entry["best_threshold"],
        "best_micro_event_f1": best_entry["best_micro_event_f1"],
        "monitor_metric": monitor_metric,
        "per_checkpoint": per_checkpoint,
    }


def _export_best_event_f1_checkpoint(
    model: nn.Module,
    dataset_config: config.OnsetDatasetConfig,
    model_config: config.OnsetModelConfig,
    run_config: config.RunConfig,
    *,
    monitor_metric: str = ONSET_CHECKPOINT_MONITOR,
) -> dict[str, Any] | None:
    """Export the event-F1-optimal checkpoint and write ``event_f1_sweep.json``."""
    report = _select_best_event_f1_checkpoint(
        dataset_config,
        model_config,
        run_config,
        monitor_metric=monitor_metric,
    )
    if report is None:
        return None
    output_dir = pathlib.Path(run_config.model_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    filepath = output_dir / _model_filename(model, "onset_model")
    ckpt = torch.load(
        pathlib.Path(report["best_checkpoint"]), map_location="cpu", weights_only=False
    )
    tmp = filepath.with_suffix(".pt.tmp")
    torch.save(ckpt, tmp)
    tmp.replace(filepath)
    report_path = output_dir / POST_HOC_EVENT_F1_REPORT_NAME
    with report_path.open("w", encoding="utf-8") as report_file:
        json.dump(report, report_file, indent=2)
    logging.info(
        "Post-hoc event-F1 export: %s @ thr=%s (micro F1 %.5f) -> %s",
        report["best_checkpoint"],
        report["best_threshold"],
        report["best_micro_event_f1"],
        filepath,
    )
    return report


def _iter_limited(loader: Any, take_count: int):
    if take_count < 0:
        return iter(loader)
    return itertools.islice(iter(loader), int(take_count))


def _binary_focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> torch.Tensor:
    logits_f = logits.float().reshape(targets.shape)
    targets_f = targets.float()
    bce = nn_func.binary_cross_entropy_with_logits(
        logits_f, targets_f, reduction="none"
    )
    probs = torch.sigmoid(logits_f)
    p_t = probs * targets_f + (1.0 - probs) * (1.0 - targets_f)
    alpha_t = alpha * targets_f + (1.0 - alpha) * (1.0 - targets_f)
    modulating = (1.0 - p_t).clamp(min=0.0) ** gamma
    return (alpha_t * modulating * bce).mean()


def _dense_onset_loss(
    model: models.DenseOnsetTorchModel,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    use_gaussian_target: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    logits = model.forward_logits(x)
    probs = torch.sigmoid(logits)
    if use_gaussian_target:
        loss = nn_func.mse_loss(probs.float(), y.float())
    else:
        loss = _binary_focal_loss(logits, y)
    return loss, probs


def _is_better(value: float, best: float | None, mode: str) -> bool:
    if best is None:
        return True
    return value < best if mode == "min" else value > best


def _build_cosine_warmup_schedule(
    total_epochs: int,
    warmup_epochs: int,
    lr_peak: float,
    lr_min: float,
) -> Callable[[int], float]:
    """Build an epoch LR schedule that linearly warms up then cosine-decays.

    Args:
        total_epochs: Total number of training epochs.
        warmup_epochs: Epochs for linear warmup from ``lr_min`` to ``lr_peak``.
        lr_peak: Peak learning rate at end of warmup.
        lr_min: Minimum learning rate at start of warmup and end of decay.

    Returns:
        Callable ``(epoch: int) -> lr: float``.
    """
    decay_epochs = total_epochs - warmup_epochs

    def schedule(epoch: int) -> float:
        if epoch < warmup_epochs:
            if warmup_epochs == 1:
                return lr_peak
            progress = epoch / (warmup_epochs - 1)
            return lr_min + (lr_peak - lr_min) * progress
        progress = (epoch - warmup_epochs) / max(decay_epochs, 1)
        return lr_min + 0.5 * (lr_peak - lr_min) * (1.0 + math.cos(math.pi * progress))

    return schedule


def run_train_from_config(
    dataset_config: config.OnsetDatasetConfig,
    model_config: config.OnsetModelConfig,
    run_config: config.RunConfig,
) -> tuple[models.DenseOnsetTorchModel, TrainingHistory]:
    """Train a PyTorch dense onset model using configuration objects.

    Args:
        dataset_config: Dataset configuration.
        model_config: Dense onset architecture configuration.
        run_config: Training run hyperparameters and output paths.

    Returns:
        Tuple ``(model, history)``.
    """
    wsl_gpu.guard_pytorch_gpu_job()
    if run_config.seed is not None:
        reproducibility.apply_training_seed(run_config.seed)

    train_split = None
    val_split = None
    train_ref = dataset_config.data_dir
    val_ref = dataset_config.val_data_dir
    data_root = dataset_config.data_root or dataset_config.data_dir

    index_ref = str(dataset_config.training_index_path).strip()
    if index_ref:
        index_path = pathlib.Path(index_ref)
        index = training_index.load_training_index(index_path)
        data_root = str(training_index.resolve_output_dir(index, index_path))
        train_ref = index_ref
        val_ref = index_ref
        train_split = training_index.SPLIT_TRAIN
        val_split = training_index.SPLIT_VAL
    elif training_index.manifest_split_enabled(
        dataset_config.data_dir,
        dataset_config.val_data_dir,
    ):
        train_split = training_index.SPLIT_TRAIN
        val_split = training_index.SPLIT_VAL

    input_features = config.resolve_onset_input_features(dataset_config, model_config)
    train_loader = datasets.create_dataset(
        data_dir=train_ref,
        batch_size=dataset_config.batch_size,
        apply_temporal_augment=dataset_config.apply_temporal_augment,
        should_apply_spec_augment=dataset_config.should_apply_spec_augment,
        use_gaussian_target=dataset_config.use_gaussian_target,
        gaussian_sigma=dataset_config.gaussian_sigma,
        feature_source=dataset_config.feature_source,
        mert_features_dir=dataset_config.mert_features_dir,
        n_features=input_features,
        max_songs=dataset_config.max_train_songs,
        song_selection_seed=run_config.seed,
        split=train_split,
        data_root=data_root,
        window_frames=dataset_config.train_window_frames,
        windows_per_song=dataset_config.train_windows_per_song,
        density_conditioning=dataset_config.density_conditioning,
    )

    val_batch_size = (
        1 if dataset_config.train_window_frames > 0 else dataset_config.batch_size
    )
    val_loader = datasets.create_dataset(
        data_dir=val_ref,
        batch_size=val_batch_size,
        apply_temporal_augment=False,
        should_apply_spec_augment=False,
        use_gaussian_target=False,
        feature_source=dataset_config.feature_source,
        mert_features_dir=dataset_config.mert_features_dir,
        n_features=input_features,
        split=val_split,
        data_root=data_root,
        density_conditioning=dataset_config.density_conditioning,
    )

    experiment_name = _get_onset_experiment_name(
        take_count=run_config.take_count,
        apply_temporal_augment=dataset_config.apply_temporal_augment,
        should_apply_spec_augment=dataset_config.should_apply_spec_augment,
        use_gaussian_target=dataset_config.use_gaussian_target,
        gaussian_sigma=dataset_config.gaussian_sigma,
        model_params=model_config,
        feature_source=dataset_config.feature_source,
    )
    model_name = run_config.model_name or experiment_name
    if config.uses_waveform_model_input(dataset_config):
        model: models.DenseOnsetTorchModel = (
            models.build_unet_wavenet_from_waveform_model(
                model_name=model_name,
                initial_filters=model_config.initial_filters,
                depth=model_config.depth,
                dilation_rates=model_config.dilation_rates,
                kernel_size=model_config.kernel_size,
                dropout_rate=model_config.dropout_rate,
                frontend_filters=model_config.waveform_frontend_filters,
            )
        )
    else:
        model = models.build_onset_dense_model(
            model_config,
            model_name=model_name,
            input_features=input_features,
        )

    device = runtime.resolve_device(None)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, eps=1e-7)

    experiment_config = config.OnsetExperimentConfig(
        dataset=dataset_config, model=model_config, run=run_config
    )
    monitor_metric = run_config.checkpoint_metric or ONSET_CHECKPOINT_MONITOR
    monitor_mode = "min" if monitor_metric.endswith("loss") else "max"

    callback_name = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    if experiment_name:
        callback_name = f"{callback_name}-{experiment_name}"
    writer: SummaryWriter | None = None
    ckpt_dir: pathlib.Path | None = None
    if run_config.callback_root_dir:
        _save_config(experiment_config, run_config.callback_root_dir, callback_name)
        writer = runtime.create_summary_writer(
            pathlib.Path(run_config.callback_root_dir) / "logs" / callback_name
        )
        ckpt_dir = pathlib.Path(run_config.callback_root_dir) / "models" / callback_name
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    event_cb: dense_overfit_eval.DenseValEventF1Callback | None = None
    if monitor_metric in config.EVENT_CHECKPOINT_METRICS:
        val_batches = list(_iter_limited(val_loader, run_config.val_take_count))
        event_cb = dense_overfit_eval.DenseValEventF1Callback(
            val_batches,
            confidence_threshold=run_config.confidence_threshold,
            tolerance_sec=run_config.tolerance_sec,
            min_onset_distance_ms=run_config.min_onset_distance_ms,
            model=model,
        )

    history = TrainingHistory()
    best_metric_val: float | None = None
    best_state: dict[str, torch.Tensor] | None = None
    stale_epochs = 0
    patience = int(getattr(run_config, "early_stopping_patience", 0))

    try:
        for epoch in range(int(run_config.epoch)):
            model.train()
            tr_f1 = metrics.OnsetF1Metric(tolerance=2, threshold=0.5)
            tr_loss_sum = 0.0
            tr_steps = 0
            for x_batch, y_batch in _iter_limited(train_loader, run_config.take_count):
                x_t = x_batch.to(device)
                y_t = y_batch.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss, probs = _dense_onset_loss(
                    model,
                    x_t,
                    y_t,
                    use_gaussian_target=dataset_config.use_gaussian_target,
                )
                loss.backward()
                optimizer.step()
                tr_loss_sum += float(loss.detach().cpu().item())
                tr_steps += 1
                tr_f1.update_state(y_t.detach(), probs.detach())

            model.eval()
            va_f1 = metrics.OnsetF1Metric(tolerance=2, threshold=0.5)
            va_loss_sum = 0.0
            va_steps = 0
            with torch.no_grad():
                for x_batch, y_batch in _iter_limited(
                    val_loader, run_config.val_take_count
                ):
                    x_t = x_batch.to(device)
                    y_t = y_batch.to(device)
                    v_loss, v_probs = _dense_onset_loss(
                        model, x_t, y_t, use_gaussian_target=False
                    )
                    va_loss_sum += float(v_loss.detach().cpu().item())
                    va_steps += 1
                    va_f1.update_state(y_t, v_probs)

            epoch_logs: dict[str, float] = {
                "loss": tr_loss_sum / max(1, tr_steps),
                "onset_f1_score": tr_f1.result(),
                "val_loss": va_loss_sum / max(1, va_steps),
                "val_onset_f1_score": va_f1.result(),
            }
            if event_cb is not None:
                event_cb.model = model
                event_cb.on_epoch_end(epoch, epoch_logs)

            for k, v in epoch_logs.items():
                history.history.setdefault(k, []).append(float(v))
                if writer is not None:
                    writer.add_scalar(k, float(v), epoch + 1)

            monitored = epoch_logs.get(monitor_metric, epoch_logs["val_loss"])
            if _is_better(monitored, best_metric_val, monitor_mode):
                best_metric_val = monitored
                best_state = copy.deepcopy(
                    {k: v.detach().cpu() for k, v in model.state_dict().items()}
                )
                stale_epochs = 0
                if ckpt_dir is not None:
                    ckpt_name = f"{monitor_metric.upper()}-{monitored:.5f}.pt"
                    core_trainers.save_core_checkpoint(
                        model,
                        ckpt_dir / ckpt_name,
                        extra={"epoch": epoch + 1, monitor_metric: monitored},
                    )
            else:
                stale_epochs += 1
                if patience > 0 and stale_epochs >= patience:
                    break
    finally:
        if writer is not None:
            writer.close()

    if best_state is not None:
        model.load_state_dict(best_state)

    _write_model(
        model,
        run_config.model_output_dir,
        callback_root_dir=run_config.callback_root_dir,
        monitor_metric=monitor_metric,
        monitor_mode=monitor_mode,
        fallback_name=model_name,
    )

    if run_config.post_hoc_event_f1_export and run_config.callback_root_dir:
        _export_best_event_f1_checkpoint(
            model,
            dataset_config,
            model_config,
            run_config,
            monitor_metric=monitor_metric,
        )

    saved_path = pathlib.Path(run_config.model_output_dir) / _model_filename(
        model, model_name
    )
    if saved_path.is_file():
        model = core_trainers.load_onset_model(saved_path, device=device)
        model.name = model_name  # type: ignore[attr-defined]

    return model, history


def run_train(
    *,
    data_dir: str,
    val_data_dir: str,
    batch_size: int,
    apply_temporal_augment: bool,
    should_apply_spec_augment: bool,
    use_gaussian_target: bool,
    gaussian_sigma: float,
    take_count: int,
    epoch: int,
    model_output_dir: str,
    callback_root_dir: str = "",
    model_name: str = "",
    val_take_count: int = -1,
    model_params: dict | None = None,
) -> tuple[models.DenseOnsetTorchModel, TrainingHistory]:
    """Train a PyTorch dense onset model from keyword arguments."""
    dataset_config = config.OnsetDatasetConfig(
        data_dir=data_dir,
        val_data_dir=val_data_dir,
        batch_size=batch_size,
        apply_temporal_augment=apply_temporal_augment,
        should_apply_spec_augment=should_apply_spec_augment,
        use_gaussian_target=use_gaussian_target,
        gaussian_sigma=gaussian_sigma,
    )
    model_config = config.OnsetModelConfig(**(model_params or {}))
    run_config = config.RunConfig(
        epoch=epoch,
        take_count=take_count,
        model_output_dir=model_output_dir,
        callback_root_dir=callback_root_dir,
        model_name=model_name,
        val_take_count=val_take_count,
    )
    return run_train_from_config(dataset_config, model_config, run_config)


def _split_arrow_batch(
    feats: dict[str, torch.Tensor] | torch.Tensor,
    cols: torch.Tensor,
    device: torch.device,
) -> tuple[
    dict[str, torch.Tensor] | torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor | None,
]:
    y_arrow = cols.to(device=device, dtype=torch.long)
    if isinstance(feats, dict):
        aux_target = feats.get("aux_interval_target")
        aux_mask = feats.get("aux_interval_mask")
        x = {
            k: v.to(device=device, dtype=torch.float32)
            for k, v in feats.items()
            if k not in ("aux_interval_target", "aux_interval_mask")
        }
        return (
            x,
            y_arrow,
            aux_target.to(device=device, dtype=torch.float32)
            if aux_target is not None
            else None,
            aux_mask.to(device=device, dtype=torch.float32)
            if aux_mask is not None
            else None,
        )
    return feats.to(device=device, dtype=torch.float32), y_arrow, None, None


def run_arrow_train_from_config(
    experiment_config: config.ArrowExperimentConfig,
) -> tuple[models.ArrowSequenceModel, TrainingHistory]:
    """Train a PyTorch arrow classification model from an experiment config.

    Args:
        experiment_config: Full arrow experiment configuration.

    Returns:
        Tuple ``(model, history)``.
    """
    wsl_gpu.guard_pytorch_gpu_job()
    dataset_config = experiment_config.dataset
    model_config = experiment_config.model
    run_config = experiment_config.run
    if run_config.seed is not None:
        reproducibility.apply_training_seed(run_config.seed)

    dataset_provides_aux = dataset_config.use_aux_interval_target
    use_aux_interval = dataset_provides_aux and run_config.aux_interval_weight > 0

    train_loader = datasets.create_arrow_dataset(
        data_dir=dataset_config.data_dir,
        batch_size=dataset_config.batch_size,
        snippet_half_frames=dataset_config.snippet_half_frames,
        use_interval=dataset_config.use_interval,
        interval_encoding=dataset_config.interval_encoding,
        use_step_index=dataset_config.use_step_index,
        use_beat_phase=dataset_config.use_beat_phase,
        use_aux_interval_target=dataset_config.use_aux_interval_target,
        timing_jitter_sigma=dataset_config.timing_jitter_sigma,
    )
    val_loader = datasets.create_arrow_dataset(
        data_dir=dataset_config.val_data_dir,
        batch_size=dataset_config.batch_size,
        snippet_half_frames=dataset_config.snippet_half_frames,
        use_interval=dataset_config.use_interval,
        interval_encoding=dataset_config.interval_encoding,
        use_step_index=dataset_config.use_step_index,
        use_beat_phase=dataset_config.use_beat_phase,
        use_aux_interval_target=dataset_config.use_aux_interval_target,
        timing_jitter_sigma=0.0,
    )

    experiment_name = _get_arrow_experiment_name(experiment_config)
    model_name = run_config.model_name or experiment_name
    input_options = models.ArrowInputOptions(
        snippet_half_frames=dataset_config.snippet_half_frames,
        use_interval=dataset_config.use_interval,
        interval_encoding=dataset_config.interval_encoding,
        use_step_index=dataset_config.use_step_index,
        use_beat_phase=dataset_config.use_beat_phase,
    )
    output_options = models.ArrowOutputOptions(
        use_aux_interval=use_aux_interval,
        model_name=model_name,
    )
    model = models.build_arrow_model_from_config(
        model_config, input_options, output_options
    )

    device = runtime.resolve_device(None)
    model.to(device)
    arrow_combined_loss = losses.build_arrow_combined_loss(run_config)

    use_lr_schedule = run_config.warmup_epochs > 0
    initial_lr = run_config.lr_min if use_lr_schedule else run_config.lr_peak
    optimizer = torch.optim.Adam(model.parameters(), lr=float(initial_lr), eps=1e-7)
    lr_schedule = (
        _build_cosine_warmup_schedule(
            total_epochs=run_config.epoch,
            warmup_epochs=run_config.warmup_epochs,
            lr_peak=run_config.lr_peak,
            lr_min=run_config.lr_min,
        )
        if use_lr_schedule
        else None
    )

    monitor_metric = (
        "val_output_probabilities_main_loss" if use_aux_interval else "val_main_loss"
    )

    callback_name = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    if experiment_name:
        callback_name = f"{callback_name}-{experiment_name}"
    writer: SummaryWriter | None = None
    ckpt_dir: pathlib.Path | None = None
    if run_config.callback_root_dir:
        _save_config(experiment_config, run_config.callback_root_dir, callback_name)
        writer = runtime.create_summary_writer(
            pathlib.Path(run_config.callback_root_dir) / "logs" / callback_name
        )
        ckpt_dir = pathlib.Path(run_config.callback_root_dir) / "models" / callback_name
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    history = TrainingHistory()
    best_val: float | None = None
    best_state: dict[str, torch.Tensor] | None = None
    stale_epochs = 0
    patience = int(getattr(run_config, "early_stopping_patience", 0))
    aux_w = float(run_config.aux_interval_weight)

    def _run_epoch(loader: Any, limit: int, train: bool) -> dict[str, float]:
        model.train(train)
        loss_sum = 0.0
        main_ce_sum = 0.0
        steps = 0
        correct = 0.0
        valid_total = 0.0
        val_metric = metrics.ChartValidityMetric()
        dist_metric = metrics.ArrowDistributionMatchMetric()
        kind_metric = metrics.ArrowNoteKindDistributionMetric()

        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            for raw_feats, raw_cols in _iter_limited(loader, limit):
                x, y_arrow, y_aux, aux_mask = _split_arrow_batch(
                    raw_feats, raw_cols, device
                )
                if train:
                    optimizer.zero_grad(set_to_none=True)
                out = model(x)
                if isinstance(out, dict):
                    probs = out["output_probabilities"]
                    total_loss = arrow_combined_loss(y_arrow, probs)
                    if use_aux_interval and y_aux is not None:
                        total_loss = (
                            total_loss
                            + aux_w
                            * losses.masked_mse_aux_interval(
                                y_aux, out["aux_interval"], aux_mask
                            )
                        )
                else:
                    probs = out
                    total_loss = arrow_combined_loss(y_arrow, probs)
                if train:
                    total_loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

                main_ce = losses.arrow_label_smoothed_crossentropy(
                    y_arrow, probs.detach(), smoothing=0.0
                )
                loss_sum += float(total_loss.detach().cpu().item())
                main_ce_sum += float(main_ce.detach().cpu().item())
                steps += 1

                valid_mask = y_arrow != constants.ARROW_PADDING_CLASS
                preds = torch.argmax(probs.detach(), dim=-1)
                correct += float(((preds == y_arrow) & valid_mask).sum().cpu().item())
                valid_total += float(valid_mask.sum().cpu().item())
                val_metric.update_state(y_arrow.detach(), probs.detach())
                dist_metric.update_state(y_arrow.detach(), probs.detach())
                kind_metric.update_state(y_arrow.detach(), probs.detach())

        return {
            "loss": loss_sum / max(1, steps),
            "main_loss": main_ce_sum / max(1, steps),
            "acc": correct / max(1.0, valid_total),
            "chart_validity": val_metric.result(),
            "arrow_dist_match": dist_metric.result(),
            "arrow_note_kind_dist_match": kind_metric.result(),
        }

    try:
        for epoch in range(int(run_config.epoch)):
            if lr_schedule is not None:
                lr = lr_schedule(epoch)
                for group in optimizer.param_groups:
                    group["lr"] = lr
            tr_stats = _run_epoch(train_loader, run_config.take_count, train=True)
            va_stats = _run_epoch(val_loader, run_config.val_take_count, train=False)

            epoch_logs: dict[str, float] = {
                **tr_stats,
                **{f"val_{k}": v for k, v in va_stats.items()},
            }
            if use_aux_interval:
                epoch_logs["val_output_probabilities_main_loss"] = va_stats["main_loss"]

            for k, v in epoch_logs.items():
                history.history.setdefault(k, []).append(float(v))
                if writer is not None:
                    writer.add_scalar(k, float(v), epoch + 1)

            monitored = epoch_logs[monitor_metric]
            if _is_better(monitored, best_val, "min"):
                best_val = monitored
                best_state = copy.deepcopy(
                    {k: v.detach().cpu() for k, v in model.state_dict().items()}
                )
                stale_epochs = 0
                if ckpt_dir is not None:
                    ckpt_name = f"{monitor_metric.upper()}-{monitored:.5f}.pt"
                    core_trainers.save_core_checkpoint(
                        model,
                        ckpt_dir / ckpt_name,
                        extra={"epoch": epoch + 1, monitor_metric: monitored},
                    )
            else:
                stale_epochs += 1
                if patience > 0 and stale_epochs >= patience:
                    break
    finally:
        if writer is not None:
            writer.close()

    if best_state is not None:
        model.load_state_dict(best_state)

    _write_model(
        model,
        run_config.model_output_dir,
        callback_root_dir=run_config.callback_root_dir,
        monitor_metric=monitor_metric,
        monitor_mode="min",
        fallback_name=model_name,
    )
    saved_path = pathlib.Path(run_config.model_output_dir) / _model_filename(
        model, model_name
    )
    if saved_path.is_file():
        model = core_trainers.load_arrow_model(saved_path, device=device)
        model.name = model_name  # type: ignore[attr-defined]

    return model, history


def run_arrow_train(
    *,
    data_dir: str,
    val_data_dir: str,
    batch_size: int,
    epoch: int,
    model_output_dir: str,
    take_count: int = -1,
    callback_root_dir: str = "",
    model_name: str = "",
    val_take_count: int = -1,
    model_params: dict | None = None,
) -> tuple[models.ArrowSequenceModel, TrainingHistory]:
    """Train a PyTorch arrow classification model from keyword arguments."""
    dataset_config = config.ArrowDatasetConfig(
        data_dir=data_dir,
        val_data_dir=val_data_dir,
        batch_size=batch_size,
    )
    model_config = config.ArrowModelConfig.from_dict(model_params or {})
    run_config = config.ArrowRunConfig(
        epoch=epoch,
        take_count=take_count,
        model_output_dir=model_output_dir,
        callback_root_dir=callback_root_dir,
        model_name=model_name,
        val_take_count=val_take_count,
    )
    experiment_config = config.ArrowExperimentConfig(
        dataset=dataset_config, model=model_config, run=run_config
    )
    return run_arrow_train_from_config(experiment_config)
