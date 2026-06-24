"""Launches the main training loop to train a new StepCovNet model."""

import datetime
import json
import logging
import math
import os
import pathlib

import keras
import tensorflow as tf

from stepcovnet import (
    config,
    datasets,
    dense_overfit_eval,
    losses,
    metrics,
    models,
    reproducibility,
)
from stepcovnet.dataset_prep import training_index

ONSET_CHECKPOINT_MONITOR = "val_onset_f1_score"


def _get_tb_callback(root_dir: str, callback_name: str):
    """Create a TensorBoard callback for logging training metrics.

    Args:
        root_dir: Root directory for storing logs.
        callback_name: Name of the callback/run directory.

    Returns:
        TensorBoard callback configured to log to the specified directory.
    """
    logdir = pathlib.Path(root_dir) / "logs" / callback_name
    return keras.callbacks.TensorBoard(
        str(logdir), histogram_freq=0, write_images=False, embeddings_freq=0
    )


def _get_ckpt_callback(
    root_dir: str,
    callback_name: str,
    monitor_metric: str,
    mode: str,
) -> keras.callbacks.ModelCheckpoint:
    """Create a model checkpoint callback for saving the best model.

    Args:
        root_dir: Root directory for storing model checkpoints.
        callback_name: Name of the callback/run directory.
        monitor_metric: Metric name to monitor for checkpointing (e.g., 'val_loss').
        mode: Mode for monitoring ('min' or 'max'). 'min' saves when metric decreases,
            'max' saves when metric increases.

    Returns:
        ModelCheckpoint callback configured to save the best model based on
        the monitored metric.
    """
    ckpt_path = (
        pathlib.Path(root_dir)
        / "models"
        / callback_name
        / (f"{monitor_metric.upper()}" + "-{" + f"{monitor_metric}" + ":.5f}.keras")
    )
    model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath=str(ckpt_path),
        monitor=monitor_metric,
        save_best_only=True,
        mode=mode,
    )
    return model_checkpoint_callback


def _get_callbacks(
    root_dir: str,
    monitor_metric: str,
    monitor_mode: str,
    experiment_name: str = "",
    early_stopping_patience: int = 0,
) -> tuple[list[keras.callbacks.Callback], str]:
    """Get training callbacks and return the callback name.

    Creates TensorBoard and ModelCheckpoint callbacks for monitoring and
    saving the best model during training. The callback name includes a
    timestamp and optional experiment name.

    Args:
        root_dir: Root directory for storing callbacks (logs and checkpoints).
        monitor_metric: Metric name to monitor for checkpointing.
        monitor_mode: Mode for monitoring ('min' or 'max').
        experiment_name: Optional experiment name to append to the callback name.
        early_stopping_patience: Epochs without improvement before stopping; 0 disables.

    Returns:
        Tuple containing:
            - List of Keras callbacks (TensorBoard and ModelCheckpoint).
            - Callback name string (timestamp + optional experiment name).
    """
    now = datetime.datetime.now()
    callback_name = now.strftime("%Y%m%d-%H%M%S")
    if experiment_name:
        callback_name = callback_name + "-" + experiment_name
    training_callbacks = [
        _get_tb_callback(root_dir, callback_name),
        _get_ckpt_callback(root_dir, callback_name, monitor_metric, monitor_mode),
    ]
    if early_stopping_patience > 0:
        training_callbacks.append(
            keras.callbacks.EarlyStopping(
                monitor=monitor_metric,
                mode=monitor_mode,
                patience=early_stopping_patience,
                restore_best_weights=True,
                verbose=1,
            )
        )
    return training_callbacks, callback_name


def _get_onset_experiment_name(
    take_count: int,
    apply_temporal_augment: bool,
    should_apply_spec_augment: bool,
    use_gaussian_target: bool,
    gaussian_sigma: float,
    model_params: config.OnsetModelConfig,
    feature_source: config.FeatureSource = config.FeatureSource.MEL,
) -> str:
    """Generate a descriptive experiment name from hyperparameters.

    Creates a human-readable name that encodes key training and model
    configuration parameters. Used for organizing runs and checkpoints.

    Args:
        take_count: Number of batches used from training dataset.
        apply_temporal_augment: Whether temporal augmentation was applied.
        should_apply_spec_augment: Whether spectrogram augmentation was applied.
        use_gaussian_target: Whether Gaussian targets were used.
        gaussian_sigma: Standard deviation for Gaussian targets.
        model_params: Model configuration OnsetModelConfig object
            containing architecture parameters.

    Returns:
        String experiment name with format:
        "ONSET-take_{N}-sigma_{X}-temporal_augment-spec_augment-unet_filters_{N}-..."
    """
    parts = ["ONSET"]

    if feature_source == config.FeatureSource.MERT:
        parts.append("mert")
    elif feature_source == config.FeatureSource.WAVEFORM:
        parts.append("waveform")

    if take_count == -1:
        parts.append("take_all")
    else:
        parts.append(f"take_{take_count}")

    if use_gaussian_target:
        sigma_str = str(gaussian_sigma).replace(".", "_")
        parts.append(f"sigma_{sigma_str}")

    if apply_temporal_augment:
        parts.append("temporal_augment")

    if should_apply_spec_augment:
        parts.append("spec_augment")

    arch = model_params.onset_architecture
    parts.append(arch.value)

    initial_filters = model_params.initial_filters
    depth = model_params.depth
    kernel_size = model_params.kernel_size
    dropout_rate = model_params.dropout_rate
    dilation_rates = model_params.dilation_rates

    parts.append(f"filters_{initial_filters}")
    parts.append(f"dropout_{str(dropout_rate).replace('.', '_')}")

    if arch == config.OnsetArchitecture.BILSTM:
        parts.append(f"bilstm_depth_{depth}")
        parts.append(f"bilstm_units_{model_params.recurrent_units}")
    elif arch == config.OnsetArchitecture.TRANSFORMER:
        parts.append(f"tfm_layers_{model_params.transformer_layers}")
        parts.append(f"tfm_heads_{model_params.transformer_heads}")
    elif arch == config.OnsetArchitecture.TCN:
        parts.append(f"tcn_blocks_{model_params.tcn_blocks}")
        parts.append(f"kernel_{kernel_size}")
        if dilation_rates is None:
            dilation_str = "N_A"
        elif isinstance(dilation_rates, list | tuple):
            dilation_str = "_".join(map(str, dilation_rates))
        else:
            dilation_str = str(dilation_rates)
        parts.append(f"dilations_{dilation_str}")
    else:
        parts.append(f"depth_{depth}")
        parts.append(f"kernel_{kernel_size}")
        if dilation_rates is None:
            dilation_str = "N_A"
        elif isinstance(dilation_rates, list | tuple):
            dilation_str = "_".join(map(str, dilation_rates))
        else:
            dilation_str = str(dilation_rates)
        parts.append(f"dilations_{dilation_str}")

    return "-".join(parts)


def _get_arrow_experiment_name(
    experiment_config: config.ArrowExperimentConfig,
) -> str:
    """Generate a descriptive experiment name from hyperparameters.

    Creates a human-readable name that encodes key training and model
    configuration parameters for arrow classification experiments.
    All distinguishing model, run, and dataset fields are provided by the
    configs' get_experiment_name_parts() so that different configs yield different names.

    Args:
        experiment_config: Full arrow experiment configuration (dataset, model, run).

    Returns:
        String experiment name, e.g. "ARROW-transformer-take_all-att_layers_1-...".
    """
    model_config = experiment_config.model
    run_config = experiment_config.run
    dataset_config = experiment_config.dataset
    parts = ["ARROW", model_config.model_type]
    parts.extend(model_config.get_experiment_name_parts())
    parts.extend(run_config.get_experiment_name_parts())
    parts.extend(dataset_config.get_experiment_name_parts())
    return "-".join(parts)


def _list_monitored_checkpoints(
    callback_root_dir: str,
    monitor_metric: str,
) -> list[str]:
    """Return all monitored checkpoint paths under a callback root, sorted by path."""
    models_root = pathlib.Path(callback_root_dir) / "models"
    if not models_root.is_dir():
        return []
    prefix = f"{monitor_metric.upper()}-"
    paths: list[str] = []
    for root, _dirs, files in os.walk(models_root):
        for name in files:
            if not name.startswith(prefix) or not name.endswith(".keras"):
                continue
            value_text = name[len(prefix) : -len(".keras")]
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
    return float(name[len(prefix) : -len(".keras")])


def _latest_monitored_checkpoint(
    callback_root_dir: str,
    monitor_metric: str,
) -> str | None:
    """Return the checkpoint with the best monitored metric under a callback root."""
    paths = _list_monitored_checkpoints(callback_root_dir, monitor_metric)
    if not paths:
        return None
    return max(
        paths,
        key=lambda path: _monitored_checkpoint_value(path, monitor_metric),
    )


def _write_model(
    model: keras.Model,
    model_output_dir: str,
    *,
    callback_root_dir: str = "",
    monitor_metric: str = ONSET_CHECKPOINT_MONITOR,
):
    """Save the trained Keras model to the specified directory.

    When ``callback_root_dir`` contains a monitored checkpoint, that model is
    saved instead of the final-epoch weights.
    """
    filepath = pathlib.Path(model_output_dir) / f"{model.name}.keras"
    filepath.parent.mkdir(parents=True, exist_ok=True)
    best_path = _latest_monitored_checkpoint(callback_root_dir, monitor_metric)
    if best_path is not None:
        logging.info("Saving best checkpoint from %s to %s", best_path, filepath)
        best_model = keras.models.load_model(best_path, compile=False)
        best_model.save(filepath=str(filepath))
        return
    logging.info("Saving trained model to %s", filepath)
    model.save(filepath=str(filepath))


POST_HOC_EVENT_F1_REPORT_NAME = "event_f1_sweep.json"


def _select_best_event_f1_checkpoint(
    dataset_config: config.OnsetDatasetConfig,
    model_config: config.OnsetModelConfig,
    run_config: config.RunConfig,
    *,
    monitor_metric: str = ONSET_CHECKPOINT_MONITOR,
) -> dict | None:
    """Sweep every saved checkpoint and threshold for best validation event F1.

    Returns a report dict describing the best (checkpoint, threshold) pair and the
    per-checkpoint sweeps, or None when no monitored checkpoints exist.
    """
    checkpoint_paths = _list_monitored_checkpoints(
        run_config.callback_root_dir,
        monitor_metric,
    )
    if not checkpoint_paths:
        return None
    thresholds = tuple(run_config.post_hoc_event_f1_thresholds)
    per_checkpoint: list[dict] = []
    best_entry: dict | None = None
    for checkpoint_path in checkpoint_paths:
        checkpoint_model = keras.models.load_model(checkpoint_path, compile=False)
        sweep = dense_overfit_eval.sweep_thresholds_dense_val_event_f1(
            checkpoint_model,
            dataset_config,
            model_config,
            thresholds=thresholds,
            min_onset_distance_ms=run_config.min_onset_distance_ms,
            tolerance_sec=run_config.tolerance_sec,
        )
        entry = {"checkpoint": checkpoint_path, **sweep}
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
    model: keras.Model,
    dataset_config: config.OnsetDatasetConfig,
    model_config: config.OnsetModelConfig,
    run_config: config.RunConfig,
    *,
    monitor_metric: str = ONSET_CHECKPOINT_MONITOR,
) -> dict | None:
    """Export the event-F1-optimal checkpoint and write its sweep report.

    Overrides the frame-F1 model file under ``model_output_dir`` with the
    checkpoint that maximizes validation peak-pick event F1. Returns the report
    dict (also written to ``event_f1_sweep.json``), or None when no checkpoints
    are available to sweep.
    """
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
    filepath = output_dir / f"{model.name}.keras"
    best_model = keras.models.load_model(report["best_checkpoint"], compile=False)
    best_model.save(filepath=str(filepath))
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


def _save_config(
    experiment_config: config.OnsetExperimentConfig | config.ArrowExperimentConfig,
    callback_root_dir: str,
    callback_name: str,
):
    """Save experiment config to JSON file in the run directory.

    Saves the complete experiment configuration (dataset, model, and run
    parameters) to a JSON file in the run's log directory. This enables
    reproducibility by allowing the exact configuration to be reloaded
    for re-running or comparing experiments.

    Args:
        experiment_config: The experiment configuration to save. Can be
            either OnsetExperimentConfig or ArrowExperimentConfig.
        callback_root_dir: Root directory for storing callbacks.
        callback_name: Name of the callback/run directory where the config
            will be saved.

    The config is saved to: {callback_root_dir}/logs/{callback_name}/config.json
    """
    logdir = pathlib.Path(callback_root_dir) / "logs" / callback_name
    logdir.mkdir(parents=True, exist_ok=True)
    config_path = logdir / "config.json"
    experiment_config.to_json(str(config_path))
    logging.info(f"Saved experiment config to {config_path}")


def _build_experiment_callbacks(
    run_config: config.RunConfig | config.ArrowRunConfig,
    experiment_name: str,
    monitor_metric: str,
    monitor_mode: str,
    experiment_config: config.OnsetExperimentConfig | config.ArrowExperimentConfig,
) -> list[keras.callbacks.Callback]:
    """Build callbacks for an experiment and save its config if callbacks enabled.

    Args:
        run_config: Training run configuration (onset or arrow).
        experiment_name: Human-readable experiment name for log directories.
        monitor_metric: Metric name to monitor for checkpointing.
        monitor_mode: Mode for monitoring ('min' or 'max').
        experiment_config: Combined dataset/model/run configuration to persist.

    Returns:
        List of Keras callbacks (empty when callback_root_dir is not set).
    """
    if not run_config.callback_root_dir:
        return []

    training_callbacks, callback_name = _get_callbacks(
        root_dir=run_config.callback_root_dir,
        monitor_metric=monitor_metric,
        monitor_mode=monitor_mode,
        experiment_name=experiment_name,
        early_stopping_patience=getattr(run_config, "early_stopping_patience", 0),
    )
    _save_config(experiment_config, run_config.callback_root_dir, callback_name)
    return training_callbacks


def _fit_and_save_model(
    model: keras.Model,
    train_dataset,
    val_dataset,
    run_config: config.RunConfig | config.ArrowRunConfig,
    callbacks: list[keras.callbacks.Callback],
    monitor_metric: str,
) -> keras.callbacks.History:
    """Run model.fit with common settings, then persist the trained model.

    Args:
        model: Compiled Keras model to train.
        train_dataset: Training dataset.
        val_dataset: Validation dataset.
        run_config: Training run configuration (onset or arrow).
        callbacks: List of callbacks to pass to model.fit.
        monitor_metric: Validation metric used for best-checkpoint selection.

    Returns:
        Training history object from model.fit.
    """
    val_data = val_dataset.take(run_config.val_take_count)
    train_history = model.fit(
        train_dataset.take(run_config.take_count),
        epochs=run_config.epoch,
        validation_data=val_data,
        callbacks=callbacks,
        verbose=run_config.fit_verbose,  # type: ignore[arg-type]
    )

    _write_model(
        model,
        run_config.model_output_dir,
        callback_root_dir=run_config.callback_root_dir,
        monitor_metric=monitor_metric,
    )

    return train_history


def build_onset_dense_compile_metrics(
    run_config: config.RunConfig,
) -> list[keras.metrics.Metric]:
    """Return Keras metrics compiled for dense onset training."""
    return [
        keras.metrics.BinaryAccuracy(name="acc"),
        keras.metrics.Precision(name="prec"),
        keras.metrics.Recall(name="rec"),
        keras.metrics.AUC(curve="PR", name="pr_auc"),
        keras.metrics.AUC(name="auc"),
        metrics.OnsetF1Metric(tolerance=2, threshold=0.5),
    ]


def _build_dense_val_event_f1_callback(
    val_dataset: tf.data.Dataset,
    run_config: config.RunConfig,
) -> dense_overfit_eval.DenseValEventF1Callback:
    return dense_overfit_eval.DenseValEventF1Callback(
        val_dataset,
        confidence_threshold=run_config.confidence_threshold,
        tolerance_sec=run_config.tolerance_sec,
        min_onset_distance_ms=run_config.min_onset_distance_ms,
    )


def run_train_from_config(
    dataset_config: config.OnsetDatasetConfig,
    model_config: config.OnsetModelConfig,
    run_config: config.RunConfig,
) -> tuple[keras.Model, keras.callbacks.History]:
    """Train a U-Net WaveNet model using configuration objects.

    This is the recommended way to train models as it provides better tracking
    and reproducibility. The config is automatically saved with the run.

    Args:
        dataset_config: Configuration for dataset creation.
        model_config: Configuration for model architecture.
        run_config: Configuration for training run parameters.

    Returns:
        A tuple containing:
            - model: The trained Keras model.
            - train_history: The training history object containing loss and
            metrics per epoch.
    """
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
        logging.info(
            "Using training index %s (data root %s)",
            index_ref,
            data_root,
        )
    elif training_index.manifest_split_enabled(
        dataset_config.data_dir,
        dataset_config.val_data_dir,
    ):
        train_split = training_index.SPLIT_TRAIN
        val_split = training_index.SPLIT_VAL
        logging.info(
            "Using training_index.json for train/val under %s",
            dataset_config.data_dir,
        )

    if dataset_config.max_train_songs != -1:
        all_train_samples = datasets.list_dense_onset_samples(
            train_ref,
            split=train_split,
        )
        selected_train_samples = datasets.select_song_pairs(
            all_train_samples,
            max_songs=dataset_config.max_train_songs,
            seed=run_config.seed,
        )
        selected_stems = sorted(
            pathlib.Path(audio_path).stem for audio_path, _, _ in selected_train_samples
        )
        logging.info(
            "Training on %d of %d songs (max_train_songs=%d, seed=%s): %s",
            len(selected_train_samples),
            len(all_train_samples),
            dataset_config.max_train_songs,
            run_config.seed,
            ", ".join(selected_stems),
        )

    train_dataset = datasets.create_dataset(
        data_dir=train_ref,
        batch_size=dataset_config.batch_size,
        apply_temporal_augment=dataset_config.apply_temporal_augment,
        should_apply_spec_augment=dataset_config.should_apply_spec_augment,
        use_gaussian_target=dataset_config.use_gaussian_target,
        gaussian_sigma=dataset_config.gaussian_sigma,
        feature_source=dataset_config.feature_source,
        mert_features_dir=dataset_config.mert_features_dir,
        n_features=config.resolve_onset_input_features(dataset_config, model_config),
        max_songs=dataset_config.max_train_songs,
        song_selection_seed=run_config.seed,
        split=train_split,
        data_root=data_root,
    )

    val_dataset = datasets.create_dataset(
        data_dir=val_ref,
        batch_size=dataset_config.batch_size,
        apply_temporal_augment=False,
        should_apply_spec_augment=False,
        use_gaussian_target=False,
        feature_source=dataset_config.feature_source,
        mert_features_dir=dataset_config.mert_features_dir,
        n_features=config.resolve_onset_input_features(dataset_config, model_config),
        split=val_split,
        data_root=data_root,
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

    input_features = config.resolve_onset_input_features(dataset_config, model_config)
    if config.uses_waveform_model_input(dataset_config):
        model = models.build_unet_wavenet_from_waveform_model(
            model_name=run_config.model_name or experiment_name,
            initial_filters=model_config.initial_filters,
            depth=model_config.depth,
            dilation_rates=model_config.dilation_rates,
            kernel_size=model_config.kernel_size,
            dropout_rate=model_config.dropout_rate,
            frontend_filters=model_config.waveform_frontend_filters,
        )
    else:
        model = models.build_onset_dense_model(
            model_config,
            model_name=run_config.model_name or experiment_name,
            input_features=input_features,
        )

    if run_config.show_model_summary:
        model.summary()

    if dataset_config.use_gaussian_target:
        loss = keras.losses.MeanSquaredError()
    else:
        loss = keras.losses.BinaryFocalCrossentropy(apply_class_balancing=True)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),  # type: ignore
        loss=loss,
        metrics=build_onset_dense_compile_metrics(run_config),
    )

    experiment_config = config.OnsetExperimentConfig(
        dataset=dataset_config, model=model_config, run=run_config
    )
    training_callbacks = _build_experiment_callbacks(
        run_config=run_config,
        experiment_name=experiment_name,
        monitor_metric=ONSET_CHECKPOINT_MONITOR,
        monitor_mode="max",
        experiment_config=experiment_config,
    )
    # DenseValEventF1Callback runs an extra full val pass with model.predict each epoch
    # (~30-45 s/epoch on 100-train; ~1.6x wall time vs frame-F1-only). Disabled for
    # faster scaling runs; report peak-pick event F1 post-hoc via eval_dense_onset.py.
    # val_data = val_dataset.take(run_config.val_take_count)
    # event_f1_callback = _build_dense_val_event_f1_callback(val_data, run_config)
    # if training_callbacks:
    #     training_callbacks.insert(1, event_f1_callback)
    # else:
    #     training_callbacks = [event_f1_callback]

    train_history = _fit_and_save_model(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        run_config=run_config,
        callbacks=training_callbacks,
        monitor_metric=ONSET_CHECKPOINT_MONITOR,
    )

    if run_config.post_hoc_event_f1_export and run_config.callback_root_dir:
        _export_best_event_f1_checkpoint(
            model,
            dataset_config,
            model_config,
            run_config,
            monitor_metric=ONSET_CHECKPOINT_MONITOR,
        )

    saved_path = pathlib.Path(run_config.model_output_dir) / f"{model.name}.keras"
    if saved_path.is_file():
        model = keras.models.load_model(saved_path, compile=False)

    return model, train_history


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
) -> tuple[keras.Model, keras.callbacks.History]:
    """Train a U-Net WaveNet model on step detection data.

    Trains a Keras model for detecting steps in audio spectrograms. The function
    handles dataset creation, model compilation with configurable loss
    functions, and training with callbacks for monitoring and checkpointing.
    Input spectrograms are always normalized.

    Args:
        data_dir: Path to the directory containing training data.
        val_data_dir: Path to the directory containing validation data.
        batch_size: Number of samples per batch during training.
        apply_temporal_augment: Whether to apply temporal augmentation to
            training data.
        should_apply_spec_augment: Whether to apply spectrogram augmentation
            to training data.
        use_gaussian_target: Whether to use Gaussian targets (True) or binary
            targets (False).
        gaussian_sigma: Standard deviation for Gaussian target distribution.
        take_count: Number of batches to use from the training dataset.
        epoch: Number of epochs to train for.
        model_output_dir: Directory where the trained model will be saved.
        callback_root_dir: Root directory for storing training callbacks (
            checkpoints, logs).
        model_name: Name of the model that will be saved. If none provided,
            generated from the experiment name.
        val_take_count: Number of batches to use from the validation dataset.
            -1 (default) uses the entire validation dataset.
        model_params: Optional dictionary of parameters to pass to the model
            builder. If omitted or None, default OnsetModelConfig values are used.

    Returns:
        A tuple containing:
            - model: The trained Keras model.
            - train_history: The training history object containing loss and
            metrics per epoch.
    """
    # Convert kwargs to config objects for backward compatibility
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


def _build_cosine_warmup_schedule(
    total_epochs: int,
    warmup_epochs: int,
    lr_peak: float,
    lr_min: float,
) -> keras.callbacks.LearningRateScheduler:
    """Build a LearningRateScheduler that linearly warms up then cosine-decays.

    Args:
        total_epochs: Total number of training epochs.
        warmup_epochs: Epochs for linear warmup from lr_min to lr_peak.
        lr_peak: Peak (maximum) learning rate reached at end of warmup.
        lr_min: Minimum learning rate at start of warmup and end of decay.

    Returns:
        A Keras LearningRateScheduler callback.
    """
    decay_epochs = total_epochs - warmup_epochs

    def schedule(epoch: int, _lr: float) -> float:
        if epoch < warmup_epochs:
            # Reach lr_peak at the last warmup epoch (epoch == warmup_epochs - 1).
            if warmup_epochs == 1:
                return lr_peak
            progress = epoch / (warmup_epochs - 1)
            return lr_min + (lr_peak - lr_min) * progress
        progress = (epoch - warmup_epochs) / max(decay_epochs, 1)
        return lr_min + 0.5 * (lr_peak - lr_min) * (1.0 + math.cos(math.pi * progress))

    return keras.callbacks.LearningRateScheduler(schedule)


def run_arrow_train_from_config(
    experiment_config: config.ArrowExperimentConfig,
) -> tuple[keras.Model, keras.callbacks.History]:
    """Train an arrow classification model using an experiment config.

    This is the recommended way to train models as it provides better tracking
    and reproducibility. The config is automatically saved with the run.

    Args:
        experiment_config: Full arrow experiment configuration (dataset, model, run).

    Returns:
        A tuple containing:
            - model: The trained Keras model.
            - train_history: The training history object containing loss and
            metrics per epoch.
    """
    dataset_config = experiment_config.dataset
    model_config = experiment_config.model
    run_config = experiment_config.run
    if run_config.seed is not None:
        reproducibility.apply_training_seed(run_config.seed)

    dataset_provides_aux = dataset_config.use_aux_interval_target
    use_aux_interval = dataset_provides_aux and run_config.aux_interval_weight > 0
    train_dataset = datasets.create_arrow_dataset(
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

    val_dataset = datasets.create_arrow_dataset(
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

    if dataset_provides_aux:

        def _prepare_aux_batch(out, cols):
            x = {
                k: v
                for k, v in out.items()
                if k not in ("aux_interval_target", "aux_interval_mask")
            }
            if use_aux_interval:
                y = {
                    "output_probabilities": cols,
                    "aux_interval": out["aux_interval_target"],
                }
                sample_weight = {
                    # Uniform weight for the main classification head.
                    "output_probabilities": tf.ones_like(cols, dtype=tf.float32),
                    # Masked weights for aux_interval regression (last step masked out).
                    "aux_interval": out["aux_interval_mask"],
                }
                return (x, y, sample_weight)
            return (x, cols)

        train_dataset = train_dataset.map(_prepare_aux_batch)
        val_dataset = val_dataset.map(_prepare_aux_batch)

    experiment_name = _get_arrow_experiment_name(experiment_config)

    input_options = models.ArrowInputOptions(
        snippet_half_frames=dataset_config.snippet_half_frames,
        use_interval=dataset_config.use_interval,
        interval_encoding=dataset_config.interval_encoding,
        use_step_index=dataset_config.use_step_index,
        use_beat_phase=dataset_config.use_beat_phase,
    )
    output_options = models.ArrowOutputOptions(
        use_aux_interval=use_aux_interval,
        model_name=run_config.model_name or experiment_name,
    )
    model = models.build_arrow_model_from_config(
        model_config, input_options, output_options
    )

    if run_config.show_model_summary:
        model.summary()

    arrow_combined_loss = losses.build_arrow_combined_loss(run_config)
    rej_threshold = run_config.chart_validity_rejection_threshold

    use_lr_schedule = run_config.warmup_epochs > 0
    initial_lr = run_config.lr_min if use_lr_schedule else run_config.lr_peak

    _main_metrics = [
        keras.metrics.SparseCategoricalAccuracy(name="acc"),
        keras.metrics.SparseCategoricalCrossentropy(name="main_loss"),
        metrics.ChartValidityAuxiliaryLossMetric(name="chart_validity_aux_loss"),
        metrics.NoteKindBalanceAuxiliaryLossMetric(name="note_kind_balance_aux_loss"),
        metrics.ArrowDistributionMatchMetric(name="arrow_dist_match"),
        metrics.ArrowNoteKindDistributionMetric(name="arrow_note_kind_dist_match"),
        metrics.ChartValidityMetric(name="chart_validity"),
    ]
    if rej_threshold is not None:
        _main_metrics.append(
            metrics.ChartValidityPassRateMetric(
                threshold=rej_threshold,
                name=f"chart_validity_pass_rate_{str(rej_threshold).replace('.', '_')}",
            )
        )

    if use_aux_interval:
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=initial_lr, clipnorm=1.0),  # type: ignore
            loss={
                "output_probabilities": arrow_combined_loss,
                "aux_interval": losses.masked_mse_aux_interval,
            },
            loss_weights={
                "output_probabilities": 1.0,
                "aux_interval": run_config.aux_interval_weight,
            },
            metrics={
                "output_probabilities": _main_metrics,
                "aux_interval": keras.metrics.MeanSquaredError(name="aux_interval_mse"),
            },
        )
    else:
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=initial_lr, clipnorm=1.0),  # type: ignore
            loss=arrow_combined_loss,
            metrics=_main_metrics,
        )

    lr_callbacks: list[keras.callbacks.Callback] = []
    if use_lr_schedule:
        lr_callbacks.append(
            _build_cosine_warmup_schedule(
                total_epochs=run_config.epoch,
                warmup_epochs=run_config.warmup_epochs,
                lr_peak=run_config.lr_peak,
                lr_min=run_config.lr_min,
            )
        )

    # When aux_interval is enabled, the main loss metric is logged under the
    # "output_probabilities" head, so its validation metric name is
    # "val_output_probabilities_main_loss" instead of "val_main_loss".
    if use_aux_interval:
        monitor_metric = "val_output_probabilities_main_loss"
    else:
        monitor_metric = "val_main_loss"

    training_callbacks = _build_experiment_callbacks(
        run_config=run_config,
        experiment_name=experiment_name,
        monitor_metric=monitor_metric,
        monitor_mode="min",
        experiment_config=experiment_config,
    )

    train_history = _fit_and_save_model(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        run_config=run_config,
        callbacks=lr_callbacks + training_callbacks,
        monitor_metric=monitor_metric,
    )

    return model, train_history


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
) -> tuple[keras.Model, keras.callbacks.History]:
    """Train an arrow classification model.

    Trains a Keras model to classify arrow types (directions) based on audio
    features. Step times are always normalized.
    Uses SparseCategoricalCrossentropy loss and ignores the background class
    (0).

    Args:
        data_dir: Path to the directory containing training data.
        val_data_dir: Path to the directory containing validation data.
        batch_size: Number of samples per batch during training.
        take_count: Number of batches to use from the training dataset. -1 (default) uses the entire dataset (tf.data accepts -1 for take-all).
        epoch: Number of epochs to train for.
        model_output_dir: Directory where the trained model will be saved.
        callback_root_dir: Root directory for storing training callbacks.
        model_name: Name of the model that will be saved. If none provided,
            generated from the experiment name.
        val_take_count: Number of batches to use from the validation dataset.
            -1 (default) uses the entire validation dataset.
        model_params: Optional dictionary of parameters to pass to the arrow
            model builder. If omitted or None, default ArrowModelConfig values
            are used.

    Returns:
        A tuple containing:
            - model: The trained Keras model.
            - train_history: The training history object containing loss and
            metrics per epoch.
    """
    # Convert kwargs to config objects for backward compatibility
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
