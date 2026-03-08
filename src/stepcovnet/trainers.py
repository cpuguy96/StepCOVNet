"""Launches the main training loop to train a new StepCovNet model."""

import datetime
import logging
import math
import os

import keras
import tensorflow as tf

from stepcovnet import config, datasets, losses, metrics, models


def _get_tb_callback(root_dir: str, callback_name: str):
    """Create a TensorBoard callback for logging training metrics.

    Args:
        root_dir: Root directory for storing logs.
        callback_name: Name of the callback/run directory.

    Returns:
        TensorBoard callback configured to log to the specified directory.
    """
    logdir = os.path.join(root_dir, "logs", callback_name)
    return keras.callbacks.TensorBoard(
        logdir, histogram_freq=0, write_images=False, embeddings_freq=0
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
    ckpt_path = os.path.join(
        root_dir,
        "models",
        callback_name,
        f"{monitor_metric.upper()}" + "-{" + f"{monitor_metric}" + ":.5f}.keras",
    )
    model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath=ckpt_path,
        monitor=monitor_metric,
        save_best_only=True,
        mode=mode,
    )
    return model_checkpoint_callback


def _get_callbacks(
    root_dir: str, monitor_metric: str, monitor_mode: str, experiment_name: str = ""
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

    Returns:
        Tuple containing:
            - List of Keras callbacks (TensorBoard and ModelCheckpoint).
            - Callback name string (timestamp + optional experiment name).
    """
    now = datetime.datetime.now()
    callback_name = now.strftime("%Y%m%d-%H%M%S")
    if experiment_name:
        callback_name = callback_name + "-" + experiment_name
    return [
        _get_tb_callback(root_dir, callback_name),
        _get_ckpt_callback(root_dir, callback_name, monitor_metric, monitor_mode),
    ], callback_name


def _get_onset_experiment_name(
    take_count: int,
    apply_temporal_augment: bool,
    should_apply_spec_augment: bool,
    use_gaussian_target: bool,
    gaussian_sigma: float,
    model_params: config.OnsetModelConfig,
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

    initial_filters = model_params.initial_filters
    depth = model_params.depth
    kernel_size = model_params.kernel_size
    dropout_rate = model_params.dropout_rate
    dilation_rates = model_params.dilation_rates

    parts.append(f"unet_filters_{initial_filters}")
    parts.append(f"unet_depth_{depth}")
    parts.append(f"unet_kernel_size_{kernel_size}")
    parts.append(f"unet_dropout_{str(dropout_rate).replace('.', '_')}")

    # Make dilation rates robust to missing or non-iterable values.
    # When using a dict without 'dilation_rates', we want the literal
    # 'N_A' instead of joining over characters of the default string.
    if dilation_rates is None:
        dilation_str = "N_A"
    elif isinstance(dilation_rates, list | tuple):
        dilation_str = "_".join(map(str, dilation_rates))
    else:
        # Fall back to simple string conversion for any other type.
        dilation_str = str(dilation_rates)

    parts.append(f"unet_dilations_{dilation_str}")

    return "-".join(parts)


def _get_arrow_experiment_name(
    model_config: config.ArrowModelConfig,
    run_config: config.ArrowRunConfig,
) -> str:
    """Generate a descriptive experiment name from hyperparameters.

    Creates a human-readable name that encodes key training and model
    configuration parameters for arrow classification experiments.
    All distinguishing model and run fields are provided by the configs'
    get_experiment_name_parts() so that different configs yield different names.

    Args:
        model_config: Model configuration (architecture and input options).
        run_config: Run configuration (take, aux weights, loss options).

    Returns:
        String experiment name, e.g. "ARROW-transformer-take_all-att_layers_1-...".
    """
    parts = ["ARROW", model_config.model_type]
    parts.extend(model_config.get_experiment_name_parts())
    parts.extend(run_config.get_experiment_name_parts())
    return "-".join(parts)


def _write_model(model: keras.Model, model_output_dir: str):
    """Saves the trained Keras model to the specified directory.

    Args:
        model: The trained Keras model instance.
        model_output_dir: Directory path where the model file will be saved.
    """
    filepath = os.path.join(model_output_dir, model.name + ".keras")
    logging.info(f"Saving trained model to {filepath}")
    os.makedirs(model_output_dir, exist_ok=True)
    model.save(filepath=filepath)


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
    logdir = os.path.join(callback_root_dir, "logs", callback_name)
    os.makedirs(logdir, exist_ok=True)
    config_path = os.path.join(logdir, "config.json")
    experiment_config.to_json(config_path)
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
    )
    _save_config(experiment_config, run_config.callback_root_dir, callback_name)
    return training_callbacks


def _fit_and_save_model(
    model: keras.Model,
    train_dataset,
    val_dataset,
    run_config: config.RunConfig | config.ArrowRunConfig,
    callbacks: list[keras.callbacks.Callback],
) -> keras.callbacks.History:
    """Run model.fit with common settings, then persist the trained model.

    Args:
        model: Compiled Keras model to train.
        train_dataset: Training dataset.
        val_dataset: Validation dataset.
        run_config: Training run configuration (onset or arrow).
        callbacks: List of callbacks to pass to model.fit.

    Returns:
        Training history object from model.fit.
    """
    if run_config.seed is not None:
        tf.random.set_seed(run_config.seed)

    val_data = val_dataset.take(run_config.val_take_count)
    train_history = model.fit(
        train_dataset.take(run_config.take_count),
        epochs=run_config.epoch,
        validation_data=val_data,
        callbacks=callbacks,
        verbose=run_config.fit_verbose,  # type: ignore[arg-type]
    )

    _write_model(model, run_config.model_output_dir)

    return train_history


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
    train_dataset = datasets.create_dataset(
        data_dir=dataset_config.data_dir,
        batch_size=dataset_config.batch_size,
        apply_temporal_augment=dataset_config.apply_temporal_augment,
        should_apply_spec_augment=dataset_config.should_apply_spec_augment,
        use_gaussian_target=dataset_config.use_gaussian_target,
        gaussian_sigma=dataset_config.gaussian_sigma,
    )

    val_dataset = datasets.create_dataset(
        data_dir=dataset_config.val_data_dir,
        batch_size=dataset_config.batch_size,
        apply_temporal_augment=False,
        should_apply_spec_augment=False,
        use_gaussian_target=False,
    )

    experiment_name = _get_onset_experiment_name(
        take_count=run_config.take_count,
        apply_temporal_augment=dataset_config.apply_temporal_augment,
        should_apply_spec_augment=dataset_config.should_apply_spec_augment,
        use_gaussian_target=dataset_config.use_gaussian_target,
        gaussian_sigma=dataset_config.gaussian_sigma,
        model_params=model_config,
    )

    model = models.build_unet_wavenet_model(
        model_name=run_config.model_name or experiment_name,
        initial_filters=model_config.initial_filters,
        depth=model_config.depth,
        dilation_rates=model_config.dilation_rates,
        kernel_size=model_config.kernel_size,
        dropout_rate=model_config.dropout_rate,
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
        metrics=[
            keras.metrics.BinaryAccuracy(name="acc"),
            keras.metrics.Precision(name="prec"),
            keras.metrics.Recall(name="rec"),
            keras.metrics.AUC(curve="PR", name="pr_auc"),
            keras.metrics.AUC(name="auc"),
            metrics.OnsetF1Metric(tolerance=2, threshold=0.5),
        ],
    )

    experiment_config = config.OnsetExperimentConfig(
        dataset=dataset_config, model=model_config, run=run_config
    )
    training_callbacks = _build_experiment_callbacks(
        run_config=run_config,
        experiment_name=experiment_name,
        monitor_metric="val_pr_auc",
        monitor_mode="max",
        experiment_config=experiment_config,
    )

    train_history = _fit_and_save_model(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        run_config=run_config,
        callbacks=training_callbacks,
    )

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

    experiment_name = _get_arrow_experiment_name(
        model_config=model_config,
        run_config=run_config,
    )

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
