"""Launches the main training loop to train a new StepCovNet model."""

import datetime
import logging
import os

import keras
import tensorflow as tf

from stepcovnet import config
from stepcovnet import datasets
from stepcovnet import metrics
from stepcovnet import models


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
    model_params: dict | config.OnsetModelConfig,
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
        model_params: Model configuration dict or OnsetModelConfig object
            containing architecture parameters.

    Returns:
        String experiment name with format:
        "ONSET-take_{N}-sigma_{X}-temporal_augment-spec_augment-unet_filters_{N}-..."
    """
    parts = ["ONSET"]

    if take_count > 0:
        parts.append(f"take_{take_count}")

    if use_gaussian_target:
        sigma_str = str(gaussian_sigma).replace(".", "_")
        parts.append(f"sigma_{sigma_str}")

    if apply_temporal_augment:
        parts.append("temporal_augment")

    if should_apply_spec_augment:
        parts.append("spec_augment")

    # Handle both dict and config object
    if isinstance(model_params, config.OnsetModelConfig):
        initial_filters = model_params.initial_filters
        depth = model_params.depth
        kernel_size = model_params.kernel_size
        dropout_rate = model_params.dropout_rate
        dilation_rates = model_params.dilation_rates
    else:
        initial_filters = model_params.get("initial_filters", "N_A")
        depth = model_params.get("depth", "N_A")
        kernel_size = model_params.get("kernel_size", "N_A")
        dropout_rate = model_params.get("dropout_rate", "N_A")
        dilation_rates = model_params.get("dilation_rates")

    parts.append(f"unet_filters_{initial_filters}")
    parts.append(f"unet_depth_{depth}")
    parts.append(f"unet_kernel_size_{kernel_size}")
    parts.append(f"unet_dropout_{str(dropout_rate).replace('.', '_')}")

    # Make dilation rates robust to missing or non-iterable values.
    # When using a dict without 'dilation_rates', we want the literal
    # 'N_A' instead of joining over characters of the default string.
    if dilation_rates is None:
        dilation_str = "N_A"
    elif isinstance(dilation_rates, (list, tuple)):
        dilation_str = "_".join(map(str, dilation_rates))
    else:
        # Fall back to simple string conversion for any other type.
        dilation_str = str(dilation_rates)

    parts.append(f"unet_dilations_{dilation_str}")

    return "-".join(parts)


def _get_arrow_experiment_name(
    take_count: int, model_params: dict | config.ArrowModelConfig
) -> str:
    """Generate a descriptive experiment name from hyperparameters.

    Creates a human-readable name that encodes key training and model
    configuration parameters for arrow classification experiments.

    Args:
        take_count: Number of batches used from training dataset.
        model_params: Model configuration dict or ArrowModelConfig object
            containing architecture parameters.

    Returns:
        String experiment name with format: "ARROW-take_{N}-att_layers_{N}".
    """
    parts = ["ARROW"]

    if take_count > 0:
        parts.append(f"take_{take_count}")

    # Handle both dict and config object
    if isinstance(model_params, config.ArrowModelConfig):
        num_layers = model_params.num_layers
    else:
        num_layers = model_params.get("num_layers", "N_A")

    parts.append(f"att_layers_{num_layers}")

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

    if run_config.callback_root_dir:
        training_callbacks, callback_name = _get_callbacks(
            root_dir=run_config.callback_root_dir,
            monitor_metric="val_pr_auc",
            monitor_mode="max",
            experiment_name=experiment_name,
        )

        # Save config
        experiment_config = config.OnsetExperimentConfig(
            dataset=dataset_config, model=model_config, run=run_config
        )
        _save_config(experiment_config, run_config.callback_root_dir, callback_name)
    else:
        training_callbacks = []

    if run_config.seed is not None:
        tf.random.set_seed(run_config.seed)

    train_history = model.fit(
        train_dataset.take(run_config.take_count),
        epochs=run_config.epoch,
        validation_data=val_dataset,
        callbacks=training_callbacks,
    )

    _write_model(model, run_config.model_output_dir)

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
    model_params: dict,
    take_count: int,
    epoch: int,
    model_output_dir: str,
    callback_root_dir: str = "",
    model_name: str = "",
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
        model_params: Dictionary of parameters to pass to the model builder.
        take_count: Number of batches to use from the training dataset.
        epoch: Number of epochs to train for.
        model_output_dir: Directory where the trained model will be saved.
        callback_root_dir: Root directory for storing training callbacks (
            checkpoints, logs).
        model_name: Name of the model that will be saved. If none provided,
            generated from the experiment name.

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
    model_config = config.OnsetModelConfig(**model_params)
    run_config = config.RunConfig(
        epoch=epoch,
        take_count=take_count,
        model_output_dir=model_output_dir,
        callback_root_dir=callback_root_dir,
        model_name=model_name,
    )
    return run_train_from_config(dataset_config, model_config, run_config)


def run_arrow_train_from_config(
    dataset_config: config.ArrowDatasetConfig,
    model_config: config.ArrowModelConfig,
    run_config: config.RunConfig,
) -> tuple[keras.Model, keras.callbacks.History]:
    """Train an arrow classification model using configuration objects.

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
    if run_config.epoch < 1:
        raise ValueError("epoch must be at least 1")

    if run_config.take_count != -1 and run_config.take_count < 1:
        raise ValueError("take_count must be -1 (entire dataset) or at least 1")

    train_dataset = datasets.create_arrow_dataset(
        data_dir=dataset_config.data_dir,
        batch_size=dataset_config.batch_size,
    )

    val_dataset = datasets.create_arrow_dataset(
        data_dir=dataset_config.val_data_dir,
        batch_size=dataset_config.batch_size,
    )

    experiment_name = _get_arrow_experiment_name(
        take_count=run_config.take_count, model_params=model_config
    )

    model = models.build_arrow_model(
        model_name=run_config.model_name or experiment_name,
        num_layers=model_config.num_layers,
        d_model=model_config.d_model,
        num_heads=model_config.num_heads,
        ff_dim=model_config.ff_dim,
        dropout_rate=model_config.dropout_rate,
    )

    model.summary()

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),  # type: ignore
        loss=keras.losses.SparseCategoricalCrossentropy(ignore_class=0),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")],
    )

    if run_config.callback_root_dir:
        training_callbacks, callback_name = _get_callbacks(
            root_dir=run_config.callback_root_dir,
            monitor_metric="val_loss",
            monitor_mode="min",
            experiment_name=experiment_name,
        )

        # Save config
        experiment_config = config.ArrowExperimentConfig(
            dataset=dataset_config, model=model_config, run=run_config
        )
        _save_config(experiment_config, run_config.callback_root_dir, callback_name)
    else:
        training_callbacks = []

    if run_config.seed is not None:
        tf.random.set_seed(run_config.seed)

    train_history = model.fit(
        train_dataset.take(run_config.take_count),
        epochs=run_config.epoch,
        validation_data=val_dataset,
        callbacks=training_callbacks,
    )

    _write_model(model, run_config.model_output_dir)

    return model, train_history


def run_arrow_train(
    *,
    data_dir: str,
    val_data_dir: str,
    batch_size: int,
    model_params: dict,
    epoch: int,
    model_output_dir: str,
    take_count: int = -1,
    callback_root_dir: str = "",
    model_name: str = "",
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
        model_params: Dictionary of parameters to pass to the arrow model
            builder.
        take_count: Number of batches to use from the training dataset. -1 (default) uses the entire dataset (tf.data accepts -1 for take-all).
        epoch: Number of epochs to train for.
        model_output_dir: Directory where the trained model will be saved.
        callback_root_dir: Root directory for storing training callbacks.
        model_name: Name of the model that will be saved. If none provided,
            generated from the experiment name.

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
    model_config = config.ArrowModelConfig(**model_params)
    run_config = config.RunConfig(
        epoch=epoch,
        take_count=take_count,
        model_output_dir=model_output_dir,
        callback_root_dir=callback_root_dir,
        model_name=model_name,
    )
    return run_arrow_train_from_config(dataset_config, model_config, run_config)
