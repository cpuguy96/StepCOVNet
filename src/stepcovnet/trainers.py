"""Launches the main training loop to train a new StepCovNet model."""

import datetime
import logging
import os

import keras

from stepcovnet import datasets
from stepcovnet import metrics
from stepcovnet import models


def _get_tb_callback(root_dir: str, callback_name: str):
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
) -> list[keras.callbacks.Callback]:
    now = datetime.datetime.now()
    callback_name = now.strftime("%Y%m%d-%H%M%S")
    if experiment_name:
        callback_name = callback_name + "-" + experiment_name
    return [
        _get_tb_callback(root_dir, callback_name),
        _get_ckpt_callback(root_dir, callback_name, monitor_metric, monitor_mode),
    ]


def _get_onset_experiment_name(
    take_count: int,
    apply_temporal_augment: bool,
    should_apply_spec_augment: bool,
    use_gaussian_target: bool,
    gaussian_sigma: float,
    model_params: dict,
) -> str:
    """
    Generates a descriptive experiment name from hyperparameters.
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

    parts.append(f"unet_filters_{model_params.get('initial_filters', 'N_A')}")
    parts.append(f"unet_depth_{model_params.get('depth', 'N_A')}")
    parts.append(f"unet_kernel_size_{model_params.get('kernel_size', 'N_A')}")
    parts.append(
        f"unet_dropout_{str(model_params.get('dropout_rate', 'N_A')).replace('.', '_')}"
    )
    parts.append(
        f"unet_dilations_{'_'.join(map(str, model_params.get('dilation_rates', 'N_A')))}"
    )

    return "-".join(parts)


def _get_arrow_experiment_name(take_count: int, model_params: dict):
    parts = ["ARROW"]

    if take_count > 0:
        parts.append(f"take_{take_count}")

    parts.append(f"att_layers_{model_params.get('num_layers', 'N_A')}")

    return "-".join(parts)


def _write_model(model: keras.Model, model_output_dir: str):
    """Saves the trained Keras model to the specified directory.

    Args:
        model: The trained Keras model instance.
        model_output_dir: Directory path where the model file will be saved.
    """
    filepath = os.path.join(model_output_dir, model.name + '.keras')
    logging.info(f"Saving trained model to {filepath}")
    os.makedirs(model_output_dir, exist_ok=True)
    model.save(filepath=filepath)


def run_train(
        *,
        data_dir: str,
        val_data_dir: str,
        batch_size: int,
        normalize: bool,
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

    Args:
        data_dir: Path to the directory containing training data.
        val_data_dir: Path to the directory containing validation data.
        batch_size: Number of samples per batch during training.
        normalize: Whether to normalize the input data.
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
    train_dataset = datasets.create_dataset(
        data_dir=data_dir,
        batch_size=batch_size,
        normalize=normalize,
        apply_temporal_augment=apply_temporal_augment,
        should_apply_spec_augment=should_apply_spec_augment,
        use_gaussian_target=use_gaussian_target,
        gaussian_sigma=gaussian_sigma,
    )

    val_dataset = datasets.create_dataset(
        data_dir=val_data_dir,
        batch_size=batch_size,
        normalize=normalize,
        apply_temporal_augment=False,
        should_apply_spec_augment=False,
        use_gaussian_target=False,
    )

    experiment_name = _get_onset_experiment_name(
        take_count=take_count,
        apply_temporal_augment=apply_temporal_augment,
        should_apply_spec_augment=should_apply_spec_augment,
        use_gaussian_target=use_gaussian_target,
        gaussian_sigma=gaussian_sigma,
        model_params=model_params,
    )

    model = models.build_unet_wavenet_model(
        model_name=model_name or experiment_name, **model_params
    )

    model.summary()

    if use_gaussian_target:
        loss = keras.losses.MeanSquaredError()
    else:
        loss = keras.losses.BinaryFocalCrossentropy(apply_class_balancing=True)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
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

    if callback_root_dir:
        training_callbacks = _get_callbacks(
            root_dir=callback_root_dir,
            monitor_metric="val_pr_auc",
            monitor_mode="max",
            experiment_name=experiment_name,
        )
    else:
        training_callbacks = []

    train_history = model.fit(
        train_dataset.take(take_count),
        epochs=epoch,
        validation_data=val_dataset,
        callbacks=training_callbacks,
    )

    _write_model(model, model_output_dir)

    return model, train_history


def run_arrow_train(
        *,
        data_dir: str,
        val_data_dir: str,
        batch_size: int,
        normalize: bool,
        model_params: dict,
        take_count: int,
        epoch: int,
        model_output_dir: str,
        callback_root_dir: str = "",
        model_name: str = "",
) -> tuple[keras.Model, keras.callbacks.History]:
    """Train an arrow classification model.

    Trains a Keras model to classify arrow types (directions) based on audio
    features.
    Uses SparseCategoricalCrossentropy loss and ignores the background class
    (0).

    Args:
        data_dir: Path to the directory containing training data.
        val_data_dir: Path to the directory containing validation data.
        batch_size: Number of samples per batch during training.
        normalize: Whether to normalize the input data.
        model_params: Dictionary of parameters to pass to the arrow model
            builder.
        take_count: Number of batches to use from the training dataset.
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

    train_dataset = datasets.create_arrow_dataset(
        data_dir=data_dir,
        batch_size=batch_size,
        normalize=normalize,
    )

    val_dataset = datasets.create_arrow_dataset(
        data_dir=val_data_dir,
        batch_size=batch_size,
        normalize=normalize,
    )

    experiment_name = _get_arrow_experiment_name(
        take_count=take_count, model_params=model_params
    )

    model = models.build_arrow_model(model_name=model_name or experiment_name,
                                     **model_params)

    model.summary()

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss=keras.losses.SparseCategoricalCrossentropy(ignore_class=0),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")],
    )

    if callback_root_dir:
        training_callbacks = _get_callbacks(
            root_dir=callback_root_dir,
            monitor_metric="val_loss",
            monitor_mode="min",
            experiment_name=experiment_name,
        )
    else:
        training_callbacks = []

    train_history = model.fit(
        train_dataset.take(take_count),
        epochs=epoch,
        validation_data=val_dataset,
        callbacks=training_callbacks,
    )

    _write_model(model, model_output_dir)

    return model, train_history
