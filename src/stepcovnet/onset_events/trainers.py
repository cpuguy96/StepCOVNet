"""Training loop for event-based onset detection models."""

from __future__ import annotations

import dataclasses
import datetime
import logging
import os

import keras
import numpy as np
import tensorflow as tf

from stepcovnet import reproducibility
from stepcovnet.onset_events import config
from stepcovnet.onset_events import datasets
from stepcovnet.onset_events import losses
from stepcovnet.onset_events import matching
from stepcovnet.onset_events import metrics
from stepcovnet.onset_events import models


def _get_tb_callback(root_dir: str, callback_name: str) -> keras.callbacks.TensorBoard:
    """Create a TensorBoard callback for logging training metrics."""
    logdir = os.path.join(root_dir, "logs", callback_name)
    return keras.callbacks.TensorBoard(
        logdir, histogram_freq=0, write_images=False, embeddings_freq=0
    )


class _BestBaseModelCheckpoint(keras.callbacks.Callback):
    """Save ``base_model`` when a monitored metric improves (wrapper-safe)."""

    def __init__(self, filepath: str, monitor: str, mode: str) -> None:
        super().__init__()
        self.filepath = filepath
        self.monitor = monitor
        self.mode = mode
        self._best_value: float | None = None

    def on_epoch_end(self, epoch, logs=None) -> None:
        """Persist the underlying detector when ``monitor`` improves."""
        _ = epoch
        logs = logs or {}
        value = logs.get(self.monitor)
        if value is None:
            return
        current = float(value)
        if self._best_value is None:
            improved = True
        elif self.mode == "max":
            improved = current > self._best_value
        else:
            improved = current < self._best_value
        if not improved:
            return
        self._best_value = current
        base_model = (
            self.model.base_model if hasattr(self.model, "base_model") else self.model
        )
        os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
        base_model.save(self.filepath)


def _get_ckpt_callback(
    root_dir: str,
    callback_name: str,
    monitor_metric: str,
    mode: str,
) -> _BestBaseModelCheckpoint:
    """Create a checkpoint callback that saves the best underlying detector."""
    ckpt_path = os.path.join(
        root_dir,
        "models",
        callback_name,
        "best.keras",
    )
    return _BestBaseModelCheckpoint(
        filepath=ckpt_path,
        monitor=monitor_metric,
        mode=mode,
    )


def _get_callbacks(
    root_dir: str,
    monitor_metric: str,
    monitor_mode: str,
    experiment_name: str = "",
) -> tuple[list[keras.callbacks.Callback], str]:
    """Build TensorBoard and ModelCheckpoint callbacks plus a run directory name."""
    now = datetime.datetime.now()
    callback_name = now.strftime("%Y%m%d-%H%M%S")
    if experiment_name:
        callback_name = callback_name + "-" + experiment_name
    return [
        _get_tb_callback(root_dir, callback_name),
        _get_ckpt_callback(root_dir, callback_name, monitor_metric, monitor_mode),
    ], callback_name


def _save_config(
    experiment_config: config.OnsetEventExperimentConfig,
    callback_root_dir: str,
    callback_name: str,
) -> None:
    """Save experiment config JSON beside TensorBoard logs for the run."""
    logdir = os.path.join(callback_root_dir, "logs", callback_name)
    os.makedirs(logdir, exist_ok=True)
    config_path = os.path.join(logdir, "config.json")
    experiment_config.to_json(config_path)
    logging.info("Saved experiment config to %s", config_path)


def _build_experiment_callbacks(
    run_config: config.OnsetEventRunConfig,
    experiment_name: str,
    monitor_metric: str,
    monitor_mode: str,
    experiment_config: config.OnsetEventExperimentConfig,
) -> list[keras.callbacks.Callback]:
    """Build callbacks and persist config when ``callback_root_dir`` is set."""
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


def _latest_best_checkpoint_path(callback_root_dir: str) -> str | None:
    """Return the most recently modified ``best.keras`` under a callback root."""
    models_dir = os.path.join(callback_root_dir, "models")
    if not os.path.isdir(models_dir):
        return None
    candidates: list[str] = []
    for root, _dirs, files in os.walk(models_dir):
        if "best.keras" in files:
            candidates.append(os.path.join(root, "best.keras"))
    if not candidates:
        return None
    candidates.sort(key=os.path.getmtime, reverse=True)
    return candidates[0]


def _write_model(
    model: keras.Model,
    model_output_dir: str,
    *,
    callback_root_dir: str = "",
) -> None:
    """Save the trained Keras model to ``model_output_dir`` as ``.keras``.

    When ``callback_root_dir`` contains a ``best.keras`` checkpoint, that model is
    saved instead of the final-epoch weights.
    """
    base_model = model.base_model if hasattr(model, "base_model") else model
    filepath = os.path.join(model_output_dir, base_model.name + ".keras")
    os.makedirs(model_output_dir, exist_ok=True)
    best_path = _latest_best_checkpoint_path(callback_root_dir)
    if best_path is not None:
        logging.info("Saving best checkpoint from %s to %s", best_path, filepath)
        best_model = keras.models.load_model(best_path, compile=False)
        best_model.save(filepath=filepath)
        return
    logging.info("Saving trained model to %s", filepath)
    base_model.save(filepath=filepath)


def _get_onset_event_experiment_name(
    take_count: int,
    model_config: config.OnsetEventModelConfig,
) -> str:
    """Generate a descriptive experiment name from hyperparameters."""
    parts = ["ONSET_EVENT", f"frontend_{model_config.frontend}"]
    if take_count == -1:
        parts.append("take_all")
    else:
        parts.append(f"take_{take_count}")
    parts.append(f"queries_{model_config.num_queries}")
    parts.append(f"embed_{model_config.embed_dim}")
    return "-".join(parts)


def _event_onset_counts_numpy_wrapper(
    pred_times: np.ndarray,
    pred_confidence: np.ndarray,
    gt_times: np.ndarray,
    gt_mask: np.ndarray,
    tolerance_sec: np.ndarray,
    confidence_threshold: np.ndarray,
    min_onset_distance_ms: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Numpy wrapper for ``tf.numpy_function`` batch error counts."""
    tp, fp, fn = metrics.count_event_onset_errors_numpy(
        pred_times,
        pred_confidence,
        gt_times,
        gt_mask,
        tolerance_sec=float(tolerance_sec.reshape(-1)[0]),
        confidence_threshold=float(confidence_threshold.reshape(-1)[0]),
        min_onset_distance_ms=float(min_onset_distance_ms.reshape(-1)[0]),
    )
    return (
        np.array(tp, dtype=np.float64),
        np.array(fp, dtype=np.float64),
        np.array(fn, dtype=np.float64),
    )


@keras.saving.register_keras_serializable(package="stepcovnet.onset_events")
class EventOnsetF1Metric(keras.metrics.Metric):
    """Keras metric wrapping event-based onset F1 at a time tolerance.

    Uses :func:`metrics.count_event_onset_errors_numpy`. When
    ``min_onset_distance_ms`` is zero, Hungarian matching runs on all query
    slots; otherwise predictions are filtered like inference before matching.

    Attributes:
        tolerance_sec: Maximum absolute time error for a valid match.
        confidence_threshold: Minimum confidence for a prediction to count.
        min_onset_distance_ms: Minimum gap between kept predictions before
            matching; zero disables inference-style filtering.
        true_positives: Accumulated true positive count.
        false_positives: Accumulated false positive count.
        false_negatives: Accumulated false negative count.
    """

    def __init__(
        self,
        tolerance_sec: float = matching.DEFAULT_TOLERANCE_SEC,
        confidence_threshold: float = 0.5,
        min_onset_distance_ms: float = 0.0,
        name: str = "event_onset_f1",
        **kwargs,
    ) -> None:
        super().__init__(name=name, **kwargs)
        self.tolerance_sec = tolerance_sec
        self.confidence_threshold = confidence_threshold
        self.min_onset_distance_ms = min_onset_distance_ms
        self.true_positives = self.add_weight(name="tp", initializer="zeros")
        self.false_positives = self.add_weight(name="fp", initializer="zeros")
        self.false_negatives = self.add_weight(name="fn", initializer="zeros")

    def update_state(
        self,
        pred_times: tf.Tensor,
        pred_confidence: tf.Tensor,
        gt_times: tf.Tensor,
        gt_mask: tf.Tensor,
        sample_weight: tf.Tensor | None = None,
    ) -> None:
        """Accumulate TP/FP/FN counts for one batch."""
        _ = sample_weight
        pred_times = tf.cast(tf.convert_to_tensor(pred_times), tf.float32)
        pred_confidence = tf.cast(tf.convert_to_tensor(pred_confidence), tf.float32)
        gt_times = tf.cast(tf.convert_to_tensor(gt_times), tf.float32)
        gt_mask = tf.cast(tf.convert_to_tensor(gt_mask), tf.float32)

        tp, fp, fn = tf.numpy_function(
            _event_onset_counts_numpy_wrapper,
            [
                pred_times,
                pred_confidence,
                gt_times,
                gt_mask,
                np.array([self.tolerance_sec], dtype=np.float64),
                np.array([self.confidence_threshold], dtype=np.float64),
                np.array([self.min_onset_distance_ms], dtype=np.float64),
            ],
            [tf.float64, tf.float64, tf.float64],
        )
        self.true_positives.assign_add(tf.cast(tp, self.dtype))
        self.false_positives.assign_add(tf.cast(fp, self.dtype))
        self.false_negatives.assign_add(tf.cast(fn, self.dtype))

    def result(self) -> tf.Tensor:
        """Return micro-averaged F1 over accumulated batches."""
        tp = self.true_positives
        fp = self.false_positives
        fn = self.false_negatives
        precision = tp / (tp + fp + 1e-9)
        recall = tp / (tp + fn + 1e-9)
        return 2.0 * precision * recall / (precision + recall + 1e-9)

    def reset_state(self) -> None:
        """Reset accumulated counts."""
        self.true_positives.assign(0.0)
        self.false_positives.assign(0.0)
        self.false_negatives.assign(0.0)

    def get_config(self) -> dict:
        """Return metric configuration for serialization."""
        config_dict = super().get_config()
        config_dict.update(
            {
                "tolerance_sec": self.tolerance_sec,
                "confidence_threshold": self.confidence_threshold,
                "min_onset_distance_ms": self.min_onset_distance_ms,
            }
        )
        return config_dict


class OnsetEventTrainingModel(keras.Model):
    """Keras model wrapper that trains with Hungarian-match onset event loss.

    Attributes:
        base_model: Underlying onset event detector.
        tolerance_sec: Hungarian matching slack in seconds.
        lambda_cls: Classification loss weight.
        lambda_time: Matched time L1 loss weight.
        confidence_threshold: Threshold for validation F1 metrics.
        min_onset_distance_ms: Minimum gap for the min-gap validation F1 metric.
        loss_tracker: Running mean training/validation loss.
        event_f1_metric: Event F1 on all K query slots (no min-gap filter).
        event_f1_mingap_metric: Event F1 after inference-style filtering.
    """

    def __init__(
        self,
        base_model: keras.Model,
        *,
        tolerance_sec: float,
        lambda_cls: float,
        lambda_time: float,
        confidence_threshold: float,
        min_onset_distance_ms: float,
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.tolerance_sec = tolerance_sec
        self.lambda_cls = lambda_cls
        self.lambda_time = lambda_time
        self.confidence_threshold = confidence_threshold
        self.min_onset_distance_ms = min_onset_distance_ms
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.event_f1_metric = EventOnsetF1Metric(
            tolerance_sec=tolerance_sec,
            confidence_threshold=confidence_threshold,
            min_onset_distance_ms=0.0,
            name="event_onset_f1",
        )
        self.event_f1_mingap_metric = EventOnsetF1Metric(
            tolerance_sec=tolerance_sec,
            confidence_threshold=confidence_threshold,
            min_onset_distance_ms=min_onset_distance_ms,
            name="event_onset_f1_mingap",
        )

    def call(self, inputs, training: bool = False):
        """Forward pass through the wrapped detector."""
        return self.base_model(inputs, training=training)

    @property
    def metrics(self):
        """Metrics reported from custom train and test steps."""
        return [self.loss_tracker, self.event_f1_metric, self.event_f1_mingap_metric]

    @staticmethod
    def _unpack_batch(data) -> dict[str, tf.Tensor]:
        """Extract the sample dict from Keras ``fit`` batch data."""
        if isinstance(data, tuple):
            batch = data[0]
            if isinstance(batch, tuple):
                batch = batch[0]
        else:
            batch = data
        return batch

    def _forward_and_loss(
        self, batch: dict[str, tf.Tensor], *, training: bool
    ) -> tuple[tf.Tensor, dict[str, tf.Tensor]]:
        """Run the detector and compute the combined onset event loss."""
        model_inputs = _model_inputs_from_batch(batch, self.base_model)
        outputs = self.base_model(model_inputs, training=training)
        loss = losses.compute_onset_event_loss(
            outputs["pred_times"],
            outputs["pred_confidence"],
            batch["gt_times"],
            batch["gt_mask"],
            batch["duration"],
            tolerance_sec=self.tolerance_sec,
            lambda_cls=self.lambda_cls,
            lambda_time=self.lambda_time,
        )
        return loss, outputs

    def train_step(self, data):
        """Custom training step using ``compute_onset_event_loss``."""
        batch = self._unpack_batch(data)
        with tf.GradientTape() as tape:
            loss, outputs = self._forward_and_loss(batch, training=True)
        trainable_vars = self.base_model.trainable_variables
        grads = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(grads, trainable_vars))
        self.loss_tracker.update_state(loss)
        self.event_f1_metric.update_state(
            outputs["pred_times"],
            outputs["pred_confidence"],
            batch["gt_times"],
            batch["gt_mask"],
        )
        self.event_f1_mingap_metric.update_state(
            outputs["pred_times"],
            outputs["pred_confidence"],
            batch["gt_times"],
            batch["gt_mask"],
        )
        return {
            "loss": self.loss_tracker.result(),
            "event_onset_f1": self.event_f1_metric.result(),
            "event_onset_f1_mingap": self.event_f1_mingap_metric.result(),
        }

    def test_step(self, data):
        """Custom validation step using ``compute_onset_event_loss``."""
        batch = self._unpack_batch(data)
        loss, outputs = self._forward_and_loss(batch, training=False)
        self.loss_tracker.update_state(loss)
        self.event_f1_metric.update_state(
            outputs["pred_times"],
            outputs["pred_confidence"],
            batch["gt_times"],
            batch["gt_mask"],
        )
        self.event_f1_mingap_metric.update_state(
            outputs["pred_times"],
            outputs["pred_confidence"],
            batch["gt_times"],
            batch["gt_mask"],
        )
        return {
            "loss": self.loss_tracker.result(),
            "event_onset_f1": self.event_f1_metric.result(),
            "event_onset_f1_mingap": self.event_f1_mingap_metric.result(),
        }


def _sync_model_config_with_dataset(
    model_config: config.OnsetEventModelConfig,
    dataset_config: config.OnsetEventDatasetConfig,
) -> config.OnsetEventModelConfig:
    """Align model input sizing fields with dataset duration and sample rate."""
    return dataclasses.replace(
        model_config,
        target_sample_rate=dataset_config.target_sample_rate,
        max_audio_seconds=dataset_config.max_audio_seconds,
    )


def _model_inputs_from_batch(
    batch: dict[str, tf.Tensor],
    base_model: keras.Model,
) -> dict[str, tf.Tensor]:
    """Map a dataset batch to Keras model inputs."""
    input_names = {tensor.name.split(":")[0] for tensor in base_model.inputs}
    if "features" in input_names:
        return {
            "features": batch["features"],
            "duration": batch["duration"],
        }
    return {
        "audio": batch["audio"],
        "duration": batch["duration"],
    }


def resolve_overfit_query_options(
    run_config: config.OnsetEventRunConfig,
) -> tuple[bool, bool]:
    """Return ``(init_query_refs_from_gt, learn_time_delta)`` for model construction."""
    if run_config.pipeline_check_shortcuts:
        return True, False
    return run_config.init_query_refs_from_gt, run_config.learn_time_delta


def _query_ref_normalized_from_batch(
    gt_times: np.ndarray,
    gt_mask: np.ndarray,
    duration: float,
    num_queries: int,
) -> tuple[float, ...]:
    """Build per-query normalized reference times from sorted ground-truth onsets.

    Query slot ``i`` references the ``i``-th valid onset sorted by time. Remaining
    slots use a uniform grid over the normalized time after the last matched onset.
    """
    if duration <= 0.0:
        raise ValueError("duration must be positive")

    valid = gt_mask.astype(bool)
    gt_values = gt_times[valid]
    if gt_values.size:
        order = np.argsort(gt_values)
        gt_sorted = gt_values[order]
    else:
        gt_sorted = np.asarray([], dtype=np.float64)

    ref = np.empty(num_queries, dtype=np.float32)
    num_gt = int(gt_sorted.size)
    num_from_gt = min(num_queries, num_gt)
    if num_from_gt:
        ref[:num_from_gt] = gt_sorted[:num_from_gt] / duration

    if num_queries > num_from_gt:
        tail_count = num_queries - num_from_gt
        start_norm = float(ref[num_from_gt - 1]) if num_from_gt else 0.0
        tail = start_norm + (1.0 - start_norm) * (
            (np.arange(tail_count, dtype=np.float32) + 0.5) / float(tail_count)
        )
        ref[num_from_gt:] = tail

    ref = np.clip(ref, 1e-4, 1.0 - 1e-4)
    return tuple(float(value) for value in ref)


def _resolve_single_song_pair(
    dataset_config: config.OnsetEventDatasetConfig,
    run_config: config.OnsetEventRunConfig,
) -> tuple[str, str] | None:
    """Return an explicit single pair for overfit training, or ``None`` for full data."""
    audio_path = dataset_config.overfit_audio_path.strip()
    chart_path = dataset_config.overfit_chart_path.strip()
    if audio_path or chart_path:
        if not audio_path or not chart_path:
            raise ValueError(
                "overfit_audio_path and overfit_chart_path must both be set for "
                "single-song overfit"
            )
        if not os.path.isfile(audio_path):
            raise ValueError(f"overfit audio file not found: {audio_path}")
        if not os.path.isfile(chart_path):
            raise ValueError(f"overfit chart file not found: {chart_path}")
        return audio_path, chart_path

    if run_config.overfit_one_song:
        search_dirs = []
        test_dir = dataset_config.test_data_dir.strip()
        if test_dir:
            search_dirs.append(test_dir)
        data_dir = dataset_config.data_dir.strip()
        if data_dir and data_dir not in search_dirs:
            search_dirs.append(data_dir)
        for root in search_dirs:
            try:
                return datasets.first_valid_pair(
                    root,
                    max_steps_per_chart=dataset_config.max_steps_per_chart,
                )
            except ValueError:
                continue
        raise ValueError(
            "overfit_one_song: no valid audio-chart pairs under "
            f"test_data_dir={test_dir!r} or data_dir={data_dir!r}"
        )

    return None


def _create_datasets(
    experiment_config: config.OnsetEventExperimentConfig,
) -> tuple[tf.data.Dataset, tf.data.Dataset]:
    """Build training and validation datasets from experiment config."""
    dataset_config = experiment_config.dataset
    run_config = experiment_config.run
    common_kwargs = {
        "batch_size": dataset_config.batch_size,
        "max_audio_seconds": dataset_config.max_audio_seconds,
        "n_max_onsets": dataset_config.n_max_onsets,
        "max_steps_per_chart": dataset_config.max_steps_per_chart,
        "target_sample_rate": dataset_config.target_sample_rate,
        "frontend": experiment_config.model.frontend,
        "mert_features_dir": dataset_config.mert_features_dir,
        "data_root": dataset_config.data_root or dataset_config.data_dir,
    }
    single_pair = _resolve_single_song_pair(dataset_config, run_config)
    if single_pair is not None:
        logging.info(
            "Single-song overfit mode: %s + %s",
            single_pair[0],
            single_pair[1],
        )
        overfit_dataset = datasets.create_onset_event_dataset_from_pairs(
            [single_pair],
            shuffle=False,
            **common_kwargs,
        )
        return overfit_dataset, overfit_dataset

    train_dataset = datasets.create_onset_event_dataset(
        dataset_config.data_dir,
        shuffle=True,
        seed=run_config.seed,
        **common_kwargs,
    )
    val_dataset = datasets.create_onset_event_dataset(
        dataset_config.val_data_dir,
        shuffle=False,
        **common_kwargs,
    )
    return train_dataset, val_dataset


def _fit_and_save_model(
    model: keras.Model,
    train_dataset: tf.data.Dataset,
    val_dataset: tf.data.Dataset,
    *,
    epochs: int,
    take_count: int,
    val_take_count: int,
    seed: int,
    model_output_dir: str,
    callback_root_dir: str,
    callbacks: list[keras.callbacks.Callback],
) -> keras.callbacks.History:
    """Run ``model.fit`` and persist the trained base model."""
    train_data = train_dataset
    if take_count != -1:
        train_data = train_dataset.take(take_count)
    val_data = val_dataset
    if val_take_count != -1:
        val_data = val_dataset.take(val_take_count)

    train_history = model.fit(
        train_data,
        epochs=epochs,
        validation_data=val_data,
        callbacks=callbacks,
        verbose=1,
    )
    _write_model(model, model_output_dir, callback_root_dir=callback_root_dir)
    return train_history


def train_onset_event(
    experiment_config: config.OnsetEventExperimentConfig,
    *,
    take_count: int = -1,
    val_take_count: int = -1,
) -> tuple[keras.Model, keras.callbacks.History]:
    """Train an event-based onset model from an experiment configuration.

    Args:
        experiment_config: Dataset, model, and run settings.
        take_count: Maximum training batches per epoch (-1 for all).
        val_take_count: Maximum validation batches per epoch (-1 for all).

    Returns:
        Tuple of the trained base Keras model and training history.
    """
    dataset_config = experiment_config.dataset
    run_config = experiment_config.run
    model_config = _sync_model_config_with_dataset(
        experiment_config.model,
        dataset_config,
    )

    if not dataset_config.data_dir or not dataset_config.val_data_dir:
        raise ValueError("dataset.data_dir and dataset.val_data_dir are required")
    if not run_config.model_output_dir:
        raise ValueError("run.model_output_dir is required")

    reproducibility.apply_training_seed(run_config.seed)

    train_dataset, val_dataset = _create_datasets(experiment_config)

    init_gt_refs, learn_time_delta = resolve_overfit_query_options(run_config)
    query_ref_normalized: tuple[float, ...] | None = None
    if init_gt_refs:
        peek_batch = next(iter(train_dataset.take(1)))
        query_ref_normalized = _query_ref_normalized_from_batch(
            peek_batch["gt_times"].numpy()[0],
            peek_batch["gt_mask"].numpy()[0],
            float(peek_batch["duration"].numpy()[0]),
            model_config.num_queries,
        )

    base_model = models.build_onset_event_model(
        model_config,
        query_ref_normalized=query_ref_normalized,
        learn_time_delta=learn_time_delta,
    )
    model = OnsetEventTrainingModel(
        base_model,
        tolerance_sec=run_config.tolerance_sec,
        lambda_cls=run_config.lambda_cls,
        lambda_time=run_config.lambda_time,
        confidence_threshold=run_config.confidence_threshold,
        min_onset_distance_ms=run_config.min_onset_distance_ms,
    )
    # Hungarian matching uses tf.ensure_shape with dynamic batch dims; graph mode fails.
    epochs = run_config.epochs
    if run_config.overfit_one_song:
        epochs = min(epochs, 300)
        learning_rate = 5e-3
    else:
        learning_rate = 2e-3
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=5.0),  # type: ignore[arg-type]
        run_eagerly=True,
    )

    experiment_name = _get_onset_event_experiment_name(
        take_count=take_count,
        model_config=model_config,
    )
    training_callbacks = _build_experiment_callbacks(
        run_config=run_config,
        experiment_name=experiment_name,
        monitor_metric="val_event_onset_f1",
        monitor_mode="max",
        experiment_config=experiment_config,
    )

    train_history = _fit_and_save_model(
        model,
        train_dataset,
        val_dataset,
        epochs=epochs,
        take_count=take_count,
        val_take_count=val_take_count,
        seed=run_config.seed,
        model_output_dir=run_config.model_output_dir,
        callback_root_dir=run_config.callback_root_dir,
        callbacks=training_callbacks,
    )

    return base_model, train_history
