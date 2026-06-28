"""Training loop for AR onset models (``gate-tide-overfit``)."""

from __future__ import annotations

import logging
import pathlib

import keras
import numpy as np
import tensorflow as tf

from stepcovnet import reproducibility
from stepcovnet.onset_ar import config, datasets, inference, losses, models
from stepcovnet.onset_events import matching
from stepcovnet.onset_events import trainers as event_trainers


def _ar_event_onset_counts_numpy(
    pred_times: np.ndarray,
    pred_mask: np.ndarray,
    gt_times: np.ndarray,
    gt_mask: np.ndarray,
    tolerance_sec: float,
) -> tuple[int, int, int]:
    """Count TP/FP/FN for teacher-fed AR decode on one batch item."""
    pred = np.asarray(pred_times, dtype=np.float64).reshape(-1)
    pred_mask_arr = np.asarray(pred_mask, dtype=np.float64).reshape(-1)
    gt = np.asarray(gt_times, dtype=np.float64).reshape(-1)
    gt_mask_arr = np.asarray(gt_mask, dtype=np.float64).reshape(-1)
    pred_kept = pred[pred_mask_arr > 0.5]
    gt_kept = gt[gt_mask_arr > 0.5]
    if pred_kept.size == 0 and gt_kept.size == 0:
        return 0, 0, 0
    if pred_kept.size == 0:
        return 0, 0, int(gt_kept.size)
    gt_mask_ones = np.ones((1, gt_kept.size), dtype=np.float32)
    result = matching.match_onsets_numpy(
        pred_kept.reshape(1, -1),
        gt_kept.reshape(1, -1),
        gt_mask_ones,
        tolerance_sec=tolerance_sec,
    )
    num_matches = int(result.num_matches[0])
    false_positives = int(pred_kept.size) - num_matches
    false_negatives = int(result.gt_unmatched_mask[0].sum())
    return num_matches, false_positives, false_negatives


def _ar_event_onset_counts_numpy_wrapper(
    pred_times: np.ndarray,
    pred_mask: np.ndarray,
    gt_times: np.ndarray,
    gt_mask: np.ndarray,
    tolerance_sec: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tp, fp, fn = _ar_event_onset_counts_numpy(
        pred_times,
        pred_mask,
        gt_times,
        gt_mask,
        tolerance_sec=float(tolerance_sec.reshape(-1)[0]),
    )
    return (
        np.array(tp, dtype=np.float64),
        np.array(fp, dtype=np.float64),
        np.array(fn, dtype=np.float64),
    )


@keras.saving.register_keras_serializable(package="stepcovnet.onset_ar")
class ArEventOnsetF1Metric(keras.metrics.Metric):
    """Micro event F1 for teacher-fed pointer+residual decode."""

    def __init__(
        self,
        tolerance_sec: float = matching.DEFAULT_TOLERANCE_SEC,
        name: str = "event_onset_f1",
        **kwargs,
    ) -> None:
        super().__init__(name=name, **kwargs)
        self.tolerance_sec = tolerance_sec
        self.true_positives = self.add_weight(name="tp", initializer="zeros")
        self.false_positives = self.add_weight(name="fp", initializer="zeros")
        self.false_negatives = self.add_weight(name="fn", initializer="zeros")

    def update_state(
        self,
        pred_times: tf.Tensor,
        pred_mask: tf.Tensor,
        gt_times: tf.Tensor,
        gt_mask: tf.Tensor,
        sample_weight: tf.Tensor | None = None,
    ) -> None:
        _ = sample_weight
        tp, fp, fn = tf.numpy_function(
            _ar_event_onset_counts_numpy_wrapper,
            [
                pred_times,
                pred_mask,
                gt_times,
                gt_mask,
                np.array([self.tolerance_sec], dtype=np.float64),
            ],
            [tf.float64, tf.float64, tf.float64],
        )
        tp = tf.reshape(tp, [])
        fp = tf.reshape(fp, [])
        fn = tf.reshape(fn, [])
        self.true_positives.assign_add(tf.cast(tp, self.dtype))
        self.false_positives.assign_add(tf.cast(fp, self.dtype))
        self.false_negatives.assign_add(tf.cast(fn, self.dtype))

    def result(self) -> tf.Tensor:
        tp = self.true_positives
        fp = self.false_positives
        fn = self.false_negatives
        precision = tp / (tp + fp + 1e-9)
        recall = tp / (tp + fn + 1e-9)
        return 2.0 * precision * recall / (precision + recall + 1e-9)

    def reset_state(self) -> None:
        self.true_positives.assign(0.0)
        self.false_positives.assign(0.0)
        self.false_negatives.assign(0.0)

    def get_config(self) -> dict:
        config_dict = super().get_config()
        config_dict.update({"tolerance_sec": self.tolerance_sec})
        return config_dict


class ArOnsetTrainingModel(keras.Model):
    """Custom train/eval loop for teacher-forced AR onset training."""

    def __init__(
        self,
        base_model: keras.Model,
        *,
        experiment_config: config.ArExperimentConfig,
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.experiment_config = experiment_config
        run_config = experiment_config.run
        model_config = experiment_config.model
        self.patch_frames = model_config.patch_frames
        self.hop_sec = experiment_config.dataset.hop_sec
        self.lambda_time = run_config.lambda_time
        self.length_normalize_ce = run_config.length_normalize_ce
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.token_loss_tracker = keras.metrics.Mean(name="token_loss")
        self.pointer_loss_tracker = keras.metrics.Mean(name="pointer_loss")
        self.time_loss_tracker = keras.metrics.Mean(name="time_loss")
        self.token_accuracy = keras.metrics.Mean(name="token_accuracy")
        self.event_f1_metric = ArEventOnsetF1Metric(
            tolerance_sec=run_config.tolerance_sec,
            name="event_onset_f1",
        )

    @property
    def metrics(self):
        return [
            self.loss_tracker,
            self.token_loss_tracker,
            self.pointer_loss_tracker,
            self.time_loss_tracker,
            self.token_accuracy,
            self.event_f1_metric,
        ]

    @staticmethod
    def _unpack_batch(data) -> dict[str, tf.Tensor]:
        if isinstance(data, tuple):
            batch = data[0]
            if isinstance(batch, tuple):
                batch = batch[0]
        else:
            batch = data
        return batch

    def _model_inputs(self, batch: dict[str, tf.Tensor]) -> dict[str, tf.Tensor]:
        return {
            "mert_patches": batch["mert_patches"],
            "patch_mask": batch["patch_mask"],
            "decoder_input_ids": batch["decoder_input_ids"],
            "decoder_mask": batch["decoder_mask"],
        }

    def _forward_and_loss(
        self,
        batch: dict[str, tf.Tensor],
        *,
        training: bool,
    ) -> tuple[tf.Tensor, dict[str, tf.Tensor], dict[str, tf.Tensor]]:
        outputs = self.base_model(self._model_inputs(batch), training=training)
        total_loss, parts = losses.compute_ar_onset_loss(
            outputs,
            batch,
            patch_frames=self.patch_frames,
            hop_sec=self.hop_sec,
            lambda_time=self.lambda_time,
            length_normalize_ce=self.length_normalize_ce,
        )
        return total_loss, parts, outputs

    def train_step(self, data):
        batch = self._unpack_batch(data)
        with tf.GradientTape() as tape:
            total_loss, parts, outputs = self._forward_and_loss(batch, training=True)
        trainable_vars = self.base_model.trainable_variables
        grads = tape.gradient(total_loss, trainable_vars)
        self.optimizer.apply_gradients(zip(grads, trainable_vars, strict=False))
        self.loss_tracker.update_state(total_loss)
        self.token_loss_tracker.update_state(parts["token_loss"])
        self.pointer_loss_tracker.update_state(parts["pointer_loss"])
        self.time_loss_tracker.update_state(parts["time_loss"])
        self.token_accuracy.update_state(
            losses.masked_token_accuracy(
                outputs["token_logits"],
                batch["decoder_target_ids"],
                batch["decoder_mask"],
            ),
        )
        pred_times, pred_mask = inference.decode_teacher_fed_times_tf(
            outputs,
            batch,
            patch_frames=self.patch_frames,
            hop_sec=self.hop_sec,
        )
        self.event_f1_metric.update_state(
            pred_times,
            pred_mask,
            batch["gt_times"],
            batch["gt_mask"],
        )
        return {metric.name: metric.result() for metric in self.metrics}

    def test_step(self, data):
        batch = self._unpack_batch(data)
        total_loss, parts, outputs = self._forward_and_loss(batch, training=False)
        self.loss_tracker.update_state(total_loss)
        self.token_loss_tracker.update_state(parts["token_loss"])
        self.pointer_loss_tracker.update_state(parts["pointer_loss"])
        self.time_loss_tracker.update_state(parts["time_loss"])
        self.token_accuracy.update_state(
            losses.masked_token_accuracy(
                outputs["token_logits"],
                batch["decoder_target_ids"],
                batch["decoder_mask"],
            ),
        )
        pred_times, pred_mask = inference.decode_teacher_fed_times_tf(
            outputs,
            batch,
            patch_frames=self.patch_frames,
            hop_sec=self.hop_sec,
        )
        self.event_f1_metric.update_state(
            pred_times,
            pred_mask,
            batch["gt_times"],
            batch["gt_mask"],
        )
        return {metric.name: metric.result() for metric in self.metrics}


def _save_config(
    experiment_config: config.ArExperimentConfig,
    callback_root_dir: str,
    callback_name: str,
) -> None:
    logdir = pathlib.Path(callback_root_dir) / "logs" / callback_name
    logdir.mkdir(parents=True, exist_ok=True)
    experiment_config.to_json(str(logdir / "config.json"))


def _get_experiment_name(experiment_config: config.ArExperimentConfig) -> str:
    return (
        f"AR_ONSET-P{experiment_config.model.patch_frames}-"
        f"d{experiment_config.model.d_model}-"
        f"enc{experiment_config.model.n_enc_layers}-"
        f"dec{experiment_config.model.n_dec_layers}"
    )


def train_ar_onset(
    experiment_config: config.ArExperimentConfig,
    *,
    steps_per_epoch: int = 1,
) -> tuple[ArOnsetTrainingModel, keras.callbacks.History]:
    """Train AR onset on the configured overfit sample."""
    run_config = experiment_config.run
    if not run_config.model_output_dir:
        raise ValueError("run.model_output_dir is required")

    reproducibility.apply_training_seed(run_config.seed)
    base_model = models.build_ar_onset_model(experiment_config)
    training_model = ArOnsetTrainingModel(
        base_model,
        experiment_config=experiment_config,
    )
    training_model.compile(
        optimizer=keras.optimizers.Adam(  # type: ignore[arg-type]
            learning_rate=run_config.learning_rate,
            clipnorm=5.0,
        ),
    )

    train_ds = datasets.create_overfit_tf_dataset(experiment_config)
    val_ds = datasets.create_overfit_tf_dataset(experiment_config)
    train_ds = train_ds.take(steps_per_epoch)
    val_ds = val_ds.take(1)

    callbacks: list[keras.callbacks.Callback] = []
    monitor_metric = run_config.checkpoint_metric
    monitor_mode = "max"
    if run_config.callback_root_dir:
        experiment_name = _get_experiment_name(experiment_config)
        tb_callbacks, callback_name = event_trainers._get_callbacks(  # noqa: SLF001
            root_dir=run_config.callback_root_dir,
            monitor_metric=monitor_metric,
            monitor_mode=monitor_mode,
            experiment_name=experiment_name,
        )
        _save_config(experiment_config, run_config.callback_root_dir, callback_name)
        callbacks.extend(tb_callbacks)

    history = training_model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=run_config.epochs,
        callbacks=callbacks,
    )

    event_trainers._write_model(  # noqa: SLF001
        training_model,
        run_config.model_output_dir,
        callback_root_dir=run_config.callback_root_dir,
    )
    logging.info("Finished AR onset training; saved to %s", run_config.model_output_dir)
    return training_model, history
