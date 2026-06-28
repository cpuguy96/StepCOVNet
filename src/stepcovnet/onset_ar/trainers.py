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
        self.lambda_time_final = run_config.lambda_time
        self.lambda_time_ramp_epochs = run_config.lambda_time_ramp_epochs
        self.lambda_time = lambda_time_for_epoch(
            -1,
            lambda_time_final=self.lambda_time_final,
            ramp_epochs=self.lambda_time_ramp_epochs,
        )
        self.length_normalize_ce = run_config.length_normalize_ce
        self.use_soft_pointer_time = run_config.use_soft_pointer_time
        self.lambda_residual = run_config.lambda_residual
        self.pointer_loss_weight = run_config.pointer_loss_weight
        self.scheduled_sampling_max_p = run_config.scheduled_sampling_max_p
        self.scheduled_sampling_ramp_epochs = run_config.scheduled_sampling_ramp_epochs
        self.scheduled_sampling_warmup_epochs = (
            run_config.scheduled_sampling_warmup_epochs
        )
        self.scheduled_sampling_p = scheduled_sampling_for_epoch(
            -1,
            max_p=self.scheduled_sampling_max_p,
            ramp_epochs=self.scheduled_sampling_ramp_epochs,
            warmup_epochs=self.scheduled_sampling_warmup_epochs,
        )
        self.max_decoder_len = experiment_config.max_decoder_len()
        self.tolerance_sec = run_config.tolerance_sec
        self.experiment_config = experiment_config
        self.token_class_weights = self._build_token_class_weights(experiment_config)
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.token_loss_tracker = keras.metrics.Mean(name="token_loss")
        self.pointer_loss_tracker = keras.metrics.Mean(name="pointer_loss")
        self.time_loss_tracker = keras.metrics.Mean(name="time_loss")
        self.residual_loss_tracker = keras.metrics.Mean(name="residual_loss")
        self.token_accuracy = keras.metrics.Mean(name="token_accuracy")
        self.event_f1_metric = ArEventOnsetF1Metric(
            tolerance_sec=run_config.tolerance_sec,
            name="event_onset_f1",
        )
        self.ar_decode_f1_metric = ArEventOnsetF1Metric(
            tolerance_sec=run_config.tolerance_sec,
            name="ar_decode_event_f1",
        )
        self.ar_decode_length_metric = keras.metrics.Mean(name="ar_decode_length")
        self.ar_decode_n_onsets_metric = keras.metrics.Mean(name="ar_decode_n_onsets")
        self._last_ar_tp = 0.0
        self._last_ar_fp = 0.0
        self._last_ar_fn = 0.0
        self._last_ar_decode_length = 0.0
        self._last_ar_decode_n_onsets = 0.0

    @property
    def metrics(self):
        return [
            self.loss_tracker,
            self.token_loss_tracker,
            self.pointer_loss_tracker,
            self.time_loss_tracker,
            self.residual_loss_tracker,
            self.token_accuracy,
            self.event_f1_metric,
            self.ar_decode_f1_metric,
            self.ar_decode_length_metric,
            self.ar_decode_n_onsets_metric,
        ]

    @staticmethod
    def _build_token_class_weights(
        experiment_config: config.ArExperimentConfig,
    ) -> tf.Tensor | None:
        """Precompute tide/overfit token CE weights from the single training batch."""
        scheme = experiment_config.run.token_class_weight
        if scheme == "none":
            return None
        batch_np = datasets.sample_to_training_batch(
            datasets.load_overfit_sample(experiment_config),
            experiment_config,
        )
        weights = losses.build_token_class_weights_numpy(
            batch_np["decoder_target_ids"][0],
            batch_np["decoder_mask"][0],
            vocab_size=experiment_config.build_vocab().vocab_size,
            scheme=scheme,
            eos_token_weight_scale=experiment_config.run.eos_token_weight_scale,
        )
        if weights is None:
            return None
        return tf.constant(weights, dtype=tf.float32)

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
        decoder_input_ids: tf.Tensor | None = None,
    ) -> tuple[tf.Tensor, dict[str, tf.Tensor], dict[str, tf.Tensor]]:
        model_inputs = self._model_inputs(batch)
        if decoder_input_ids is not None:
            model_inputs = {
                **model_inputs,
                "decoder_input_ids": decoder_input_ids,
            }
        outputs = self.base_model(model_inputs, training=training)
        total_loss, parts = losses.compute_ar_onset_loss(
            outputs,
            batch,
            patch_frames=self.patch_frames,
            hop_sec=self.hop_sec,
            lambda_time=self.lambda_time,
            lambda_residual=self.lambda_residual,
            pointer_loss_weight=self.pointer_loss_weight,
            length_normalize_ce=self.length_normalize_ce,
            token_class_weights=self.token_class_weights,
            use_soft_pointer_time=self.use_soft_pointer_time,
        )
        return total_loss, parts, outputs

    def _update_teacher_fed_f1(
        self,
        outputs: dict[str, tf.Tensor],
        batch: dict[str, tf.Tensor],
    ) -> None:
        pred_times, pred_mask = inference.decode_teacher_fed_times_tf(
            outputs,
            batch,
            patch_frames=self.patch_frames,
            hop_sec=self.hop_sec,
            use_soft_expected=self.use_soft_pointer_time,
        )
        self.event_f1_metric.update_state(
            pred_times,
            pred_mask,
            batch["gt_times"],
            batch["gt_mask"],
        )

    def run_ar_decode_eval_eager(
        self,
        mert_patches: np.ndarray,
        patch_mask: np.ndarray,
        gt_times: np.ndarray,
        gt_mask: np.ndarray,
    ) -> tuple[float, float, float, float, float]:
        """Free-running AR decode in eager mode; return TP/FP/FN and decode stats."""
        tp, fp, fn, length, n_onsets = self._ar_decode_eval_wrapper(
            mert_patches,
            patch_mask,
            gt_times,
            gt_mask,
        )
        return (
            float(tp),
            float(fp),
            float(fn),
            float(length),
            float(n_onsets),
        )

    def set_ar_decode_metrics(
        self,
        tp: float,
        fp: float,
        fn: float,
        decode_length: float,
        n_onsets: float,
    ) -> None:
        """Publish AR-decode counts to metrics and the carry-forward cache."""
        self.ar_decode_f1_metric.true_positives.assign(
            tf.cast(tp, self.ar_decode_f1_metric.dtype),
        )
        self.ar_decode_f1_metric.false_positives.assign(
            tf.cast(fp, self.ar_decode_f1_metric.dtype),
        )
        self.ar_decode_f1_metric.false_negatives.assign(
            tf.cast(fn, self.ar_decode_f1_metric.dtype),
        )
        self.ar_decode_length_metric.reset_state()
        self.ar_decode_n_onsets_metric.reset_state()
        self.ar_decode_length_metric.update_state(decode_length)
        self.ar_decode_n_onsets_metric.update_state(n_onsets)
        self._last_ar_tp = tp
        self._last_ar_fp = fp
        self._last_ar_fn = fn
        self._last_ar_decode_length = decode_length
        self._last_ar_decode_n_onsets = n_onsets

    def restore_ar_decode_metrics_from_cache(self) -> None:
        """Reuse last AR-decode values when skipping expensive free-run val."""
        self.set_ar_decode_metrics(
            self._last_ar_tp,
            self._last_ar_fp,
            self._last_ar_fn,
            self._last_ar_decode_length,
            self._last_ar_decode_n_onsets,
        )

    def _ar_decode_eval_wrapper(
        self,
        mert_patches: np.ndarray,
        patch_mask: np.ndarray,
        gt_times: np.ndarray,
        gt_mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        decode_stats = inference.decode_autoregressive_with_stats_numpy(
            self.base_model,
            mert_patches,
            patch_mask,
            max_decoder_len=self.max_decoder_len,
            patch_frames=self.patch_frames,
            hop_sec=self.hop_sec,
            experiment_config=self.experiment_config,
        )
        pred_mask = np.ones((decode_stats.times.size,), dtype=np.float32)
        tp, fp, fn = _ar_event_onset_counts_numpy(
            decode_stats.times,
            pred_mask,
            np.asarray(gt_times).reshape(-1),
            np.asarray(gt_mask).reshape(-1),
            tolerance_sec=self.tolerance_sec,
        )
        return (
            np.array(tp, dtype=np.float64),
            np.array(fp, dtype=np.float64),
            np.array(fn, dtype=np.float64),
            np.array(decode_stats.n_forward_steps, dtype=np.float64),
            np.array(decode_stats.n_onset_tokens, dtype=np.float64),
        )

    def train_step(self, data):
        batch = self._unpack_batch(data)
        probe_outputs = None
        if self.scheduled_sampling_p > 0.0:
            _, _, probe_outputs = self._forward_and_loss(batch, training=True)
        with tf.GradientTape() as tape:
            if probe_outputs is not None:
                mixed_inputs = inference.build_scheduled_decoder_inputs(
                    batch["decoder_input_ids"],
                    tf.stop_gradient(probe_outputs["token_logits"]),
                    batch["decoder_mask"],
                    self.scheduled_sampling_p,
                )
                total_loss, parts, outputs = self._forward_and_loss(
                    batch,
                    training=True,
                    decoder_input_ids=mixed_inputs,
                )
            else:
                total_loss, parts, outputs = self._forward_and_loss(
                    batch,
                    training=True,
                )
        trainable_vars = self.base_model.trainable_variables
        grads = tape.gradient(total_loss, trainable_vars)
        self.optimizer.apply_gradients(zip(grads, trainable_vars, strict=False))
        self.loss_tracker.update_state(total_loss)
        self.token_loss_tracker.update_state(parts["token_loss"])
        self.pointer_loss_tracker.update_state(parts["pointer_loss"])
        self.time_loss_tracker.update_state(parts["time_loss"])
        self.residual_loss_tracker.update_state(parts["residual_loss"])
        self.token_accuracy.update_state(
            losses.masked_token_accuracy(
                outputs["token_logits"],
                batch["decoder_target_ids"],
                batch["decoder_mask"],
            ),
        )
        self._update_teacher_fed_f1(outputs, batch)
        return {metric.name: metric.result() for metric in self.metrics}

    def test_step(self, data):
        batch = self._unpack_batch(data)
        total_loss, parts, outputs = self._forward_and_loss(batch, training=False)
        self.loss_tracker.update_state(total_loss)
        self.token_loss_tracker.update_state(parts["token_loss"])
        self.pointer_loss_tracker.update_state(parts["pointer_loss"])
        self.time_loss_tracker.update_state(parts["time_loss"])
        self.residual_loss_tracker.update_state(parts["residual_loss"])
        self.token_accuracy.update_state(
            losses.masked_token_accuracy(
                outputs["token_logits"],
                batch["decoder_target_ids"],
                batch["decoder_mask"],
            ),
        )
        self._update_teacher_fed_f1(outputs, batch)
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


def lambda_time_for_epoch(
    epoch_index: int,
    *,
    lambda_time_final: float,
    ramp_epochs: int,
) -> float:
    """Linear ramp of ``lambda_time`` from 0 to ``lambda_time_final``."""
    if ramp_epochs <= 0:
        return float(lambda_time_final)
    if lambda_time_final <= 0.0:
        return 0.0
    progress = min(1.0, float(epoch_index + 1) / float(ramp_epochs))
    return float(lambda_time_final) * progress


def should_run_ar_decode_validation(
    epoch_index: int,
    *,
    every_n_epochs: int,
) -> bool:
    """Return whether free-running AR decode should run this validation epoch."""
    if every_n_epochs <= 0:
        return False
    if every_n_epochs == 1:
        return True
    return epoch_index % every_n_epochs == 0


class ArDecodeValidationCallback(keras.callbacks.Callback):
    """Run free-running AR decode eagerly after fast teacher-fed validation."""

    def __init__(
        self,
        training_model: ArOnsetTrainingModel,
        *,
        experiment_config: config.ArExperimentConfig,
        every_n_epochs: int,
    ) -> None:
        super().__init__()
        self.training_model = training_model
        self.every_n_epochs = int(every_n_epochs)
        self._val_batch = datasets.sample_to_training_batch(
            datasets.load_overfit_sample(experiment_config),
            experiment_config,
        )

    def on_epoch_end(self, epoch: int, logs: dict | None = None) -> None:
        if logs is None:
            logs = {}
        if should_run_ar_decode_validation(epoch, every_n_epochs=self.every_n_epochs):
            tp, fp, fn, decode_length, n_onsets = (
                self.training_model.run_ar_decode_eval_eager(
                    self._val_batch["mert_patches"],
                    self._val_batch["patch_mask"],
                    self._val_batch["gt_times"],
                    self._val_batch["gt_mask"],
                )
            )
            self.training_model.set_ar_decode_metrics(
                tp,
                fp,
                fn,
                decode_length,
                n_onsets,
            )
        else:
            self.training_model.restore_ar_decode_metrics_from_cache()

        logs["val_ar_decode_event_f1"] = float(
            self.training_model.ar_decode_f1_metric.result(),
        )
        logs["val_ar_decode_length"] = float(
            self.training_model.ar_decode_length_metric.result(),
        )
        logs["val_ar_decode_n_onsets"] = float(
            self.training_model.ar_decode_n_onsets_metric.result(),
        )


def scheduled_sampling_for_epoch(
    epoch_index: int,
    *,
    max_p: float,
    ramp_epochs: int,
    warmup_epochs: int = 0,
) -> float:
    """Linear ramp of scheduled sampling probability from 0 to ``max_p``."""
    if max_p <= 0.0:
        return 0.0
    if epoch_index < warmup_epochs:
        return 0.0
    if ramp_epochs <= 0:
        return float(max_p)
    progress = min(
        1.0,
        float(epoch_index - warmup_epochs + 1) / float(ramp_epochs),
    )
    return float(max_p) * progress


class ScheduledSamplingRampCallback(keras.callbacks.Callback):
    """Update ``ArOnsetTrainingModel.scheduled_sampling_p`` each epoch."""

    def __init__(
        self,
        training_model: ArOnsetTrainingModel,
        *,
        max_p: float,
        ramp_epochs: int,
        warmup_epochs: int = 0,
    ) -> None:
        super().__init__()
        self.training_model = training_model
        self.max_p = float(max_p)
        self.ramp_epochs = int(ramp_epochs)
        self.warmup_epochs = int(warmup_epochs)

    def on_epoch_begin(self, epoch: int, logs: dict | None = None) -> None:
        _ = logs
        self.training_model.scheduled_sampling_p = scheduled_sampling_for_epoch(
            epoch,
            max_p=self.max_p,
            ramp_epochs=self.ramp_epochs,
            warmup_epochs=self.warmup_epochs,
        )


class LambdaTimeRampCallback(keras.callbacks.Callback):
    """Update ``ArOnsetTrainingModel.lambda_time`` each epoch."""

    def __init__(
        self,
        training_model: ArOnsetTrainingModel,
        *,
        lambda_time_final: float,
        ramp_epochs: int,
    ) -> None:
        super().__init__()
        self.training_model = training_model
        self.lambda_time_final = float(lambda_time_final)
        self.ramp_epochs = int(ramp_epochs)

    def on_epoch_begin(self, epoch: int, logs: dict | None = None) -> None:
        _ = logs
        self.training_model.lambda_time = lambda_time_for_epoch(
            epoch,
            lambda_time_final=self.lambda_time_final,
            ramp_epochs=self.ramp_epochs,
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
    if run_config.init_model_path:
        init_path = pathlib.Path(run_config.init_model_path)
        if not init_path.is_file():
            raise FileNotFoundError(f"init_model_path not found: {init_path}")
        init_model = keras.models.load_model(str(init_path), compile=False)
        base_model.set_weights(init_model.get_weights())
        logging.info("Loaded AR onset weights from %s", init_path)
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

    if run_config.lambda_time_ramp_epochs > 0 and run_config.lambda_time > 0.0:
        callbacks.append(
            LambdaTimeRampCallback(
                training_model,
                lambda_time_final=run_config.lambda_time,
                ramp_epochs=run_config.lambda_time_ramp_epochs,
            ),
        )

    if (
        run_config.scheduled_sampling_ramp_epochs > 0
        and run_config.scheduled_sampling_max_p > 0.0
    ):
        callbacks.append(
            ScheduledSamplingRampCallback(
                training_model,
                max_p=run_config.scheduled_sampling_max_p,
                ramp_epochs=run_config.scheduled_sampling_ramp_epochs,
                warmup_epochs=run_config.scheduled_sampling_warmup_epochs,
            ),
        )

    if run_config.ar_decode_val_every_n_epochs > 0:
        callbacks.insert(
            0,
            ArDecodeValidationCallback(
                training_model,
                experiment_config=experiment_config,
                every_n_epochs=run_config.ar_decode_val_every_n_epochs,
            ),
        )

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
