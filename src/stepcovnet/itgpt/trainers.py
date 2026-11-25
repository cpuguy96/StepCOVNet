"""Train an ITGPT hierarchical placement model (`omalley2026itgpt`)."""

from __future__ import annotations

import datetime
import json
import pathlib
import shutil
from typing import cast

import keras
import numpy as np
import tensorflow as tf
from keras import ops

from stepcovnet.ddcl import constants as ddcl_constants
from stepcovnet.ddcl import datasets as ddcl_datasets
from stepcovnet.ddcl import evaluation as ddcl_eval
from stepcovnet.itgpt import config, constants, datasets, evaluation, models


def set_seed(seed: int) -> None:
    """Seed NumPy and TensorFlow RNGs.

    Args:
        seed: Reproducibility seed.
    """
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


@keras.saving.register_keras_serializable(package="stepcovnet.itgpt")
def placement_binary_crossentropy(y_true: object, y_pred: object) -> object:
    """Return per-beat mean of ITGPT grid-weighted BCE.

    Keras reduces this ``(batch, beats)`` tensor against the beat mask
    ``sample_weight``. Slot grid weights (16th=2, 24th=1, 32nd=1, micro=0.5)
    are applied inside the loss so they are not collapsed by the default
    last-axis mean.

    Args:
        y_true: Slot labels ``(batch, beats, 48)``.
        y_pred: Slot probabilities ``(batch, beats, 48)``.

    Returns:
        Weighted BCE ``(batch, beats)``.
    """
    epsilon = keras.backend.epsilon()
    y_pred = ops.clip(ops.cast(y_pred, "float32"), epsilon, 1.0 - epsilon)
    y_true = ops.cast(y_true, "float32")
    per_slot = -(y_true * ops.log(y_pred) + (1.0 - y_true) * ops.log(1.0 - y_pred))
    weighted = per_slot * models.grid_importance_weights()
    return ops.mean(weighted, axis=-1)


def load_placement_model(path: str | pathlib.Path) -> keras.Model:
    """Load a saved ITGPT placement model, including the custom loss.

    Args:
        path: Path to a ``.keras`` checkpoint.

    Returns:
        Loaded Keras model.
    """
    return keras.models.load_model(
        path,
        custom_objects={
            "placement_binary_crossentropy": placement_binary_crossentropy,
            "ItgptPlacementModel": models.ItgptPlacementModel,
        },
    )


def build_learning_rate(
    run_config: config.ItgptRunConfig,
) -> float | keras.optimizers.schedules.LearningRateSchedule:
    """Return a constant LR or step-level warmup + cosine decay schedule.

    Args:
        run_config: Optimizer and epoch-count settings.

    Returns:
        Scalar LR when ``lr_schedule == "constant"``, or a Keras
        :class:`keras.optimizers.schedules.CosineDecay` schedule when
        ``lr_schedule == "cosine"``.
    """
    if run_config.lr_schedule == constants.LR_SCHEDULE_CONSTANT:
        return float(run_config.learning_rate)
    steps_per_epoch = max(1, int(run_config.steps_per_epoch))
    total_steps = int(run_config.epoch) * steps_per_epoch
    warmup_steps = int(run_config.warmup_epochs) * steps_per_epoch
    decay_steps = max(1, total_steps - warmup_steps)
    alpha = float(run_config.min_learning_rate) / float(run_config.learning_rate)
    if warmup_steps > 0:
        return keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=float(run_config.min_learning_rate),
            decay_steps=decay_steps,
            alpha=alpha,
            warmup_target=float(run_config.learning_rate),
            warmup_steps=warmup_steps,
        )
    return keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=float(run_config.learning_rate),
        decay_steps=decay_steps,
        alpha=alpha,
    )


def compile_placement_model(
    model: keras.Model,
    run_config: config.ItgptRunConfig,
) -> keras.Model:
    """Compile with Adam/AdamW + slot-weighted binary cross-entropy.

    Upstream uses ``BCEWithLogitsLoss`` times grid weights. The Keras port
    emits probabilities; grid weights live in the loss and the beat mask is
    ``sample_weight``. Uses ``AdamW`` when ``weight_decay > 0`` and ``Adam``
    when ``weight_decay == 0``.

    Args:
        model: Uncompiled placement model.
        run_config: Optimizer settings.

    Returns:
        The same model, compiled in place.
    """
    learning_rate = build_learning_rate(run_config)
    if run_config.weight_decay > 0.0:
        optimizer: keras.optimizers.Optimizer = keras.optimizers.AdamW(
            learning_rate=learning_rate,
            weight_decay=run_config.weight_decay,
            clipnorm=run_config.clipnorm,
        )
    else:
        optimizer = keras.optimizers.Adam(
            learning_rate=learning_rate,
            clipnorm=run_config.clipnorm,
        )
    model.compile(
        optimizer=optimizer,
        loss=placement_binary_crossentropy,
        jit_compile=run_config.jit_compile,
    )
    return model


def tensorboard_run_log_dir(callback_root_dir: str, model_name: str) -> pathlib.Path:
    """Return a per-run TensorBoard directory under the stage ``logs/`` tree.

    Args:
        callback_root_dir: Stage root shared by ITGPT placement runs.
        model_name: Run label.

    Returns:
        ``{callback_root_dir}/logs/{YYYYMMDD-HHMMSS}-{model_name}``.
    """
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    safe_name = str(model_name).strip() or "itgpt_placement"
    log_dir = pathlib.Path(callback_root_dir) / "logs" / f"{stamp}-{safe_name}"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


BEST_CHECKPOINT_FILENAME = "best.keras"
BEST_SLOT_F1_CHECKPOINT_FILENAME = "best_slot_f1.keras"
LAST_CHECKPOINT_FILENAME = "last.keras"
BACKUP_DIRNAME = "backup"
LAST_EVAL_FILENAME = "eval_val_slot48.json"
BEST_EVAL_FILENAME = "eval_val_slot48_best.json"
BEST_SLOT_F1_EVAL_FILENAME = "eval_val_slot48_best_f1.json"


class ItgptValSlot48CheckpointCallback(keras.callbacks.Callback):
    """Evaluate full-val ``M-slot48`` periodically and save best-F1 weights.

    Place before :class:`keras.callbacks.TensorBoard` so
    ``val_slot_f1_at_05`` and ``val_slot_f1_max`` are recorded in event logs.

    Args:
        val_charts: Loaded validation charts scored on schedule.
        filepath: Destination ``.keras`` checkpoint for best ``f1_at_05``.
        eval_every: Epoch cadence (positive integer).
        total_epochs: Total epochs in ``fit`` so the final epoch is scored.
        max_beats: Pad cap passed to ``evaluation.evaluate_slot48``.
        seed: RNG seed passed to ``evaluation.evaluate_slot48``.
    """

    def __init__(
        self,
        val_charts: list[ddcl_datasets.DdclChart],
        filepath: pathlib.Path,
        *,
        eval_every: int,
        total_epochs: int,
        max_beats: int,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.val_charts = val_charts
        self.filepath = pathlib.Path(filepath)
        self.eval_every = int(eval_every)
        self.total_epochs = int(total_epochs)
        self.max_beats = int(max_beats)
        self.seed = int(seed)
        self.best_f1: float | None = None
        self.best_epoch: int | None = None

    def on_epoch_end(self, epoch: int, logs: dict | None = None) -> None:
        """Score full val split on schedule and checkpoint best ``f1_at_05``.

        Args:
            epoch: Zero-based completed epoch index.
            logs: Mutable Keras metric dictionary for TensorBoard.
        """
        step_epoch = int(epoch) + 1
        if self.eval_every < 1 or (
            step_epoch % self.eval_every != 0 and step_epoch != self.total_epochs
        ):
            return
        report = evaluation.evaluate_slot48(
            cast(ddcl_eval.SlotPredictor, self.model),
            self.val_charts,
            seed=self.seed,
            max_beats=self.max_beats,
        )
        if logs is not None:
            logs["val_slot_f1_at_05"] = float(report.f1_at_05)
            logs["val_slot_f1_max"] = float(report.f1_max)
        if self.best_f1 is None or report.f1_at_05 > self.best_f1:
            prev = "None" if self.best_f1 is None else f"{self.best_f1:.5f}"
            self.best_f1 = float(report.f1_at_05)
            self.best_epoch = step_epoch
            self.model.save(self.filepath)
            print(
                f"\nEpoch {step_epoch}: val_slot_f1_at_05 improved from {prev} to "
                f"{report.f1_at_05:.5f} (f1_max={report.f1_max:.5f} @ "
                f"{report.best_threshold:.2f}), saving model to {self.filepath}",
                flush=True,
            )
        else:
            print(
                f"\nEpoch {step_epoch}: val_slot_f1_at_05={report.f1_at_05:.5f} "
                f"(f1_max={report.f1_max:.5f}; best={self.best_f1:.5f} @ "
                f"ep {self.best_epoch})",
                flush=True,
            )


def _write_eval_report(
    report: ddcl_eval.Slot48EvalReport,
    path: pathlib.Path,
    *,
    weights: str,
) -> None:
    """Write ``M-slot48`` JSON.

    Args:
        report: Pooled scores.
        path: Destination file.
        weights: ``last`` or ``best``.
    """
    path.write_text(
        json.dumps(evaluation.report_as_dict(report, weights=weights), indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(evaluation.report_as_dict(report, weights=weights), indent=2))


def train_placement(
    experiment: config.ItgptExperimentConfig,
    extra_callbacks: list[keras.callbacks.Callback] | None = None,
) -> keras.Model:
    """Load Dataset B charts, train the transformer, and save weights.

    Args:
        experiment: Placement experiment config.
        extra_callbacks: Optional Keras callbacks (tests).

    Returns:
        Trained Keras model.

    Raises:
        ValueError: If the train or val split is empty.
    """
    set_seed(experiment.run.seed)
    if experiment.run.mixed_precision:
        keras.mixed_precision.set_global_policy("mixed_float16")
        print("ITGPT training: mixed_float16 policy enabled")
    train_charts = datasets.load_split_charts(
        experiment.dataset,
        "train",
    )
    val_charts = datasets.load_split_charts(
        experiment.dataset,
        "val",
    )
    if not train_charts:
        raise ValueError("train split is empty")
    if not val_charts:
        raise ValueError("val split is empty")
    feature_dim = (
        int(train_charts[0].beat_audio.shape[-1])
        if experiment.dataset.audio_feature_type == ddcl_constants.AUDIO_FEATURE_MERT
        else ddcl_constants.MERT_FEATURE_DIM
    )
    model = build_and_compile(experiment, feature_dim=feature_dim)
    dummy_inputs, _, _ = datasets.pack_chart(
        train_charts[0], max_beats=experiment.dataset.max_beats
    )
    model(dummy_inputs, training=False)
    output_dir = pathlib.Path(experiment.run.model_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    backup_dir = output_dir / BACKUP_DIRNAME
    if not experiment.run.resume and backup_dir.exists():
        shutil.rmtree(backup_dir)
    best_path = output_dir / BEST_CHECKPOINT_FILENAME
    best_slot_f1_path = output_dir / BEST_SLOT_F1_CHECKPOINT_FILENAME
    last_path = output_dir / LAST_CHECKPOINT_FILENAME
    callbacks: list[keras.callbacks.Callback] = [
        keras.callbacks.BackupAndRestore(
            backup_dir=str(backup_dir),
            save_freq="epoch",
            double_checkpoint=True,
            delete_checkpoint=True,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=str(last_path),
            save_best_only=False,
            verbose=0,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=str(best_path),
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            verbose=1,
        ),
    ]
    if experiment.run.eval_slot_f1_every > 0:
        callbacks.append(
            ItgptValSlot48CheckpointCallback(
                val_charts,
                best_slot_f1_path,
                eval_every=experiment.run.eval_slot_f1_every,
                total_epochs=experiment.run.epoch,
                max_beats=experiment.dataset.max_beats,
                seed=experiment.run.seed,
            )
        )
    if experiment.run.callback_root_dir:
        log_dir = tensorboard_run_log_dir(
            experiment.run.callback_root_dir,
            experiment.run.model_name,
        )
        callbacks.append(
            keras.callbacks.TensorBoard(
                log_dir=str(log_dir),
                histogram_freq=0,
                write_graph=False,
                write_images=False,
            )
        )
    if extra_callbacks:
        callbacks.extend(extra_callbacks)
    model.fit(
        datasets.tf_dataset(
            train_charts,
            max_beats=experiment.dataset.max_beats,
            seed=experiment.run.seed,
        ),
        epochs=experiment.run.epoch,
        steps_per_epoch=max(1, experiment.run.steps_per_epoch),
        validation_data=datasets.tf_dataset(
            val_charts,
            max_beats=experiment.dataset.max_beats,
            seed=experiment.run.seed + 1,
        ),
        validation_steps=max(1, experiment.run.validation_steps),
        callbacks=callbacks,
        verbose="auto",
        shuffle=False,
    )
    save_path = output_dir / f"{experiment.run.model_name}.keras"
    model.save(save_path)
    print(f"saved {save_path}")
    last_report = evaluation.evaluate_slot48(
        cast(ddcl_eval.SlotPredictor, model),
        val_charts,
        seed=experiment.run.seed,
        max_beats=experiment.dataset.max_beats,
    )
    _write_eval_report(last_report, output_dir / LAST_EVAL_FILENAME, weights="last")
    if best_path.is_file():
        best_model = load_placement_model(best_path)
        best_report = evaluation.evaluate_slot48(
            cast(ddcl_eval.SlotPredictor, best_model),
            val_charts,
            seed=experiment.run.seed,
            max_beats=experiment.dataset.max_beats,
        )
        _write_eval_report(best_report, output_dir / BEST_EVAL_FILENAME, weights="best")
        print(f"saved {best_path}")
    if best_slot_f1_path.is_file():
        best_f1_model = load_placement_model(best_slot_f1_path)
        best_f1_report = evaluation.evaluate_slot48(
            cast(ddcl_eval.SlotPredictor, best_f1_model),
            val_charts,
            seed=experiment.run.seed,
            max_beats=experiment.dataset.max_beats,
        )
        _write_eval_report(
            best_f1_report,
            output_dir / BEST_SLOT_F1_EVAL_FILENAME,
            weights="best_slot_f1",
        )
        print(f"saved {best_slot_f1_path}")
    return model


def build_and_compile(
    experiment: config.ItgptExperimentConfig,
    *,
    feature_dim: int = ddcl_constants.MERT_FEATURE_DIM,
) -> keras.Model:
    """Build and compile the placement model.

    Args:
        experiment: Experiment config.
        feature_dim: Per-frame feature width when ``audio_feature_type == "mert"``.

    Returns:
        Compiled Keras model.
    """
    model = models.build_itgpt_placement_model(
        d_model=experiment.model.d_model,
        n_heads=experiment.model.n_heads,
        n_enc_layers=experiment.model.n_enc_layers,
        cnn_hidden=experiment.model.cnn_hidden,
        dropout_rate=experiment.model.dropout_rate,
        max_beats=experiment.dataset.max_beats,
        audio_feature_type=experiment.dataset.audio_feature_type,
        n_frames_per_beat=experiment.dataset.n_frames_per_beat,
        feature_dim=feature_dim,
        model_name=experiment.run.model_name,
    )
    return compile_placement_model(model, experiment.run)
