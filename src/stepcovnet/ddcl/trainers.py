"""Train a DDCL ConvLSTM placement model (`omalley2025ddcl`)."""

from __future__ import annotations

import datetime
import json
import pathlib
import shutil
from collections.abc import Iterator
from typing import cast

import keras
import numpy as np
import tensorflow as tf

from stepcovnet.dataset_prep import training_index
from stepcovnet.ddcl import config, datasets, evaluation, models


def set_seed(seed: int) -> None:
    """Seed NumPy, Python, and TensorFlow RNGs.

    Args:
        seed: Reproducibility seed.
    """
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def compile_placement_model(
    model: keras.Model,
    run_config: config.DdclRunConfig,
) -> keras.Model:
    """Compile with DDCL Adam + binary focal loss.

    Matches ``get_onset_model`` in
    https://github.com/miguelomalley/DDCL/blob/5b1375c642bb708b3c66baf5d880fbf865b85097/models.py
    (Adam ``1e-4``, ``clipnorm=1``, ``BinaryFocalCrossentropy``).

    Args:
        model: Uncompiled ConvLSTM.
        run_config: Optimizer settings.

    Returns:
        The same model, compiled in place.
    """
    optimizer = keras.optimizers.Adam(
        learning_rate=run_config.learning_rate,
        clipnorm=run_config.clipnorm,
    )
    model.compile(
        optimizer=optimizer,
        loss=keras.losses.BinaryFocalCrossentropy(from_logits=False),
    )
    return model


def _batch_generator(
    charts: list[datasets.DdclChart],
    *,
    batch_size: int,
    seed: int,
) -> Iterator[tuple[dict[str, np.ndarray], np.ndarray]]:
    """Infinite generator of random-beat batches.

    Args:
        charts: Loaded train charts.
        batch_size: Beats per batch.
        seed: RNG seed.

    Yields:
        tuple[dict[str, np.ndarray], np.ndarray]: ``(inputs, target)`` pairs for
        ``model.fit``.
    """
    rng = np.random.default_rng(seed)
    while True:
        yield datasets.sample_train_batch(
            charts,
            batch_size=batch_size,
            rng=rng,
        )


def tensorboard_run_log_dir(callback_root_dir: str, model_name: str) -> pathlib.Path:
    """Return a per-run TensorBoard directory under the stage ``logs/`` tree.

    Args:
        callback_root_dir: Stage root shared by DDCL placement runs.
        model_name: Run label (appended after a timestamp).

    Returns:
        ``{callback_root_dir}/logs/{YYYYMMDD-HHMMSS}-{model_name}``.
    """
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    safe_name = str(model_name).strip() or "ddcl_placement"
    log_dir = pathlib.Path(callback_root_dir) / "logs" / f"{stamp}-{safe_name}"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


BEST_CHECKPOINT_FILENAME = "best.keras"
LAST_CHECKPOINT_FILENAME = "last.keras"
TRAIN_STATE_FILENAME = "train_state.json"
BACKUP_DIRNAME = "backup"
LAST_EVAL_FILENAME = "eval_val_slot48.json"
BEST_EVAL_FILENAME = "eval_val_slot48_best.json"


class TrainStateCallback(keras.callbacks.Callback):
    """Write completed-epoch count and best ``val_loss`` after each epoch."""

    def __init__(self, path: pathlib.Path, *, total_epochs: int) -> None:
        """Store the sidecar path.

        Args:
            path: JSON path under the model output directory.
            total_epochs: Configured ``fit`` epoch budget.
        """
        super().__init__()
        self.path = path
        self.total_epochs = int(total_epochs)
        self.best_val_loss: float | None = None
        if path.is_file():
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                payload = {}
            raw = payload.get("best_val_loss") if isinstance(payload, dict) else None
            if raw is not None:
                self.best_val_loss = float(raw)

    def on_epoch_end(self, epoch: int, logs: dict | None = None) -> None:
        """Persist resume metadata after the epoch flushes checkpoints.

        Args:
            epoch: Zero-based epoch index that just finished.
            logs: Keras metric dict (may include ``val_loss``).
        """
        logs = logs or {}
        val_loss = logs.get("val_loss")
        if val_loss is not None:
            current = float(val_loss)
            if self.best_val_loss is None or current < self.best_val_loss:
                self.best_val_loss = current
        payload = {
            "epoch": int(epoch) + 1,
            "best_val_loss": self.best_val_loss,
            "total_epochs": self.total_epochs,
        }
        self.path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def load_train_state(path: pathlib.Path) -> dict | None:
    """Load ``train_state.json`` if present.

    Args:
        path: Sidecar path.

    Returns:
        Parsed mapping, or None.
    """
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _write_eval_report(
    report: evaluation.Slot48EvalReport,
    path: pathlib.Path,
    *,
    weights: str,
) -> None:
    """Write a ``M-slot48`` eval JSON tagged with which weights were scored.

    Args:
        report: Val slot-F1 report.
        path: Output JSON path.
        weights: ``last`` or ``best``.
    """
    payload = report.as_dict()
    payload["weights"] = weights
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


def train_placement(
    experiment: config.DdclExperimentConfig,
    extra_callbacks: list[keras.callbacks.Callback] | None = None,
) -> keras.Model:
    """Load Dataset A charts, train the ConvLSTM, and save weights.

    Interrupted runs resume from ``{model_output_dir}/backup`` via Keras
    ``BackupAndRestore``. ``last.keras`` is rewritten every epoch so a reboot
    still leaves an evaluable checkpoint beside ``best.keras``.

    Args:
        experiment: Placement experiment config.
        extra_callbacks: Optional Keras callbacks (tests).

    Returns:
        Trained Keras model.
    """
    set_seed(experiment.run.seed)
    train_charts = datasets.load_split_charts(
        experiment.dataset,
        training_index.SPLIT_TRAIN,
    )
    val_charts = datasets.load_split_charts(
        experiment.dataset,
        training_index.SPLIT_VAL,
    )
    model = compile_placement_model(
        models.build_convlstm_placement_model(
            memlen=experiment.dataset.memlen,
            lstm_units=experiment.model.lstm_units,
            dropout_rate=experiment.model.dropout_rate,
            dense_sizes=tuple(experiment.model.dense_sizes),
            model_name=experiment.run.model_name,
        ),
        experiment.run,
    )
    output_dir = pathlib.Path(experiment.run.model_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    backup_dir = output_dir / BACKUP_DIRNAME
    if not experiment.run.resume and backup_dir.exists():
        shutil.rmtree(backup_dir)
    best_path = output_dir / BEST_CHECKPOINT_FILENAME
    last_path = output_dir / LAST_CHECKPOINT_FILENAME
    state_path = output_dir / TRAIN_STATE_FILENAME
    state = load_train_state(state_path) if experiment.run.resume else None
    best_threshold = None
    if state is not None and state.get("best_val_loss") is not None:
        best_threshold = float(state["best_val_loss"])
    if not model.built:
        sample, _ = datasets.sample_train_batch(
            train_charts,
            batch_size=1,
            rng=np.random.default_rng(experiment.run.seed),
        )
        model(sample, training=False)
    if experiment.run.resume and (backup_dir / "latest.weights.h5").is_file():
        print(f"resuming from {backup_dir}")
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
            initial_value_threshold=best_threshold,
        ),
        TrainStateCallback(state_path, total_epochs=experiment.run.epoch),
    ]
    if experiment.run.callback_root_dir:
        log_dir = tensorboard_run_log_dir(
            experiment.run.callback_root_dir,
            experiment.run.model_name,
        )
        callbacks.append(
            keras.callbacks.TensorBoard(
                log_dir=str(log_dir),
                histogram_freq=0,
                write_images=False,
            )
        )
    if extra_callbacks:
        callbacks.extend(extra_callbacks)
    model.fit(
        _batch_generator(
            train_charts,
            batch_size=experiment.dataset.batch_size,
            seed=experiment.run.seed,
        ),
        epochs=experiment.run.epoch,
        steps_per_epoch=max(1, experiment.run.steps_per_epoch),
        validation_data=_batch_generator(
            val_charts,
            batch_size=experiment.dataset.batch_size,
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
        cast(evaluation.SlotPredictor, model),
        val_charts,
        seed=experiment.run.seed,
    )
    _write_eval_report(
        last_report,
        output_dir / LAST_EVAL_FILENAME,
        weights="last",
    )
    if best_path.is_file():
        best_model = keras.models.load_model(best_path)
        best_report = evaluation.evaluate_slot48(
            cast(evaluation.SlotPredictor, best_model),
            val_charts,
            seed=experiment.run.seed,
        )
        _write_eval_report(
            best_report,
            output_dir / BEST_EVAL_FILENAME,
            weights="best",
        )
        print(f"saved {best_path}")
    return model
