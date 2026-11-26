"""JSON config for ITGPT hierarchical placement (`omalley2026itgpt`)."""

from __future__ import annotations

import dataclasses
import json
import pathlib

from stepcovnet.ddcl import constants as ddcl_constants
from stepcovnet.itgpt import constants


def _coerce_json_values(value: object) -> object:
    """Recursively convert values for JSON serialization.

    Args:
        value: Nested JSON-compatible object.

    Returns:
        JSON-serializable structure.
    """
    if isinstance(value, dict):
        return {key: _coerce_json_values(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_coerce_json_values(item) for item in value]
    return value


class _DictSerializableMixin:
    """Mixin providing default as_dict and from_dict for dataclass configs."""

    def as_dict(self) -> dict:
        """Convert config to dictionary for JSON serialization.

        Returns:
            JSON-serializable mapping of dataclass fields.
        """
        return _coerce_json_values(dataclasses.asdict(self))  # type: ignore[arg-type]

    @classmethod
    def from_dict(cls, data: dict):
        """Create config from dictionary.

        Args:
            data: Mapping of field names to values.

        Returns:
            Instance of the config class.
        """
        return cls(**data)


@dataclasses.dataclass
class ItgptDatasetConfig(_DictSerializableMixin):
    """Dataset settings for ITGPT placement.

    Attributes:
        training_index_path: Manifest of standard-difficulty charts.
        data_root: Prepared output root; inferred from the manifest when empty.
        batch_size: Charts per step (upstream generator is 1).
        max_train_songs: Cap unique train songs (-1 for all).
        max_val_songs: Cap unique val songs (-1 for all).
        cache_features: Write ``*.ddc_mel.npy`` beside audio.
        max_beats: Pad/truncate integer beats to a multiple of 64, capped here.
        exclude_normalized_ids: Song slugs dropped before load.
        audio_feature_type: Beat audio frontend (``logmel`` or ``mert``).
        n_frames_per_beat: Sub-beat frames per beat (32 for logmel, 24 for MERT).
        mert_cache_suffix: Sibling filename suffix for cached MERT arrays
            (default ``.mert.npy``).
        mert_spectral_flux: If True when ``audio_feature_type == "mert"``,
            append 6 z-scored 100 Hz 3-band spectral-flux/energy channels
            derived from ``*.ddc_mel.npy`` onto each MERT frame.
    """

    training_index_path: str
    data_root: str = ""
    batch_size: int = 1
    max_train_songs: int = -1
    max_val_songs: int = -1
    cache_features: bool = True
    max_beats: int = constants.MAX_BEATS
    exclude_normalized_ids: list[str] = dataclasses.field(default_factory=list)
    audio_feature_type: str = ddcl_constants.AUDIO_FEATURE_LOGMEL
    n_frames_per_beat: int = constants.N_FRAMES_PER_BEAT
    mert_cache_suffix: str = ddcl_constants.MERT_CACHE_SUFFIX
    mert_spectral_flux: bool = False

    def __post_init__(self) -> None:
        """Validate dataset hyperparameters.

        Raises:
            ValueError: If sizes, ``audio_feature_type``, or
                ``mert_cache_suffix`` are invalid.
        """
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be at least 1, got {self.batch_size}")
        if self.max_beats < constants.CHUNK_ALIGN:
            raise ValueError(
                f"max_beats must be >= {constants.CHUNK_ALIGN}, got {self.max_beats}"
            )
        if self.audio_feature_type not in ddcl_constants.AUDIO_FEATURE_TYPES:
            raise ValueError(
                "audio_feature_type must be one of "
                f"{ddcl_constants.AUDIO_FEATURE_TYPES}, got {self.audio_feature_type!r}"
            )
        if self.n_frames_per_beat < 1:
            raise ValueError(
                f"n_frames_per_beat must be at least 1, got {self.n_frames_per_beat}"
            )
        if (
            self.audio_feature_type == ddcl_constants.AUDIO_FEATURE_MERT
            and self.n_frames_per_beat != constants.CNN_FRAMES_OUT
        ):
            raise ValueError(
                "MERT ITGPT requires n_frames_per_beat == "
                f"{constants.CNN_FRAMES_OUT}, got {self.n_frames_per_beat}"
            )
        if not self.mert_cache_suffix.startswith(".") or not self.mert_cache_suffix.endswith(".npy"):
            raise ValueError(
                "mert_cache_suffix must start with '.' and end with '.npy', "
                f"got {self.mert_cache_suffix!r}"
            )


@dataclasses.dataclass
class ItgptModelConfig(_DictSerializableMixin):
    """Hierarchical transformer widths from ITGPT ``OnsetConfig``.

    Attributes:
        d_model: Transformer width (paper: 256).
        n_heads: Attention heads (paper: 8).
        n_enc_layers: Global encoder blocks (paper default CLI: 4; config: 8).
        cnn_hidden: Beat CNN base channels (paper: 32).
        dropout_rate: Attention / FFN dropout (paper: 0.1).
        early_film: If True, apply zero-initialized FiLM conditioning from
            ``[difficulty, bpm]`` before the hierarchical sub-beat blocks and
            before the global encoder.
        cyclic_measure_pos: If True, add zero-initialized cyclic ``beat % 4``
            (bar) and ``beat % 16`` (4-bar phrase) positional embeddings in
            ``level6``.
        continuous_beat_conv: If True, run ``ItgptBeatMertEncoder`` across the
            continuous ``(batch, beats * n_frames_per_beat, feature_dim)``
            stream so its ``Conv1D`` spans beat boundaries rather than
            zero-padding each beat independently.
    """

    d_model: int = constants.D_MODEL
    n_heads: int = constants.N_HEADS
    n_enc_layers: int = constants.N_ENC_LAYERS
    cnn_hidden: int = constants.CNN_HIDDEN
    dropout_rate: float = constants.DROPOUT_RATE
    early_film: bool = False
    cyclic_measure_pos: bool = False
    continuous_beat_conv: bool = False

    def __post_init__(self) -> None:
        """Validate model hyperparameters.

        Raises:
            ValueError: If widths are invalid.
        """
        if self.d_model < 1:
            raise ValueError(f"d_model must be at least 1, got {self.d_model}")
        if self.n_heads < 1 or self.d_model % self.n_heads != 0:
            raise ValueError(
                f"n_heads must divide d_model, got {self.n_heads} and {self.d_model}"
            )
        if self.n_enc_layers < 1:
            raise ValueError(
                f"n_enc_layers must be at least 1, got {self.n_enc_layers}"
            )
        if self.cnn_hidden < 1:
            raise ValueError(f"cnn_hidden must be at least 1, got {self.cnn_hidden}")
        if not 0.0 <= self.dropout_rate < 1.0:
            raise ValueError(f"dropout_rate must be in [0, 1), got {self.dropout_rate}")


@dataclasses.dataclass
class ItgptRunConfig(_DictSerializableMixin):
    """Optimization and artifact settings.

    Attributes:
        epoch: Number of training epochs (ITGPT default 200).
        steps_per_epoch: Train charts per epoch.
        validation_steps: Val charts per epoch.
        learning_rate: AdamW learning rate (ITGPT: 1e-4).
        weight_decay: AdamW weight decay (ITGPT: 1e-2).
        clipnorm: Global gradient clip (ITGPT: 1.0).
        seed: Reproducibility seed.
        model_output_dir: Directory for the saved ``.keras`` file.
        callback_root_dir: TensorBoard / checkpoint root.
        model_name: Artifact name suffix.
        resume: If True, ``BackupAndRestore`` continues an interrupted run.
        mixed_precision: If True, use Keras ``mixed_float16`` (8 GB GPU).
        jit_compile: If True, XLA-compile ``fit`` (can spike allocator workspace).
        eval_slot_f1_every: Evaluate full-val ``M-slot48`` every N epochs to log
            ``val_slot_f1_at_05`` and save ``best_slot_f1.keras`` (0 disables).
        lr_schedule: Optimizer schedule (``constant`` or ``cosine``).
        warmup_epochs: Linear warmup epochs before cosine decay (0 disables warmup).
        min_learning_rate: Floor LR at warmup start and cosine end.
        focal_gamma: Focal modulating exponent ``(1 - p_t)^gamma`` applied to
            grid-weighted slot BCE (``0.0`` keeps plain weighted BCE).
        grid_weight_24th: Slot loss weight for 24th/triplet grid indices
            ``(8, 16, 32, 40)`` (default ``1.0``).
        gradient_accumulation_steps: Number of micro-batches to accumulate
            before applying an optimizer update (``1`` disables accumulation).
        use_ema: If True, enable optimizer exponential moving average (EMA)
            weight tracking and evaluate/checkpoint with EMA weights via
            :class:`keras.callbacks.SwapEMAWeights`.
        ema_momentum: EMA decay factor in ``(0, 1)`` when ``use_ema`` is True
            (default ``0.99``).
        max_bar_shift: Maximum number of 4-beat bars of masked leading padding
            randomly prepended to each training chart (``0`` disables shift
            augmentation; e.g. ``15`` shifts by ``0..60`` beats across a
            64-beat ``level5`` chunk).
    """

    epoch: int = 8
    steps_per_epoch: int = 32
    validation_steps: int = 8
    learning_rate: float = constants.ADAM_LR
    weight_decay: float = constants.ADAM_WEIGHT_DECAY
    clipnorm: float = constants.ADAM_CLIPNORM
    seed: int = 42
    model_output_dir: str = "models_wsl/ddc/itgpt_placement_fraxtil_exp"
    callback_root_dir: str = "callbacks/ddc/itgpt_placement"
    model_name: str = "itgpt_placement_fraxtil_exp"
    resume: bool = True
    mixed_precision: bool = False
    jit_compile: bool = False
    eval_slot_f1_every: int = 0
    lr_schedule: str = constants.LR_SCHEDULE_CONSTANT
    warmup_epochs: int = 0
    min_learning_rate: float = 0.0
    focal_gamma: float = 0.0
    grid_weight_24th: float = constants.GRID_WEIGHT_24TH
    gradient_accumulation_steps: int = 1
    use_ema: bool = False
    ema_momentum: float = 0.99
    max_bar_shift: int = 0

    def __post_init__(self) -> None:
        """Validate run hyperparameters.

        Raises:
            ValueError: If ``epoch``, ``learning_rate``, ``weight_decay``,
                ``eval_slot_f1_every``, LR schedule, ``focal_gamma``,
                ``grid_weight_24th``, ``gradient_accumulation_steps``,
                ``ema_momentum``, or ``max_bar_shift`` fields are invalid.
        """
        if self.epoch < 1:
            raise ValueError(f"epoch must be at least 1, got {self.epoch}")
        if self.learning_rate <= 0.0:
            raise ValueError(
                f"learning_rate must be positive, got {self.learning_rate}"
            )
        if self.weight_decay < 0.0:
            raise ValueError(
                f"weight_decay must be >= 0, got {self.weight_decay}"
            )
        if self.eval_slot_f1_every < 0:
            raise ValueError(
                f"eval_slot_f1_every must be >= 0, got {self.eval_slot_f1_every}"
            )
        if self.lr_schedule not in constants.LR_SCHEDULES:
            raise ValueError(
                f"lr_schedule must be one of {constants.LR_SCHEDULES}, "
                f"got {self.lr_schedule!r}"
            )
        if self.warmup_epochs < 0:
            raise ValueError(
                f"warmup_epochs must be >= 0, got {self.warmup_epochs}"
            )
        if (
            self.lr_schedule == constants.LR_SCHEDULE_COSINE
            and self.warmup_epochs >= self.epoch
        ):
            raise ValueError(
                f"warmup_epochs ({self.warmup_epochs}) must be < epoch ({self.epoch})"
            )
        if not 0.0 <= self.min_learning_rate <= self.learning_rate:
            raise ValueError(
                "min_learning_rate must be in [0, learning_rate], "
                f"got {self.min_learning_rate} vs {self.learning_rate}"
            )
        if self.focal_gamma < 0.0:
            raise ValueError(
                f"focal_gamma must be >= 0, got {self.focal_gamma}"
            )
        if self.grid_weight_24th <= 0.0:
            raise ValueError(
                f"grid_weight_24th must be positive, got {self.grid_weight_24th}"
            )
        if self.gradient_accumulation_steps < 1:
            raise ValueError(
                "gradient_accumulation_steps must be at least 1, "
                f"got {self.gradient_accumulation_steps}"
            )
        if not 0.0 < self.ema_momentum < 1.0:
            raise ValueError(
                f"ema_momentum must be in (0, 1), got {self.ema_momentum}"
            )
        if self.max_bar_shift < 0:
            raise ValueError(
                f"max_bar_shift must be >= 0, got {self.max_bar_shift}"
            )


@dataclasses.dataclass
class ItgptExperimentConfig(_DictSerializableMixin):
    """Top-level ITGPT placement experiment config.

    Attributes:
        dataset: Data loading and batching.
        model: Transformer widths.
        run: Optimizer and artifact paths.
    """

    dataset: ItgptDatasetConfig
    model: ItgptModelConfig = dataclasses.field(default_factory=ItgptModelConfig)
    run: ItgptRunConfig = dataclasses.field(default_factory=ItgptRunConfig)

    def as_dict(self) -> dict:
        """Serialize nested configs.

        Returns:
            JSON-serializable mapping.
        """
        return {
            "dataset": self.dataset.as_dict(),
            "model": self.model.as_dict(),
            "run": self.run.as_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> ItgptExperimentConfig:
        """Load nested configs from a mapping.

        Args:
            data: JSON object with ``dataset`` / ``model`` / ``run`` keys.

        Returns:
            Experiment config.
        """
        return cls(
            dataset=ItgptDatasetConfig.from_dict(data["dataset"]),
            model=ItgptModelConfig.from_dict(data.get("model") or {}),
            run=ItgptRunConfig.from_dict(data.get("run") or {}),
        )

    @classmethod
    def from_json(cls, path: str | pathlib.Path) -> ItgptExperimentConfig:
        """Load an experiment config from a JSON file.

        Args:
            path: Path to an ITGPT placement JSON config.

        Returns:
            Experiment config.
        """
        with pathlib.Path(path).open(encoding="utf-8") as handle:
            payload = json.load(handle)
        return cls.from_dict(payload)
