"""JSON config for DDCL ConvLSTM placement experiments (`omalley2025ddcl`)."""

from __future__ import annotations

import dataclasses
import json
import pathlib

from stepcovnet.ddcl import constants


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
class DdclDatasetConfig(_DictSerializableMixin):
    """Dataset settings for DDCL placement.

    Attributes:
        training_index_path: Manifest of standard-difficulty charts.
        data_root: Prepared output root; inferred from the manifest when empty.
        batch_size: Beats per training batch (DDCL default: 32).
        memlen: Past/future beats besides the current one (paper: 15).
        max_train_songs: Cap unique train songs (-1 for all).
        max_val_songs: Cap unique val songs (-1 for all).
        cache_features: Write ``*.ddc_mel.npy`` beside audio.
        exclude_normalized_ids: Song slugs dropped before load (for example
            charts whose BPM ladder the slot converter rejects).
        audio_feature_type: Beat audio frontend (``logmel`` or ``mert``).
        n_frames_per_beat: Sub-beat frames resampled per beat (32 for logmel,
            16 recommended for MERT).
        mert_cache_suffix: Sibling filename suffix for cached MERT arrays
            (default ``.mert.npy``).
        mert_spectral_flux: If True when ``audio_feature_type == "mert"``,
            concatenate 6 z-scored 100 Hz 3-band spectral-flux/energy channels
            derived from ``*.ddc_mel.npy`` onto each MERT frame.
        lazy_mert_cache_size: Optional bounded per-split LRU song cache size for
            on-demand MERT loading (0 = eager preload in RAM; >0 loads per-song
            MERT tensors lazily from disk on first access).
    """

    training_index_path: str
    data_root: str = ""
    batch_size: int = 32
    memlen: int = constants.MEMLEN
    max_train_songs: int = -1
    max_val_songs: int = -1
    cache_features: bool = True
    exclude_normalized_ids: list[str] = dataclasses.field(default_factory=list)
    audio_feature_type: str = constants.AUDIO_FEATURE_LOGMEL
    n_frames_per_beat: int = constants.N_FRAMES_PER_BEAT
    mert_cache_suffix: str = constants.MERT_CACHE_SUFFIX
    mert_spectral_flux: bool = False
    lazy_mert_cache_size: int = 0

    def __post_init__(self) -> None:
        """Validate dataset hyperparameters.

        Raises:
            ValueError: If ``batch_size``, ``memlen``, ``n_frames_per_beat``,
                ``audio_feature_type``, ``mert_cache_suffix``, or
                ``lazy_mert_cache_size`` is invalid.
        """
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be at least 1, got {self.batch_size}")
        if self.memlen < 0:
            raise ValueError(f"memlen must be >= 0, got {self.memlen}")
        if self.audio_feature_type not in constants.AUDIO_FEATURE_TYPES:
            raise ValueError(
                f"audio_feature_type must be one of {constants.AUDIO_FEATURE_TYPES}, "
                f"got {self.audio_feature_type!r}"
            )
        if self.n_frames_per_beat < 1:
            raise ValueError(
                f"n_frames_per_beat must be at least 1, got {self.n_frames_per_beat}"
            )
        if not self.mert_cache_suffix.startswith(
            "."
        ) or not self.mert_cache_suffix.endswith(".npy"):
            raise ValueError(
                "mert_cache_suffix must start with '.' and end with '.npy', "
                f"got {self.mert_cache_suffix!r}"
            )
        if self.lazy_mert_cache_size < 0:
            raise ValueError(
                f"lazy_mert_cache_size must be >= 0, got {self.lazy_mert_cache_size}"
            )


@dataclasses.dataclass
class DdclModelConfig(_DictSerializableMixin):
    """Placement model hyperparameters for log-mel ConvLSTM or MERT LSTM.

    Attributes:
        lstm_units: Hidden size per LSTM layer.
        dropout_rate: Dropout on LSTM and dense layers.
        dense_sizes: Fully-connected hidden widths.
        mert_proj_dim: Per-frame projection width for MERT beat encoder.
        mert_conv_filters: Sub-beat 1D conv width for MERT beat encoder.
        mert_noise_std: Optional Gaussian noise stddev after MERT LayerNorm
            during training (0.0 disables).
    """

    lstm_units: int = constants.LSTM_UNITS
    dropout_rate: float = constants.DROPOUT_RATE
    dense_sizes: list[int] = dataclasses.field(
        default_factory=lambda: list(constants.DENSE_SIZES)
    )
    mert_proj_dim: int = constants.MERT_PROJ_DIM
    mert_conv_filters: int = constants.MERT_CONV_FILTERS
    mert_noise_std: float = 0.0

    def __post_init__(self) -> None:
        """Validate model hyperparameters.

        Raises:
            ValueError: If widths, dropout, or noise stddev are out of range.
        """
        if self.lstm_units < 1:
            raise ValueError(f"lstm_units must be at least 1, got {self.lstm_units}")
        if not 0.0 <= self.dropout_rate < 1.0:
            raise ValueError(f"dropout_rate must be in [0, 1), got {self.dropout_rate}")
        if not self.dense_sizes or any(width < 1 for width in self.dense_sizes):
            raise ValueError("dense_sizes must be a non-empty list of positive ints")
        if self.mert_proj_dim < 1:
            raise ValueError(
                f"mert_proj_dim must be at least 1, got {self.mert_proj_dim}"
            )
        if self.mert_conv_filters < 1:
            raise ValueError(
                f"mert_conv_filters must be at least 1, got {self.mert_conv_filters}"
            )
        if self.mert_noise_std < 0.0:
            raise ValueError(f"mert_noise_std must be >= 0, got {self.mert_noise_std}")


@dataclasses.dataclass
class DdclRunConfig(_DictSerializableMixin):
    """Optimization and artifact settings.

    Attributes:
        epoch: Number of training epochs (DDCL default 100–300).
        steps_per_epoch: Train batches per epoch.
        validation_steps: Val batches per epoch.
        learning_rate: Adam / AdamW learning rate (DDCL: 1e-4).
        clipnorm: Global gradient clip (DDCL: 1.0).
        weight_decay: Decoupled AdamW weight decay (0.0 keeps plain Adam).
        eval_slot_f1_every: Evaluate full-val ``M-slot48`` every N epochs to log
            ``val_slot_f1_at_05`` and save ``best_slot_f1.pt`` (0 disables).
        seed: Reproducibility seed.
        model_output_dir: Directory for the saved ``.pt`` checkpoint.
        callback_root_dir: TensorBoard / checkpoint root.
        model_name: Artifact name suffix.
        resume: If True, resume from saved backup checkpoints in ``model_output_dir``.
        lr_schedule: Optimizer schedule (``constant`` or ``cosine``).
        warmup_epochs: Linear warmup epochs before cosine decay (0 disables).
        min_learning_rate: Floor LR at warmup start and cosine end.
        use_ema: If True, enable EMA weight tracking and evaluate/checkpoint
            with EMA weights.
        ema_momentum: EMA decay factor in ``(0, 1)`` when ``use_ema`` is True.
    """

    epoch: int = 8
    steps_per_epoch: int = 100
    validation_steps: int = 20
    learning_rate: float = constants.ADAM_LR
    clipnorm: float = constants.ADAM_CLIPNORM
    weight_decay: float = 0.0
    eval_slot_f1_every: int = 0
    seed: int = 42
    model_output_dir: str = "models_wsl/ddc/ddcl_placement_fraxtil"
    callback_root_dir: str = "callbacks/ddc/ddcl_placement"
    model_name: str = "ddcl_placement_fraxtil"
    resume: bool = True
    lr_schedule: str = constants.LR_SCHEDULE_CONSTANT
    warmup_epochs: int = 0
    min_learning_rate: float = 0.0
    use_ema: bool = False
    ema_momentum: float = 0.99

    def __post_init__(self) -> None:
        """Validate run hyperparameters.

        Raises:
            ValueError: If ``epoch``, ``learning_rate``, ``weight_decay``,
                ``eval_slot_f1_every``, LR schedule, or EMA fields are invalid.
        """
        if self.epoch < 1:
            raise ValueError(f"epoch must be at least 1, got {self.epoch}")
        if self.learning_rate <= 0.0:
            raise ValueError(
                f"learning_rate must be positive, got {self.learning_rate}"
            )
        if self.weight_decay < 0.0:
            raise ValueError(f"weight_decay must be >= 0, got {self.weight_decay}")
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
            raise ValueError(f"warmup_epochs must be >= 0, got {self.warmup_epochs}")
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
        if not 0.0 < self.ema_momentum < 1.0:
            raise ValueError(f"ema_momentum must be in (0, 1), got {self.ema_momentum}")


@dataclasses.dataclass
class DdclExperimentConfig(_DictSerializableMixin):
    """Top-level DDCL placement experiment config.

    Attributes:
        dataset: Data loading and batching.
        model: ConvLSTM widths.
        run: Optimizer and artifact paths.
    """

    dataset: DdclDatasetConfig
    model: DdclModelConfig = dataclasses.field(default_factory=DdclModelConfig)
    run: DdclRunConfig = dataclasses.field(default_factory=DdclRunConfig)

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
    def from_dict(cls, data: dict) -> DdclExperimentConfig:
        """Load nested configs from a mapping.

        Args:
            data: JSON object with ``dataset`` / ``model`` / ``run`` keys.

        Returns:
            Experiment config.
        """
        return cls(
            dataset=DdclDatasetConfig.from_dict(data["dataset"]),
            model=DdclModelConfig.from_dict(data.get("model") or {}),
            run=DdclRunConfig.from_dict(data.get("run") or {}),
        )

    @classmethod
    def from_json(cls, path: str | pathlib.Path) -> DdclExperimentConfig:
        """Load an experiment config from a JSON file.

        Args:
            path: Path to a DDCL placement JSON config.

        Returns:
            Experiment config.
        """
        with pathlib.Path(path).open(encoding="utf-8") as handle:
            payload = json.load(handle)
        return cls.from_dict(payload)
