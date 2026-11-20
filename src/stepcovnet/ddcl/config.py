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

    def __post_init__(self) -> None:
        """Validate dataset hyperparameters.

        Raises:
            ValueError: If ``batch_size``, ``memlen``, ``n_frames_per_beat``,
                or ``audio_feature_type`` is invalid.
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


@dataclasses.dataclass
class DdclModelConfig(_DictSerializableMixin):
    """Placement model hyperparameters for log-mel ConvLSTM or MERT LSTM.

    Attributes:
        lstm_units: Hidden size per LSTM layer.
        dropout_rate: Dropout on LSTM and dense layers.
        dense_sizes: Fully-connected hidden widths.
        mert_proj_dim: Per-frame projection width for MERT beat encoder.
        mert_conv_filters: Sub-beat 1D conv width for MERT beat encoder.
    """

    lstm_units: int = constants.LSTM_UNITS
    dropout_rate: float = constants.DROPOUT_RATE
    dense_sizes: list[int] = dataclasses.field(
        default_factory=lambda: list(constants.DENSE_SIZES)
    )
    mert_proj_dim: int = constants.MERT_PROJ_DIM
    mert_conv_filters: int = constants.MERT_CONV_FILTERS

    def __post_init__(self) -> None:
        """Validate model hyperparameters.

        Raises:
            ValueError: If widths or dropout are out of range.
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


@dataclasses.dataclass
class DdclRunConfig(_DictSerializableMixin):
    """Optimization and artifact settings.

    Attributes:
        epoch: Number of training epochs (DDCL default 100–300).
        steps_per_epoch: Train batches per epoch.
        validation_steps: Val batches per epoch.
        learning_rate: Adam learning rate (DDCL: 1e-4).
        clipnorm: Global gradient clip (DDCL: 1.0).
        seed: Reproducibility seed.
        model_output_dir: Directory for the saved ``.keras`` file.
        callback_root_dir: TensorBoard / checkpoint root.
        model_name: Artifact name suffix.
        resume: If True, ``BackupAndRestore`` continues an interrupted run
            in ``model_output_dir``.
    """

    epoch: int = 8
    steps_per_epoch: int = 100
    validation_steps: int = 20
    learning_rate: float = constants.ADAM_LR
    clipnorm: float = constants.ADAM_CLIPNORM
    seed: int = 42
    model_output_dir: str = "models_wsl/ddc/ddcl_placement_fraxtil"
    callback_root_dir: str = "callbacks/ddc/ddcl_placement"
    model_name: str = "ddcl_placement_fraxtil"
    resume: bool = True

    def __post_init__(self) -> None:
        """Validate run hyperparameters.

        Raises:
            ValueError: If ``epoch`` is not positive.
        """
        if self.epoch < 1:
            raise ValueError(f"epoch must be at least 1, got {self.epoch}")
        if self.learning_rate <= 0.0:
            raise ValueError(
                f"learning_rate must be positive, got {self.learning_rate}"
            )


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
