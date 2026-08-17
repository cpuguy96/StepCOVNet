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
    """

    training_index_path: str
    data_root: str = ""
    batch_size: int = 32
    memlen: int = constants.MEMLEN
    max_train_songs: int = -1
    max_val_songs: int = -1
    cache_features: bool = True

    def __post_init__(self) -> None:
        """Validate dataset hyperparameters.

        Raises:
            ValueError: If ``batch_size`` is not positive or ``memlen`` is negative.
        """
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be at least 1, got {self.batch_size}")
        if self.memlen < 0:
            raise ValueError(f"memlen must be >= 0, got {self.memlen}")


@dataclasses.dataclass
class DdclModelConfig(_DictSerializableMixin):
    """ConvLSTM hyperparameters from DDCL ``get_onset_model``.

    Attributes:
        lstm_units: Hidden size per LSTM layer.
        dropout_rate: Dropout on LSTM and dense layers.
        dense_sizes: Fully-connected hidden widths.
    """

    lstm_units: int = constants.LSTM_UNITS
    dropout_rate: float = constants.DROPOUT_RATE
    dense_sizes: list[int] = dataclasses.field(
        default_factory=lambda: list(constants.DENSE_SIZES)
    )

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
