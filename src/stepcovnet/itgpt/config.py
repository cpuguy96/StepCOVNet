"""JSON config for ITGPT hierarchical placement (`omalley2026itgpt`)."""

from __future__ import annotations

import dataclasses
import json
import pathlib

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
    """

    training_index_path: str
    data_root: str = ""
    batch_size: int = 1
    max_train_songs: int = -1
    max_val_songs: int = -1
    cache_features: bool = True
    max_beats: int = constants.MAX_BEATS

    def __post_init__(self) -> None:
        """Validate dataset hyperparameters.

        Raises:
            ValueError: If sizes are invalid.
        """
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be at least 1, got {self.batch_size}")
        if self.max_beats < constants.CHUNK_ALIGN:
            raise ValueError(
                f"max_beats must be >= {constants.CHUNK_ALIGN}, got {self.max_beats}"
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
    """

    d_model: int = constants.D_MODEL
    n_heads: int = constants.N_HEADS
    n_enc_layers: int = constants.N_ENC_LAYERS
    cnn_hidden: int = constants.CNN_HIDDEN
    dropout_rate: float = constants.DROPOUT_RATE

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

    def __post_init__(self) -> None:
        """Validate run hyperparameters.

        Raises:
            ValueError: If ``epoch`` or learning rate are invalid.
        """
        if self.epoch < 1:
            raise ValueError(f"epoch must be at least 1, got {self.epoch}")
        if self.learning_rate <= 0.0:
            raise ValueError(
                f"learning_rate must be positive, got {self.learning_rate}"
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
