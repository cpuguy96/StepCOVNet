"""Configuration classes for dataset, model, and training run parameters.

This module provides typed configuration objects for better tracking and
reproducibility of experiments. Configs can be serialized to/from JSON
for saving with runs and loading for re-running experiments.
"""

from __future__ import annotations

import dataclasses
import json
import os


@dataclasses.dataclass
class OnsetDatasetConfig:
    """Configuration for onset detection dataset creation.

    Attributes:
        data_dir: Path to training data directory.
        val_data_dir: Path to validation data directory.
        batch_size: Number of samples per batch.
        apply_temporal_augment: Whether to apply temporal augmentation during training.
        should_apply_spec_augment: Whether to apply spectrogram augmentation during training.
        use_gaussian_target: Whether to use Gaussian targets instead of binary targets.
        gaussian_sigma: Standard deviation for Gaussian target distribution.
    """

    data_dir: str
    val_data_dir: str
    batch_size: int = 1
    apply_temporal_augment: bool = False
    should_apply_spec_augment: bool = False
    use_gaussian_target: bool = False
    gaussian_sigma: float = 1.0

    def as_dict(self) -> dict:
        """Convert config to dictionary for JSON serialization.

        Returns:
            Dictionary representation of the config with all fields.
        """
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> OnsetDatasetConfig:
        """Create config from dictionary.

        Args:
            data: Dictionary containing config fields. Must include 'data_dir'
                and 'val_data_dir', other fields are optional and will use defaults.

        Returns:
            OnsetDatasetConfig instance created from the dictionary.
        """
        return cls(**data)


@dataclasses.dataclass
class ArrowDatasetConfig:
    """Configuration for arrow classification dataset creation.

    Attributes:
        data_dir: Path to training data directory.
        val_data_dir: Path to validation data directory.
        batch_size: Number of samples per batch.
        snippet_half_frames: Half-window of frames around each onset (total frames = 2*snippet_half_frames+1).
            When > 0, audio snippets are loaded and included per step; when 0, only timing and chart are used.
    """

    data_dir: str
    val_data_dir: str
    batch_size: int = 1
    snippet_half_frames: int = 0

    def as_dict(self) -> dict:
        """Convert config to dictionary for JSON serialization.

        Returns:
            Dictionary representation of the config with all fields.
        """
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> ArrowDatasetConfig:
        """Create config from dictionary.

        Args:
            data: Dictionary containing config fields. Must include 'data_dir'
                and 'val_data_dir', other fields are optional and will use defaults.

        Returns:
            ArrowDatasetConfig instance created from the dictionary.
        """
        return cls(**data)


@dataclasses.dataclass
class OnsetModelConfig:
    """Configuration for U-Net WaveNet model architecture.

    Attributes:
        initial_filters: Number of filters in the first layer (doubles at each level).
        depth: Number of downsampling/upsampling levels in the U-Net.
        dilation_rates: List of dilation factors for convolutions within each level.
        kernel_size: Size of convolutional kernels.
        dropout_rate: Dropout rate for regularization.
    """

    initial_filters: int = 16
    depth: int = 2
    dilation_rates: list[int] = dataclasses.field(default_factory=lambda: [1, 2, 4, 8])
    kernel_size: int = 3
    dropout_rate: float = 0.0

    def as_dict(self) -> dict:
        """Convert config to dictionary for JSON serialization.

        Returns:
            Dictionary representation of the config with all fields.
        """
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> OnsetModelConfig:
        """Create config from dictionary.

        Args:
            data: Dictionary containing config fields. All fields are optional
                and will use defaults if not provided.

        Returns:
            OnsetModelConfig instance created from the dictionary.
        """
        return cls(**data)


@dataclasses.dataclass
class TransformerArrowParams:
    """Parameters for the transformer-based arrow model. Used when model_type is 'transformer'."""

    num_layers: int = 1
    d_model: int = 128
    num_heads: int = 4
    ff_dim: int = 512
    dropout_rate: float = 0.0

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> TransformerArrowParams:
        return cls(
            **{
                k: v
                for k, v in data.items()
                if k in {f.name for f in dataclasses.fields(cls)}
            }
        )


@dataclasses.dataclass
class MLPArrowParams:
    """Parameters for the MLP-based arrow model. Used when model_type is 'mlp'."""

    hidden_dims: list[int] = dataclasses.field(default_factory=lambda: [256, 128])
    dropout_rate: float = 0.0

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> MLPArrowParams:
        return cls(
            **{
                k: v
                for k, v in data.items()
                if k in {f.name for f in dataclasses.fields(cls)}
            }
        )


# Flat transformer keys used for backward compatibility when parsing old configs.
_TRANSFORMER_FLAT_KEYS = frozenset(
    {"num_layers", "d_model", "num_heads", "ff_dim", "dropout_rate"}
)


@dataclasses.dataclass
class ArrowModelConfig:
    """Configuration for arrow classification model architecture.

    Supports multiple model types via nested architecture-specific params.
    Shared: model_type (which architecture), snippet_half_frames (input option).
    Per-architecture blocks: transformer (TransformerArrowParams), mlp (MLPArrowParams).
    Only the block matching model_type is required when building; others can be None.
    """

    model_type: str = "transformer"
    snippet_half_frames: int = 0
    transformer: TransformerArrowParams | None = None
    mlp: MLPArrowParams | None = None

    def as_dict(self) -> dict:
        """Convert config to dictionary for JSON serialization (nested shape)."""
        out: dict = {
            "model_type": self.model_type,
            "snippet_half_frames": self.snippet_half_frames,
        }
        if self.transformer is not None:
            out["transformer"] = self.transformer.as_dict()
        if self.mlp is not None:
            out["mlp"] = self.mlp.as_dict()
        return out

    @classmethod
    def from_dict(cls, data: dict) -> ArrowModelConfig:
        """Create config from dictionary. Supports nested format and flat (legacy) format."""
        model_type = data.get("model_type", "transformer")
        snippet_half_frames = data.get("snippet_half_frames", 0)

        # Nested: explicit "transformer" or "mlp" blocks
        if "transformer" in data:
            transformer = TransformerArrowParams.from_dict(data["transformer"])
        elif model_type == "transformer" and _TRANSFORMER_FLAT_KEYS.intersection(data):
            # Backward compat: flat transformer keys at top level (only when transformer is active)
            flat = {k: data[k] for k in _TRANSFORMER_FLAT_KEYS if k in data}
            transformer = TransformerArrowParams.from_dict(flat)
        else:
            transformer = (
                TransformerArrowParams() if model_type == "transformer" else None
            )

        # Overlay flat keys at top level (e.g. from apply_overrides with model.dropout_rate)
        flat_overlay = {k: data[k] for k in _TRANSFORMER_FLAT_KEYS if k in data}
        if flat_overlay and transformer is not None:
            merged = {**transformer.as_dict(), **flat_overlay}
            transformer = TransformerArrowParams.from_dict(merged)

        if "mlp" in data:
            mlp = MLPArrowParams.from_dict(data["mlp"])
        else:
            mlp = MLPArrowParams() if model_type == "mlp" else None

        return cls(
            model_type=model_type,
            snippet_half_frames=snippet_half_frames,
            transformer=transformer,
            mlp=mlp,
        )


@dataclasses.dataclass
class RunConfig:
    """Configuration for training run parameters (shared by onset and arrow).

    Attributes:
        epoch: Number of epochs to train for.
        take_count: Number of batches to use from training dataset (-1 for entire dataset).
        val_take_count: Number of batches to use from validation dataset (-1 for entire dataset).
        model_output_dir: Directory where trained model will be saved.
        callback_root_dir: Root directory for storing training callbacks (checkpoints, logs).
        model_name: Name of the model. If empty, generated from experiment name.
        seed: Random seed for reproducibility (optional).
        show_model_summary: If True, print model summary before training. Default True.
        fit_verbose: Keras model.fit verbosity: 0 (silent), 1 (progress bar), or 2 (one line per epoch).
            Default 1.
    """

    epoch: int
    take_count: int
    model_output_dir: str
    callback_root_dir: str = ""
    model_name: str = ""
    seed: int | None = None
    val_take_count: int = -1
    show_model_summary: bool = True
    fit_verbose: int = 1

    def __post_init__(self) -> None:
        """Validate run parameters."""
        if self.epoch < 1:
            raise ValueError(f"epoch must be at least 1, got {self.epoch}")
        if self.take_count != -1 and self.take_count < 1:
            raise ValueError(
                "take_count must be -1 (entire dataset) or at least 1, "
                f"got {self.take_count}"
            )
        if self.val_take_count != -1 and self.val_take_count < 1:
            raise ValueError(
                "val_take_count must be -1 (entire dataset) or at least 1, "
                f"got {self.val_take_count}"
            )
        if self.fit_verbose not in (0, 1, 2):
            raise ValueError(f"fit_verbose must be 0, 1, or 2, got {self.fit_verbose}")

    def as_dict(self) -> dict:
        """Convert config to dictionary for JSON serialization.

        Returns:
            Dictionary representation of the config with all fields.
        """
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> RunConfig:
        """Create config from dictionary. Only valid RunConfig field names are accepted."""
        return cls(**data)


@dataclasses.dataclass
class ArrowRunConfig(RunConfig):
    """Arrow-specific run configuration: RunConfig plus aux loss weights.

    Used only by ArrowExperimentConfig. Onset experiments use RunConfig instead.

    Attributes:
        chart_validity_aux_weight: Weight for chart-validity auxiliary loss.
            Higher values punish invalid charts more; use with diversity_aux_weight to avoid
            collapse to boring (e.g. all-tap) charts. Default 0.0.
        diversity_aux_weight: Weight for note-kind balance auxiliary loss.
            Encourages predicted hold/tap mix to match labels; use to balance chart_validity.
            Default 0.0.
        warmup_epochs: Number of epochs for linear LR warmup before cosine decay. 0 disables the schedule (fixed LR).
        lr_peak: Peak learning rate reached at end of warmup (also the fixed LR when warmup is disabled).
        lr_min: Minimum learning rate at start of warmup and end of cosine decay.
    """

    chart_validity_aux_weight: float = 0.0
    diversity_aux_weight: float = 0.0
    warmup_epochs: int = 0
    lr_peak: float = 1e-3
    lr_min: float = 1e-5

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.chart_validity_aux_weight < 0:
            raise ValueError(
                f"chart_validity_aux_weight must be >= 0, got {self.chart_validity_aux_weight}"
            )
        if self.diversity_aux_weight < 0:
            raise ValueError(
                f"diversity_aux_weight must be >= 0, got {self.diversity_aux_weight}"
            )
        if self.warmup_epochs < 0:
            raise ValueError(f"warmup_epochs must be >= 0, got {self.warmup_epochs}")
        if self.warmup_epochs >= self.epoch:
            raise ValueError(
                f"warmup_epochs ({self.warmup_epochs}) must be < epoch ({self.epoch})"
            )
        if self.lr_peak <= 0:
            raise ValueError(f"lr_peak must be > 0, got {self.lr_peak}")
        if self.lr_min < 0:
            raise ValueError(f"lr_min must be >= 0, got {self.lr_min}")
        if self.lr_min >= self.lr_peak:
            raise ValueError(
                f"lr_min ({self.lr_min}) must be < lr_peak ({self.lr_peak})"
            )

    @classmethod
    def from_dict(cls, data: dict) -> ArrowRunConfig:
        """Create config from dictionary. Only valid ArrowRunConfig field names are accepted."""
        return cls(**data)


@dataclasses.dataclass
class OnsetExperimentConfig:
    """Complete configuration for an onset detection experiment.

    Attributes:
        dataset: OnsetDatasetConfig object containing dataset configuration.
        model: OnsetModelConfig object containing model architecture configuration.
        run: RunConfig object containing training run parameters.
    """

    dataset: OnsetDatasetConfig
    model: OnsetModelConfig
    run: RunConfig

    def as_dict(self) -> dict:
        """Convert config to dictionary for JSON serialization.

        Returns:
            Dictionary representation containing nested dictionaries for
            'dataset', 'model', and 'run' configurations.
        """
        return {
            "dataset": self.dataset.as_dict(),
            "model": self.model.as_dict(),
            "run": self.run.as_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> OnsetExperimentConfig:
        """Create config from dictionary.

        Args:
            data: Dictionary containing 'dataset', 'model', and 'run' keys,
                each containing their respective configuration dictionaries.

        Returns:
            OnsetExperimentConfig instance created from the dictionary.

        Raises:
            KeyError: If required keys ('dataset', 'model', 'run') are missing.
        """
        return cls(
            dataset=OnsetDatasetConfig.from_dict(data["dataset"]),
            model=OnsetModelConfig.from_dict(data["model"]),
            run=RunConfig.from_dict(data["run"]),
        )

    def to_json(self, path: str):
        """Save config to JSON file.

        Creates the directory if it doesn't exist, then writes the config
        as a formatted JSON file.

        Args:
            path: File path where the JSON config will be saved.
        """
        os.makedirs(
            os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True
        )
        with open(path, "w") as f:
            json.dump(self.as_dict(), f, indent=2)

    @classmethod
    def from_json(cls, path: str) -> OnsetExperimentConfig:
        """Load config from JSON file.

        Args:
            path: File path to the JSON config file.

        Returns:
            OnsetExperimentConfig instance loaded from the JSON file.

        Raises:
            FileNotFoundError: If the config file doesn't exist.
            json.JSONDecodeError: If the file contains invalid JSON.
            KeyError: If required keys are missing from the JSON.
        """
        with open(path) as f:
            data = json.load(f)
        return cls.from_dict(data)


@dataclasses.dataclass
class ArrowExperimentConfig:
    """Complete configuration for an arrow classification experiment.

    Combines dataset, model, and run configurations into a single object.

    Attributes:
        dataset: ArrowDatasetConfig object containing dataset configuration.
        model: ArrowModelConfig object containing model architecture configuration.
        run: ArrowRunConfig object containing training run parameters (includes aux loss weights).
    """

    dataset: ArrowDatasetConfig
    model: ArrowModelConfig
    run: ArrowRunConfig

    def as_dict(self) -> dict:
        """Convert config to dictionary for JSON serialization.

        Returns:
            Dictionary representation containing nested dictionaries for
            'dataset', 'model', and 'run' configurations.
        """
        return {
            "dataset": self.dataset.as_dict(),
            "model": self.model.as_dict(),
            "run": self.run.as_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> ArrowExperimentConfig:
        """Create config from dictionary.

        Args:
            data: Dictionary containing 'dataset', 'model', and 'run' keys,
                each containing their respective configuration dictionaries.

        Returns:
            ArrowExperimentConfig instance created from the dictionary.

        Raises:
            KeyError: If required keys ('dataset', 'model', 'run') are missing.
        """
        return cls(
            dataset=ArrowDatasetConfig.from_dict(data["dataset"]),
            model=ArrowModelConfig.from_dict(data["model"]),
            run=ArrowRunConfig.from_dict(data["run"]),
        )

    def to_json(self, path: str):
        """Save config to JSON file.

        Creates the directory if it doesn't exist, then writes the config
        as a formatted JSON file.

        Args:
            path: File path where the JSON config will be saved.
        """
        os.makedirs(
            os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True
        )
        with open(path, "w") as f:
            json.dump(self.as_dict(), f, indent=2)

    @classmethod
    def from_json(cls, path: str) -> ArrowExperimentConfig:
        """Load config from JSON file.

        Args:
            path: File path to the JSON config file.

        Returns:
            ArrowExperimentConfig instance loaded from the JSON file.

        Raises:
            FileNotFoundError: If the config file doesn't exist.
            json.JSONDecodeError: If the file contains invalid JSON.
            KeyError: If required keys are missing from the JSON.
        """
        with open(path) as f:
            data = json.load(f)
        return cls.from_dict(data)
