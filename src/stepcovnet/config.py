"""Configuration classes for dataset, model, and training run parameters.

This module provides typed configuration objects for better tracking and
reproducibility of experiments. Configs can be serialized to/from JSON
for saving with runs and loading for re-running experiments.
"""

from __future__ import annotations

import dataclasses
import json
import os
from typing import Protocol


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
        use_interval: If True, include inter-step interval (time since previous step) as an input.
        interval_encoding: How to encode interval: "default", "log" (log(1+interval)), or "multi" (extra channels).
            Must match model config. Default "default".
        use_step_index: If True, include step index (position in sequence) as an input. Must match model config.
        use_beat_phase: If True, include beat/phase features (BPM from chart txt). Must match model config.
        use_aux_interval_target: If True, include aux_interval_target (next-step interval) in batch for auxiliary loss.
    """

    data_dir: str
    val_data_dir: str
    batch_size: int = 1
    snippet_half_frames: int = 0
    use_interval: bool = False
    interval_encoding: str = "default"
    use_step_index: bool = False
    use_beat_phase: bool = False
    use_aux_interval_target: bool = False

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


class ArrowParamsProtocol(Protocol):
    """Protocol for arrow model params. All arrow param classes must implement this.

    Implementors must provide as_dict() and experiment_name_parts().
    """

    def as_dict(self) -> dict: ...
    def experiment_name_parts(self) -> list[str]: ...


@dataclasses.dataclass
class TransformerArrowParams:
    """Parameters for the transformer-based arrow model. Used when model_type is 'transformer'.

    Attributes:
        num_layers: Number of transformer encoder layers.
        d_model: Model dimension (embedding and hidden size).
        num_heads: Number of attention heads.
        ff_dim: Feed-forward inner dimension.
        dropout_rate: Dropout rate applied in sublayers.
        use_timing_position: If True, use timing-based positional encoding instead of sinusoidal.
    """

    num_layers: int = 1
    d_model: int = 128
    num_heads: int = 4
    ff_dim: int = 512
    dropout_rate: float = 0.0
    use_timing_position: bool = False

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)

    def experiment_name_parts(self) -> list[str]:
        parts = [
            f"att_layers_{self.num_layers}",
            f"d_model_{self.d_model}",
            f"num_heads_{self.num_heads}",
            f"ff_dim_{self.ff_dim}",
            f"dropout_{str(self.dropout_rate).replace('.', '_')}",
        ]
        if self.use_timing_position:
            parts.append("timing_pos")
        return parts

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
    """Parameters for the MLP-based arrow model. Used when model_type is 'mlp'.

    Attributes:
        hidden_dims: List of hidden layer dimensions (e.g. [256, 128]).
        dropout_rate: Dropout rate between dense layers.
    """

    hidden_dims: list[int] = dataclasses.field(default_factory=lambda: [256, 128])
    dropout_rate: float = 0.0

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)

    def experiment_name_parts(self) -> list[str]:
        return [
            "mlp_" + "_".join(str(d) for d in self.hidden_dims),
            f"dropout_{str(self.dropout_rate).replace('.', '_')}",
        ]

    @classmethod
    def from_dict(cls, data: dict) -> MLPArrowParams:
        return cls(
            **{
                k: v
                for k, v in data.items()
                if k in {f.name for f in dataclasses.fields(cls)}
            }
        )


@dataclasses.dataclass
class LSTMArrowParams:
    """Parameters for the LSTM-based arrow model. Used when model_type is 'lstm'.

    Attributes:
        units: Number of LSTM units per layer.
        num_layers: Number of stacked LSTM layers.
        dropout_rate: Dropout rate for the LSTM layers.
        bidirectional: If True, use bidirectional LSTM.
    """

    units: int = 128
    num_layers: int = 1
    dropout_rate: float = 0.0
    bidirectional: bool = False

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)

    def experiment_name_parts(self) -> list[str]:
        parts = [
            f"lstm_units_{self.units}",
            f"lstm_layers_{self.num_layers}",
            f"dropout_{str(self.dropout_rate).replace('.', '_')}",
        ]
        if self.bidirectional:
            parts.append("lstm_bidir")
        return parts

    @classmethod
    def from_dict(cls, data: dict) -> LSTMArrowParams:
        return cls(
            **{
                k: v
                for k, v in data.items()
                if k in {f.name for f in dataclasses.fields(cls)}
            }
        )


@dataclasses.dataclass
class GRUArrowParams:
    """Parameters for the GRU-based arrow model. Used when model_type is 'gru'.

    Attributes:
        units: Number of GRU units per layer.
        num_layers: Number of stacked GRU layers.
        dropout_rate: Dropout rate for the GRU layers.
        bidirectional: If True, use bidirectional GRU.
        add_attention_layer: If True, add a multi-head self-attention layer after the GRU stack.
        attention_heads: Number of attention heads when add_attention_layer is True.
        attention_dim: Dimension per head (or total key dim) when add_attention_layer is True.
    """

    units: int = 128
    num_layers: int = 1
    dropout_rate: float = 0.0
    bidirectional: bool = False
    add_attention_layer: bool = False
    attention_heads: int = 4
    attention_dim: int = 64

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)

    def experiment_name_parts(self) -> list[str]:
        parts = [
            f"gru_units_{self.units}",
            f"gru_layers_{self.num_layers}",
            f"dropout_{str(self.dropout_rate).replace('.', '_')}",
        ]
        if self.bidirectional:
            parts.append("gru_bidir")
        if self.add_attention_layer:
            parts.append("attn")
            parts.append(f"attn_heads_{self.attention_heads}")
            parts.append(f"attn_dim_{self.attention_dim}")
        return parts

    @classmethod
    def from_dict(cls, data: dict) -> GRUArrowParams:
        return cls(
            **{
                k: v
                for k, v in data.items()
                if k in {f.name for f in dataclasses.fields(cls)}
            }
        )


@dataclasses.dataclass
class TCNArrowParams:
    """Parameters for the TCN (Temporal Convolutional Network) arrow model. Used when model_type is 'tcn'.

    Attributes:
        filters: Number of convolutional filters per layer.
        kernel_size: Size of the causal convolution kernel.
        num_layers: Number of TCN blocks/layers.
        dilation_base: Base for exponential dilation (dilation = dilation_base^layer_idx).
        dropout_rate: Dropout rate applied in the TCN stack.
    """

    filters: int = 64
    kernel_size: int = 3
    num_layers: int = 4
    dilation_base: int = 2
    dropout_rate: float = 0.0

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)

    def experiment_name_parts(self) -> list[str]:
        return [
            f"tcn_filters_{self.filters}",
            f"tcn_kernel_{self.kernel_size}",
            f"tcn_layers_{self.num_layers}",
            f"tcn_dilation_base_{self.dilation_base}",
            f"dropout_{str(self.dropout_rate).replace('.', '_')}",
        ]

    @classmethod
    def from_dict(cls, data: dict) -> TCNArrowParams:
        return cls(
            **{
                k: v
                for k, v in data.items()
                if k in {f.name for f in dataclasses.fields(cls)}
            }
        )


@dataclasses.dataclass
class CNN1DArrowParams:
    """Parameters for the 1D CNN arrow model. Used when model_type is 'cnn1d'.

    Attributes:
        filters: Number of convolutional filters per layer.
        kernel_sizes: List of kernel sizes per layer (e.g. [3, 3, 3]).
        dropout_rate: Dropout rate applied after conv stack.
    """

    filters: int = 64
    kernel_sizes: list[int] = dataclasses.field(default_factory=lambda: [3, 3, 3])
    dropout_rate: float = 0.0

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)

    def experiment_name_parts(self) -> list[str]:
        return [
            f"cnn1d_filters_{self.filters}",
            "cnn1d_kernels_" + "_".join(str(k) for k in self.kernel_sizes),
            f"dropout_{str(self.dropout_rate).replace('.', '_')}",
        ]

    @classmethod
    def from_dict(cls, data: dict) -> CNN1DArrowParams:
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

# Registry: model_type -> attribute name on ArrowModelConfig. Used for serialization and active block lookup.
_ARROW_MODEL_TYPE_ATTR: dict[str, str] = {
    "transformer": "transformer",
    "mlp": "mlp",
    "lstm": "lstm",
    "gru": "gru",
    "tcn": "tcn",
    "cnn1d": "cnn1d",
}


@dataclasses.dataclass
class ArrowModelConfig:
    """Configuration for arrow classification model architecture.

    Supports multiple model types via nested architecture-specific params.
    Only the block matching model_type is required when building; others can be None.

    Attributes:
        model_type: One of 'transformer', 'mlp', 'lstm', 'gru', 'tcn', 'cnn1d'.
        snippet_half_frames: Half-window of frames per step (0 = timing only).
        use_interval: If True, model expects interval_input (time since previous step).
        interval_encoding: How to encode interval: "default", "log", or "multi". Must match dataset config.
        use_step_index: If True, model expects step_index input. Must match dataset config.
        use_beat_phase: If True, model expects beat_phase input (BPM from chart). Must match dataset config.
        transformer: Params for transformer model; used when model_type is 'transformer'.
        mlp: Params for MLP model; used when model_type is 'mlp'.
        lstm: Params for LSTM model; used when model_type is 'lstm'.
        gru: Params for GRU model; used when model_type is 'gru'.
        tcn: Params for TCN model; used when model_type is 'tcn'.
        cnn1d: Params for 1D CNN model; used when model_type is 'cnn1d'.
    """

    model_type: str = "transformer"
    snippet_half_frames: int = 0
    use_interval: bool = False
    interval_encoding: str = "default"
    use_step_index: bool = False
    use_beat_phase: bool = False
    transformer: TransformerArrowParams | None = None
    mlp: MLPArrowParams | None = None
    lstm: LSTMArrowParams | None = None
    gru: GRUArrowParams | None = None
    tcn: TCNArrowParams | None = None
    cnn1d: CNN1DArrowParams | None = None

    def get_active_params_block(self) -> ArrowParamsProtocol | None:
        """Return the params block for the current model_type, or None if not set."""
        attr = _ARROW_MODEL_TYPE_ATTR.get(self.model_type)
        if attr is None:
            return None
        return getattr(self, attr, None)  # type: ignore[return-value]

    def get_experiment_name_parts(self) -> list[str]:
        """Return experiment name fragments: active params block plus input-related options."""
        block = self.get_active_params_block()
        parts = block.experiment_name_parts() if block is not None else []
        if self.snippet_half_frames > 0:
            parts.append(f"snippets_half_{self.snippet_half_frames}")
        if self.use_interval:
            parts.append("use_interval")
            if self.interval_encoding != "default":
                parts.append(f"interval_enc_{self.interval_encoding}")
        if self.use_step_index:
            parts.append("use_step_index")
        if self.use_beat_phase:
            parts.append("use_beat_phase")
        return parts

    def as_dict(self) -> dict:
        """Convert config to dictionary for JSON serialization (nested shape)."""
        out: dict = {
            "model_type": self.model_type,
            "snippet_half_frames": self.snippet_half_frames,
            "use_interval": self.use_interval,
            "interval_encoding": self.interval_encoding,
            "use_step_index": self.use_step_index,
            "use_beat_phase": self.use_beat_phase,
        }
        for _model_type, attr in _ARROW_MODEL_TYPE_ATTR.items():
            block = getattr(self, attr, None)
            if block is not None:
                out[attr] = block.as_dict()
        return out

    @classmethod
    def from_dict(cls, data: dict) -> ArrowModelConfig:
        """Create config from dictionary. Supports nested format and flat (legacy) format."""
        model_type = data.get("model_type", "transformer")
        snippet_half_frames = data.get("snippet_half_frames", 0)
        use_interval = data.get("use_interval", False)
        interval_encoding = data.get("interval_encoding", "default")
        use_step_index = data.get("use_step_index", False)
        use_beat_phase = data.get("use_beat_phase", False)

        # Parse active block; transformer has flat-key backward compat.
        transformer: TransformerArrowParams | None = None
        mlp: MLPArrowParams | None = None
        lstm: LSTMArrowParams | None = None
        gru: GRUArrowParams | None = None
        tcn: TCNArrowParams | None = None
        cnn1d: CNN1DArrowParams | None = None

        if model_type == "transformer":
            if "transformer" in data:
                transformer = TransformerArrowParams.from_dict(data["transformer"])
            elif _TRANSFORMER_FLAT_KEYS.intersection(data):
                flat = {k: data[k] for k in _TRANSFORMER_FLAT_KEYS if k in data}
                transformer = TransformerArrowParams.from_dict(flat)
            else:
                transformer = TransformerArrowParams()
            # Flat keys only when no nested block (nested format takes precedence).
            if "transformer" not in data:
                flat_overlay = {k: data[k] for k in _TRANSFORMER_FLAT_KEYS if k in data}
                if flat_overlay:
                    merged = {**transformer.as_dict(), **flat_overlay}
                    transformer = TransformerArrowParams.from_dict(merged)
        elif model_type == "mlp":
            mlp = (
                MLPArrowParams.from_dict(data["mlp"])
                if "mlp" in data
                else MLPArrowParams()
            )
        elif model_type == "lstm":
            lstm = (
                LSTMArrowParams.from_dict(data["lstm"])
                if "lstm" in data
                else LSTMArrowParams()
            )
        elif model_type == "gru":
            gru = (
                GRUArrowParams.from_dict(data["gru"])
                if "gru" in data
                else GRUArrowParams()
            )
        elif model_type == "tcn":
            tcn = (
                TCNArrowParams.from_dict(data["tcn"])
                if "tcn" in data
                else TCNArrowParams()
            )
        elif model_type == "cnn1d":
            cnn1d = (
                CNN1DArrowParams.from_dict(data["cnn1d"])
                if "cnn1d" in data
                else CNN1DArrowParams()
            )

        return cls(
            model_type=model_type,
            snippet_half_frames=snippet_half_frames,
            use_interval=use_interval,
            interval_encoding=interval_encoding,
            use_step_index=use_step_index,
            use_beat_phase=use_beat_phase,
            transformer=transformer,
            mlp=mlp,
            lstm=lstm,
            gru=gru,
            tcn=tcn,
            cnn1d=cnn1d,
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
        loss_type: Main classification loss: "crossentropy" or "focal". Default "crossentropy".
        focal_gamma: Gamma for focal loss when loss_type is "focal"; ignored otherwise.
        label_smoothing: Label smoothing factor for crossentropy (0 = none). Used when loss_type is "crossentropy".
        aux_interval_weight: Weight for auxiliary next-interval regression loss. 0 disables. Default 0.0.
    """

    chart_validity_aux_weight: float = 0.0
    diversity_aux_weight: float = 0.0
    warmup_epochs: int = 0
    lr_peak: float = 1e-3
    lr_min: float = 1e-5
    loss_type: str = "crossentropy"
    focal_gamma: float = 2.0
    label_smoothing: float = 0.0
    aux_interval_weight: float = 0.0

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
        if self.loss_type not in ("crossentropy", "focal"):
            raise ValueError(
                f"loss_type must be 'crossentropy' or 'focal', got {self.loss_type!r}"
            )
        if self.focal_gamma < 0:
            raise ValueError(f"focal_gamma must be >= 0, got {self.focal_gamma}")
        if not 0 <= self.label_smoothing < 1:
            raise ValueError(
                f"label_smoothing must be in [0, 1), got {self.label_smoothing}"
            )
        if self.aux_interval_weight < 0:
            raise ValueError(
                f"aux_interval_weight must be >= 0, got {self.aux_interval_weight}"
            )

    def get_experiment_name_parts(self) -> list[str]:
        """Return experiment name fragments for run-level options (take, aux weights, loss)."""
        parts: list[str] = []
        if self.take_count == -1:
            parts.append("take_all")
        else:
            parts.append(f"take_{self.take_count}")
        if self.chart_validity_aux_weight > 0:
            parts.append(
                f"chart_val_aux_{str(self.chart_validity_aux_weight).replace('.', '_')}"
            )
        if self.diversity_aux_weight > 0:
            parts.append(
                f"diversity_aux_{str(self.diversity_aux_weight).replace('.', '_')}"
            )
        if self.loss_type == "focal":
            parts.append(f"focal_gamma_{str(self.focal_gamma).replace('.', '_')}")
        if self.label_smoothing > 0:
            parts.append(f"label_smooth_{str(self.label_smoothing).replace('.', '_')}")
        if self.aux_interval_weight > 0:
            parts.append(
                f"aux_interval_{str(self.aux_interval_weight).replace('.', '_')}"
            )
        return parts

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


def validate_arrow_dataset_model_alignment(
    dataset_config: ArrowDatasetConfig,
    model_config: ArrowModelConfig,
) -> None:
    """Ensure dataset and model configs agree on inputs and encoding options.

    Training requires the dataset to produce inputs that match what the model
    expects. snippet_half_frames, use_interval, interval_encoding, use_step_index,
    and use_beat_phase must match.

    Args:
        dataset_config: Dataset configuration.
        model_config: Model configuration.

    Raises:
        ValueError: If any aligned field differs between dataset and model configs.
    """
    if dataset_config.snippet_half_frames != model_config.snippet_half_frames:
        raise ValueError(
            "dataset.snippet_half_frames and model.snippet_half_frames must match "
            f"(got dataset={dataset_config.snippet_half_frames}, "
            f"model={model_config.snippet_half_frames})."
        )
    if dataset_config.use_interval != model_config.use_interval:
        raise ValueError(
            "dataset.use_interval and model.use_interval must match "
            f"(got dataset={dataset_config.use_interval}, model={model_config.use_interval})."
        )
    if dataset_config.use_step_index != model_config.use_step_index:
        raise ValueError(
            "dataset.use_step_index and model.use_step_index must match "
            f"(got dataset={dataset_config.use_step_index}, "
            f"model={model_config.use_step_index})."
        )
    if dataset_config.use_beat_phase != model_config.use_beat_phase:
        raise ValueError(
            "dataset.use_beat_phase and model.use_beat_phase must match "
            f"(got dataset={dataset_config.use_beat_phase}, "
            f"model={model_config.use_beat_phase})."
        )
    valid_encodings = ("default", "log", "multi")
    if dataset_config.interval_encoding not in valid_encodings:
        raise ValueError(
            f"dataset.interval_encoding must be one of {valid_encodings!r}, "
            f"got {dataset_config.interval_encoding!r}."
        )
    if model_config.interval_encoding not in valid_encodings:
        raise ValueError(
            f"model.interval_encoding must be one of {valid_encodings!r}, "
            f"got {model_config.interval_encoding!r}."
        )
    if dataset_config.interval_encoding != model_config.interval_encoding:
        raise ValueError(
            "dataset.interval_encoding and model.interval_encoding must match "
            f"(got dataset={dataset_config.interval_encoding!r}, "
            f"model={model_config.interval_encoding!r})."
        )


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
            ValueError: If dataset and model snippet_half_frames or use_interval differ.
        """
        dataset = ArrowDatasetConfig.from_dict(data["dataset"])
        model = ArrowModelConfig.from_dict(data["model"])
        run = ArrowRunConfig.from_dict(data["run"])
        validate_arrow_dataset_model_alignment(dataset, model)
        return cls(dataset=dataset, model=model, run=run)

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
