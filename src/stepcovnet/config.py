"""Configuration classes for dataset, model, and training run parameters.

This module provides typed configuration objects for better tracking and
reproducibility of experiments. Configs can be serialized to/from JSON
for saving with runs and loading for re-running experiments.
"""

from __future__ import annotations

import dataclasses
import enum
import json
import os
from typing import Any, get_args, get_origin, get_type_hints

from stepcovnet import constants


class IntervalEncoding(enum.StrEnum):
    """How to encode inter-step interval as model input.

    DEFAULT: raw normalized interval (interval_input).
    LOG: log(1+interval) (interval_log_input).
    MULTI: both log and next-interval channels (interval_log_input, interval_next_input).
    """

    DEFAULT = "default"
    LOG = "log"
    MULTI = "multi"


class FeatureSource(enum.StrEnum):
    """Audio feature representation for onset model training and inference.

    MEL: Librosa log-mel spectrogram (default).
    MERT: Precomputed MERT hidden states loaded from ``.mert.npy`` files.
    """

    MEL = "mel"
    MERT = "mert"


class _DictSerializableMixin:
    """Mixin providing default as_dict and from_dict for dataclass configs."""

    def as_dict(self) -> dict:
        """Convert config to dictionary for JSON serialization."""
        return dataclasses.asdict(self)  # type: ignore[arg-type]

    @classmethod
    def from_dict(cls, data: dict):
        """Create config from dictionary.

        Returns:
            Instance of the config class with fields taken from data.
        """
        return cls(**data)


@dataclasses.dataclass
class OnsetDatasetConfig(_DictSerializableMixin):
    """Configuration for onset detection dataset creation.

    Attributes:
        data_dir: Path to training data directory.
        val_data_dir: Path to validation data directory.
        batch_size: Number of samples per batch.
        apply_temporal_augment: Whether to apply temporal augmentation during training.
        should_apply_spec_augment: Whether to apply spectrogram augmentation during training.
        use_gaussian_target: Whether to use Gaussian targets instead of binary targets.
        gaussian_sigma: Standard deviation for Gaussian target distribution.
        feature_source: Feature representation (FeatureSource.MEL or FeatureSource.MERT).
        mert_features_dir: Directory containing precomputed ``.mert.npy`` files when
            feature_source is MERT. When empty, features are loaded beside each audio file.
    """

    data_dir: str
    val_data_dir: str
    batch_size: int = 1
    apply_temporal_augment: bool = False
    should_apply_spec_augment: bool = False
    use_gaussian_target: bool = False
    gaussian_sigma: float = 1.0
    feature_source: FeatureSource = FeatureSource.MEL
    mert_features_dir: str = ""

    def __post_init__(self) -> None:
        """Normalize feature_source from string to enum when loaded from dict/JSON."""
        if isinstance(self.feature_source, str):
            object.__setattr__(
                self, "feature_source", FeatureSource(self.feature_source)
            )

    def as_dict(self) -> dict:
        """Convert to dict for JSON; feature_source serialized as string."""
        d = dataclasses.asdict(self)  # type: ignore[arg-type]
        d["feature_source"] = self.feature_source.value
        return d

    @classmethod
    def from_dict(cls, data: dict) -> OnsetDatasetConfig:
        """Create from dict; feature_source accepted as string or enum."""
        kwargs = dict(data)
        kwargs["feature_source"] = FeatureSource(
            kwargs.get("feature_source", FeatureSource.MEL)
        )
        return cls(**kwargs)


@dataclasses.dataclass
class ArrowDatasetConfig(_DictSerializableMixin):
    """Configuration for arrow classification dataset creation.

    Attributes:
        data_dir: Path to training data directory.
        val_data_dir: Path to validation data directory.
        batch_size: Number of samples per batch.
        snippet_half_frames: Half-window of frames around each onset (total frames = 2*snippet_half_frames+1).
            When > 0, audio snippets are loaded and included per step; when 0, only timing and chart are used.
        use_interval: If True, include inter-step interval (time since previous step) as an input.
        interval_encoding: How to encode interval (IntervalEncoding): DEFAULT, LOG (log(1+interval)), or MULTI (extra channels).
            These input options are the single source of truth; they are applied to the
            model config when loading an experiment so dataset and model stay in sync.
        use_step_index: If True, include step index (position in sequence) as an input.
        use_beat_phase: If True, include beat/phase features (BPM from chart txt).
        use_aux_interval_target: If True, include aux_interval_target (next-step interval) in batch for auxiliary loss.
        timing_jitter_sigma: If > 0, add Gaussian jitter to timing_input during training only;
            magnitude in [0, 1] (e.g. 0.01). 0 disables jitter.
    """

    data_dir: str
    val_data_dir: str
    batch_size: int = 1
    snippet_half_frames: int = 0
    use_interval: bool = False
    interval_encoding: IntervalEncoding = IntervalEncoding.DEFAULT
    use_step_index: bool = False
    use_beat_phase: bool = False
    use_aux_interval_target: bool = False
    timing_jitter_sigma: float = 0.0

    def __post_init__(self) -> None:
        """Normalize interval_encoding from string to enum when loaded from dict/JSON."""
        if isinstance(self.interval_encoding, str):
            object.__setattr__(
                self, "interval_encoding", IntervalEncoding(self.interval_encoding)
            )

    def as_dict(self) -> dict:
        """Convert to dict for JSON; interval_encoding serialized as string."""
        d = dataclasses.asdict(self)  # type: ignore[arg-type]
        d["interval_encoding"] = self.interval_encoding.value
        return d

    def get_experiment_name_parts(self) -> list[str]:
        """Return experiment name fragments for dataset-level options (e.g. timing jitter)."""
        parts: list[str] = []
        if self.timing_jitter_sigma > 0:
            parts.append(
                f"timing_jitter_{str(self.timing_jitter_sigma).replace('.', '_')}"
            )
        if self.snippet_half_frames > 0:
            parts.append(f"snippets_{self.snippet_half_frames}")
        if self.use_step_index:
            parts.append("step_index")
        if self.use_interval:
            parts.append(f"interval_{self.interval_encoding.value}")
        if self.use_beat_phase:
            parts.append("beat_phase")
        return parts

    @classmethod
    def from_dict(cls, data: dict) -> ArrowDatasetConfig:
        """Create from dict; interval_encoding accepted as string or enum."""
        kwargs = dict(data)
        kwargs["interval_encoding"] = IntervalEncoding(
            kwargs.get("interval_encoding", IntervalEncoding.DEFAULT)
        )
        return cls(**kwargs)


@dataclasses.dataclass
class OnsetModelConfig(_DictSerializableMixin):
    """Configuration for U-Net WaveNet model architecture.

    Attributes:
        initial_filters: Number of filters in the first layer (doubles at each level).
        depth: Number of downsampling/upsampling levels in the U-Net.
        dilation_rates: List of dilation factors for convolutions within each level.
        kernel_size: Size of convolutional kernels.
        dropout_rate: Dropout rate for regularization.
        input_features: Width of the input feature vector per time step. When None,
            defaults to 128 for mel features or 1024 for MERT (see resolve_onset_input_features).
    """

    initial_filters: int = 16
    depth: int = 2
    dilation_rates: list[int] = dataclasses.field(default_factory=lambda: [1, 2, 4, 8])
    kernel_size: int = 3
    dropout_rate: float = 0.0
    input_features: int | None = None


def resolve_onset_input_features(
    dataset_config: OnsetDatasetConfig,
    model_config: OnsetModelConfig,
) -> int:
    """Resolve U-Net input feature width from dataset and model config.

    Args:
        dataset_config: Onset dataset configuration (feature source).
        model_config: Onset model configuration (optional explicit input_features).

    Returns:
        Number of feature channels per time step for the onset model.
    """
    if model_config.input_features is not None:
        return model_config.input_features
    if dataset_config.feature_source == FeatureSource.MERT:
        return constants.MERT_HIDDEN_SIZE
    return constants.N_MELS


class ArrowParamsBase(_DictSerializableMixin):
    """Base for arrow model params. All arrow param classes must implement this.

    Implementors must provide experiment_name_parts().
    """

    def experiment_name_parts(self) -> list[str]: ...


@dataclasses.dataclass
class TransformerArrowParams(ArrowParamsBase):
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


@dataclasses.dataclass
class MLPArrowParams(ArrowParamsBase):
    """Parameters for the MLP-based arrow model. Used when model_type is 'mlp'.

    Attributes:
        hidden_dims: List of hidden layer dimensions (e.g. [256, 128]).
        dropout_rate: Dropout rate between dense layers.
    """

    hidden_dims: list[int] = dataclasses.field(default_factory=lambda: [256, 128])
    dropout_rate: float = 0.0

    def experiment_name_parts(self) -> list[str]:
        return [
            "mlp_" + "_".join(str(d) for d in self.hidden_dims),
            f"dropout_{str(self.dropout_rate).replace('.', '_')}",
        ]


@dataclasses.dataclass
class LSTMArrowParams(ArrowParamsBase):
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

    def experiment_name_parts(self) -> list[str]:
        parts = [
            f"lstm_units_{self.units}",
            f"lstm_layers_{self.num_layers}",
            f"dropout_{str(self.dropout_rate).replace('.', '_')}",
        ]
        if self.bidirectional:
            parts.append("lstm_bidir")
        return parts


@dataclasses.dataclass
class GRUArrowParams(ArrowParamsBase):
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


@dataclasses.dataclass
class TCNArrowParams(ArrowParamsBase):
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

    def experiment_name_parts(self) -> list[str]:
        return [
            f"tcn_filters_{self.filters}",
            f"tcn_kernel_{self.kernel_size}",
            f"tcn_layers_{self.num_layers}",
            f"tcn_dilation_base_{self.dilation_base}",
            f"dropout_{str(self.dropout_rate).replace('.', '_')}",
        ]


@dataclasses.dataclass
class CNN1DArrowParams(ArrowParamsBase):
    """Parameters for the 1D CNN arrow model. Used when model_type is 'cnn1d'.

    Attributes:
        filters: Number of convolutional filters per layer.
        kernel_sizes: List of kernel sizes per layer (e.g. [3, 3, 3]).
        dropout_rate: Dropout rate applied after conv stack.
    """

    filters: int = 64
    kernel_sizes: list[int] = dataclasses.field(default_factory=lambda: [3, 3, 3])
    dropout_rate: float = 0.0

    def experiment_name_parts(self) -> list[str]:
        return [
            f"cnn1d_filters_{self.filters}",
            "cnn1d_kernels_" + "_".join(str(k) for k in self.kernel_sizes),
            f"dropout_{str(self.dropout_rate).replace('.', '_')}",
        ]


@dataclasses.dataclass
class ArrowModelConfig(_DictSerializableMixin):
    """Configuration for arrow classification model architecture.

    Supports multiple model types via nested architecture-specific params.
    Only the block matching model_type is required when building; others can be None.

    Attributes:
        model_type: One of 'transformer', 'mlp', 'lstm', 'gru', 'tcn', 'cnn1d'.
        transformer: Params for transformer model; used when model_type is 'transformer'.
        mlp: Params for MLP model; used when model_type is 'mlp'.
        lstm: Params for LSTM model; used when model_type is 'lstm'.
        gru: Params for GRU model; used when model_type is 'gru'.
        tcn: Params for TCN model; used when model_type is 'tcn'.
        cnn1d: Params for 1D CNN model; used when model_type is 'cnn1d'.
    """

    model_type: str = "transformer"
    transformer: TransformerArrowParams | None = None
    mlp: MLPArrowParams | None = None
    lstm: LSTMArrowParams | None = None
    gru: GRUArrowParams | None = None
    tcn: TCNArrowParams | None = None
    cnn1d: CNN1DArrowParams | None = None

    def get_active_params_block(self) -> ArrowParamsBase | None:
        """Return the params block for the current model_type, or None if not set."""
        return getattr(self, self.model_type, None)  # type: ignore[return-value]

    def get_experiment_name_parts(self) -> list[str]:
        """Return experiment name fragments from the active params block."""
        block = self.get_active_params_block()
        parts = block.experiment_name_parts() if block is not None else []
        return parts

    @classmethod
    def from_dict(cls, data: dict) -> ArrowModelConfig:
        """Create config from dictionary. Only model_type and the active param block key are accepted.

        Args:
            data: Dictionary with 'model_type' and one of the param block keys
                ('transformer', 'mlp', 'lstm', 'gru', 'tcn', 'cnn1d') with nested params.

        Returns:
            ArrowModelConfig instance.

        Raises:
            ValueError: If invalid keys are present or model_type is unknown.
        """
        param_blocks = _arrow_model_param_blocks(cls)
        allowed_keys = {"model_type"} | set(param_blocks.keys())
        invalid = set(data.keys()) - allowed_keys
        if invalid:
            raise ValueError(f"Invalid keys for ArrowModelConfig: {sorted(invalid)}")

        model_type: str = data.get("model_type", "transformer")
        if model_type not in param_blocks:
            raise ValueError(f"Invalid model_type: {model_type}")

        kwargs: dict[str, Any] = {"model_type": model_type}
        for param_name, param_class in param_blocks.items():
            if param_name == model_type:
                raw = data.get(param_name)
                block_data = raw if isinstance(raw, dict) else {}
                kwargs[param_name] = param_class.from_dict(block_data)
            else:
                kwargs[param_name] = None
        return cls(**kwargs)


def _arrow_model_param_blocks(
    config_cls: type[ArrowModelConfig],
) -> dict[str, type[ArrowParamsBase]]:
    """Build model_type -> param class map from config dataclass fields.

    Introspects param fields (those typed as Optional[ArrowParamsBase])
    so adding a new param block only requires adding the field to the config class.
    Resolves annotations via get_type_hints for PEP 563 (forward refs).

    Returns:
        Ordered mapping from param field name (e.g. 'transformer') to param class.
    """
    result: dict[str, type[ArrowParamsBase]] = {}
    try:
        hints = get_type_hints(config_cls)
    except Exception:
        hints = {}
    for f in dataclasses.fields(config_cls):
        if f.name == "model_type":
            continue
        ann = hints.get(f.name, f.type)
        if get_origin(ann) is not None:
            args = get_args(ann)
            for a in args:
                if a is type(None):
                    continue
                if isinstance(a, type) and issubclass(a, ArrowParamsBase):
                    result[f.name] = a
                    break
        elif isinstance(ann, type) and issubclass(ann, ArrowParamsBase):
            result[f.name] = ann
    return result


@dataclasses.dataclass
class RunConfig(_DictSerializableMixin):
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
        chart_validity_rejection_threshold: Minimum validity (0, 1] to consider a batch valid; None disables tiered loss.
        chart_validity_rejection_scale: Multiplier for rejection penalty when below threshold. Used only when threshold is set.
        chart_validity_rejection_temperature: Sigmoid temperature for tiered-loss gate; larger = sharper transition.
    """

    chart_validity_aux_weight: float = 0.0
    diversity_aux_weight: float = 0.0
    chart_validity_rejection_threshold: float | None = None
    chart_validity_rejection_scale: float = 10.0
    chart_validity_rejection_temperature: float = 50.0
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
        if self.chart_validity_rejection_threshold is not None:
            if not (0.0 < self.chart_validity_rejection_threshold <= 1.0):
                raise ValueError(
                    "chart_validity_rejection_threshold must be in (0, 1] when set, "
                    f"got {self.chart_validity_rejection_threshold}"
                )
            if self.chart_validity_rejection_scale <= 0:
                raise ValueError(
                    f"chart_validity_rejection_scale must be > 0 when threshold is set, "
                    f"got {self.chart_validity_rejection_scale}"
                )
            if self.chart_validity_rejection_temperature <= 0:
                raise ValueError(
                    f"chart_validity_rejection_temperature must be > 0 when threshold is set, "
                    f"got {self.chart_validity_rejection_temperature}"
                )

    def get_experiment_name_parts(self) -> list[str]:
        """Return experiment name fragments for run-level options (take, warmup, aux weights, loss)."""
        parts: list[str] = []
        if self.take_count == -1:
            parts.append("take_all")
        else:
            parts.append(f"take_{self.take_count}")
        if self.warmup_epochs > 0:
            parts.append(f"warmup_epochs_{self.warmup_epochs}")
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
        if self.chart_validity_rejection_threshold is not None:
            parts.append(
                f"chart_val_rej_{str(self.chart_validity_rejection_threshold).replace('.', '_')}"
            )
            parts.append(
                f"chart_val_rej_scale_{str(self.chart_validity_rejection_scale).replace('.', '_')}"
            )
            parts.append(
                f"chart_val_rej_temp_{str(self.chart_validity_rejection_temperature).replace('.', '_')}"
            )
        return parts


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
            Dictionary with 'dataset', 'model', and 'run' keys containing
            the serialized nested configurations.
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
