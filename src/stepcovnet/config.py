"""Configuration classes for dataset, model, and training run parameters.

This module provides typed configuration objects for better tracking and
reproducibility of experiments. Configs can be serialized to/from JSON
for saving with runs and loading for re-running experiments.
"""

from __future__ import annotations

import dataclasses
import enum
import json
import pathlib
from typing import Any, get_args, get_origin, get_type_hints

from stepcovnet import constants

# Dense difficulty conditioning modes. ``onset_density`` mirrors the AR track's
# winning choice: a measured onsets-per-second rate rather than the simfile
# difficulty label, which is unreliable (see NOTE-20260803-01).
DENSITY_CONDITIONING_NONE = "none"
DENSITY_CONDITIONING_ONSET = "onset_density"
DENSITY_CONDITIONING_MODES = frozenset(
    {DENSITY_CONDITIONING_NONE, DENSITY_CONDITIONING_ONSET}
)

# Onsets-per-second that maps to a conditioning value of 1.0; matches
# ``onset_ar.config.compute_density_scalar`` so both tracks share a scale.
DENSITY_ONSET_HZ_NORM = 15.0

# Peak-picked event metrics; selecting on these requires the dense event
# validation callback, which runs an extra predict pass over the val split.
EVENT_CHECKPOINT_METRICS = frozenset(
    {
        "val_timing_match",
        "val_dense_event_onset_f1",
        "val_skill_event_f1",
    }
)

# Every validation metric allowed to drive checkpointing and early stopping.
CHECKPOINT_METRICS = EVENT_CHECKPOINT_METRICS | frozenset(
    {
        "val_onset_f1_score",
        "val_pr_auc",
        "val_loss",
    }
)


class IntervalEncoding(enum.StrEnum):
    """How to encode inter-step interval as model input.

    DEFAULT: raw normalized interval (interval_input).
    LOG: log(1+interval) (interval_log_input).
    MULTI: both log and next-interval channels (interval_log_input, interval_next_input).
    """

    DEFAULT = "default"
    LOG = "log"
    MULTI = "multi"


class OnsetArchitecture(enum.StrEnum):
    """Dense frame onset model backbone."""

    UNET_WAVENET = "unet_wavenet"
    TCN = "tcn"
    BILSTM = "bilstm"
    TRANSFORMER = "transformer"


class FeatureSource(enum.StrEnum):
    """Audio feature representation for onset model training and inference.

    MEL: Librosa log-mel spectrogram (default).
    MERT: Precomputed MERT hidden states loaded from ``.mert.npy`` files.
    WAVEFORM: Mono waveform at ``constants.TARGET_SR`` for a learned Conv1D frontend.
    """

    MEL = "mel"
    MERT = "mert"
    WAVEFORM = "waveform"


def _coerce_json_values(value: Any) -> Any:
    """Recursively convert Path and enum values for JSON serialization."""
    if isinstance(value, pathlib.Path):
        return str(value)
    if isinstance(value, enum.Enum):
        return value.value
    if isinstance(value, dict):
        return {key: _coerce_json_values(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_coerce_json_values(item) for item in value]
    return value


class _DictSerializableMixin:
    """Mixin providing default as_dict and from_dict for dataclass configs."""

    def as_dict(self) -> dict:
        """Convert config to dictionary for JSON serialization."""
        return _coerce_json_values(dataclasses.asdict(self))  # type: ignore[arg-type]

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
        feature_source: Feature representation (FeatureSource.MEL, MERT, or WAVEFORM).
        mert_features_dir: Directory containing precomputed ``.mert.npy`` files when
            feature_source is MERT. When empty, features are loaded beside each audio file.
        max_train_songs: Maximum training songs to use (-1 for all songs under data_dir).
        training_index_path: Optional path to ``training_index.json``. When set,
            train/val splits are read from the manifest and ``data_dir`` / ``val_data_dir``
            are optional.
        data_root: Prepared output root for nested MERT paths; inferred from the manifest
            when ``training_index_path`` is set.
        train_window_frames: When > 0, train on fixed-length random crops of this
            many feature frames instead of whole songs. Enables uniform shapes and
            real batching; validation always stays whole-song. Not supported with
            ``feature_source=WAVEFORM``.
        train_windows_per_song: Random crops drawn per song per epoch when
            ``train_window_frames`` > 0.
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
    max_train_songs: int = -1
    training_index_path: str = ""
    data_root: str = ""
    train_window_frames: int = 0
    train_windows_per_song: int = 1
    density_conditioning: str = DENSITY_CONDITIONING_NONE

    def __post_init__(self) -> None:
        """Normalize feature_source from string to enum when loaded from dict/JSON."""
        if isinstance(self.feature_source, str):
            object.__setattr__(
                self, "feature_source", FeatureSource(self.feature_source)
            )
        if self.max_train_songs != -1 and self.max_train_songs < 1:
            raise ValueError(
                "max_train_songs must be -1 (all songs) or at least 1, "
                f"got {self.max_train_songs}"
            )
        if self.train_window_frames < 0:
            raise ValueError(
                f"train_window_frames must be >= 0, got {self.train_window_frames}"
            )
        if self.train_windows_per_song < 1:
            raise ValueError(
                "train_windows_per_song must be >= 1, "
                f"got {self.train_windows_per_song}"
            )
        if (
            self.train_window_frames > 0
            and self.feature_source == FeatureSource.WAVEFORM
        ):
            raise ValueError(
                "train_window_frames requires a frame feature source (mel/mert)"
            )
        if self.density_conditioning not in DENSITY_CONDITIONING_MODES:
            raise ValueError(
                "density_conditioning must be one of "
                f"{sorted(DENSITY_CONDITIONING_MODES)}, "
                f"got {self.density_conditioning!r}"
            )
        if (
            self.density_conditioning != DENSITY_CONDITIONING_NONE
            and self.feature_source == FeatureSource.WAVEFORM
        ):
            raise ValueError(
                "density_conditioning requires a frame feature source (mel/mert)"
            )

    def as_dict(self) -> dict:
        """Convert to dict for JSON; feature_source serialized as string."""
        d = dataclasses.asdict(self)  # type: ignore[arg-type]
        d["feature_source"] = self.feature_source.value
        return _coerce_json_values(d)

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
        return _coerce_json_values(d)

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
    """Configuration for dense frame onset model architecture.

    Attributes:
        onset_architecture: Backbone type (U-Net WaveNet, TCN, BiLSTM, or Transformer).
        initial_filters: Channel width for conv / TCN / transformer projection.
        depth: U-Net encoder depth, or number of stacked BiLSTM layers.
        dilation_rates: Dilation factors for WaveNet/TCN blocks.
        kernel_size: Convolution kernel size.
        dropout_rate: Dropout rate for regularization.
        input_features: Width of the input feature vector per time step. When None,
            defaults to 128 for mel, 1024 for MERT, or ``waveform_frontend_filters``
            for WAVEFORM (see resolve_onset_input_features).
        waveform_frontend_filters: Conv1D channel width after the learned waveform
            frontend when ``feature_source`` is WAVEFORM.
        tcn_blocks: Number of TCN macro-blocks (each runs all ``dilation_rates``).
        recurrent_units: Hidden size per direction for BiLSTM backbones.
        transformer_layers: Number of transformer encoder blocks.
        transformer_heads: Attention heads (``initial_filters`` must be divisible).
    """

    onset_architecture: OnsetArchitecture = OnsetArchitecture.UNET_WAVENET
    initial_filters: int = 16
    depth: int = 2
    dilation_rates: list[int] = dataclasses.field(default_factory=lambda: [1, 2, 4, 8])
    kernel_size: int = 3
    dropout_rate: float = 0.0
    input_features: int | None = None
    waveform_frontend_filters: int = 32
    tcn_blocks: int = 4
    recurrent_units: int = 128
    transformer_layers: int = 2
    transformer_heads: int = 4

    def __post_init__(self) -> None:
        """Validate architecture-specific parameters."""
        if isinstance(self.onset_architecture, str):
            object.__setattr__(
                self,
                "onset_architecture",
                OnsetArchitecture(self.onset_architecture),
            )
        if self.depth < 1:
            raise ValueError(f"depth must be at least 1, got {self.depth}")
        if self.tcn_blocks < 1:
            raise ValueError(f"tcn_blocks must be at least 1, got {self.tcn_blocks}")
        if self.recurrent_units < 1:
            raise ValueError(
                f"recurrent_units must be at least 1, got {self.recurrent_units}"
            )
        if self.transformer_layers < 1:
            raise ValueError(
                f"transformer_layers must be at least 1, got {self.transformer_layers}"
            )
        if self.transformer_heads < 1:
            raise ValueError(
                f"transformer_heads must be at least 1, got {self.transformer_heads}"
            )
        if (
            self.onset_architecture == OnsetArchitecture.TRANSFORMER
            and self.initial_filters % self.transformer_heads != 0
        ):
            raise ValueError(
                "initial_filters must be divisible by transformer_heads for "
                f"transformer onset models, got {self.initial_filters} and "
                f"{self.transformer_heads}"
            )

    def as_dict(self) -> dict:
        """Convert to dict for JSON; onset_architecture serialized as string."""
        d = dataclasses.asdict(self)  # type: ignore[arg-type]
        d["onset_architecture"] = self.onset_architecture.value
        return _coerce_json_values(d)

    @classmethod
    def from_dict(cls, data: dict) -> OnsetModelConfig:
        """Create config from dictionary, coercing architecture enum strings."""
        kwargs = dict(data)
        arch = kwargs.get("onset_architecture", OnsetArchitecture.UNET_WAVENET)
        if isinstance(arch, str):
            kwargs["onset_architecture"] = OnsetArchitecture(arch)
        return cls(**kwargs)


def uses_waveform_model_input(dataset_config: OnsetDatasetConfig) -> bool:
    """Return True when the dense onset model consumes a 1D waveform tensor."""
    return dataset_config.feature_source == FeatureSource.WAVEFORM


def density_conditioning_channels(dataset_config: OnsetDatasetConfig) -> int:
    """Return the number of extra input channels added by density conditioning."""
    return 1 if dataset_config.density_conditioning != DENSITY_CONDITIONING_NONE else 0


def resolve_onset_input_features(
    dataset_config: OnsetDatasetConfig,
    model_config: OnsetModelConfig,
) -> int:
    """Resolve U-Net input feature width from dataset and model config.

    Density conditioning appends one constant channel per frame, so it widens
    the model input by one.

    Args:
        dataset_config: Onset dataset configuration (feature source).
        model_config: Onset model configuration (optional explicit input_features).

    Returns:
        Number of feature channels per time step for the onset model.
    """
    extra = density_conditioning_channels(dataset_config)
    if model_config.input_features is not None:
        return model_config.input_features + extra
    if dataset_config.feature_source == FeatureSource.MERT:
        return constants.MERT_HIDDEN_SIZE + extra
    if dataset_config.feature_source == FeatureSource.WAVEFORM:
        return model_config.waveform_frontend_filters
    return constants.N_MELS + extra


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
        confidence_threshold: Peak-pick confidence threshold for dense event-F1 validation.
        tolerance_sec: Event matching tolerance in seconds for dense event-F1 validation.
        min_onset_distance_ms: Minimum gap between predicted peaks for dense event-F1 validation.
        early_stopping_patience: Stop when the monitored metric stalls for this many epochs;
            0 disables early stopping.
        post_hoc_event_f1_export: When True and callbacks are enabled, select the saved
            checkpoint and confidence threshold that maximize validation peak-pick event F1
            (instead of the frame-F1 monitor's best checkpoint) and export it as the model.
        post_hoc_event_f1_thresholds: Confidence thresholds swept per checkpoint during the
            post-hoc event-F1 selection. Each value must be in [0, 1].
        checkpoint_metric: Validation metric driving checkpoint selection and early
            stopping. Empty uses the default frame metric. The members of
            :data:`EVENT_CHECKPOINT_METRICS` are peak-picked event metrics and enable
            the dense event validation callback that produces them. Prefer
            ``val_skill_event_f1`` on multi-song dense val: it discounts the
            audio-blind floor, which raw F1 does not, and unlike
            ``val_timing_match`` it is not pinned near chance for a partially
            correct model.
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
    confidence_threshold: float = 0.05
    tolerance_sec: float = 0.02
    min_onset_distance_ms: float = 50.0
    early_stopping_patience: int = 25
    checkpoint_metric: str = ""
    post_hoc_event_f1_export: bool = False
    post_hoc_event_f1_thresholds: list[float] = dataclasses.field(
        default_factory=lambda: [
            0.05,
            0.1,
            0.15,
            0.2,
            0.25,
            0.3,
            0.35,
            0.4,
            0.45,
            0.5,
        ]
    )

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
        if self.confidence_threshold < 0.0 or self.confidence_threshold > 1.0:
            raise ValueError(
                "confidence_threshold must be in [0, 1], "
                f"got {self.confidence_threshold}"
            )
        if self.tolerance_sec <= 0.0:
            raise ValueError(
                f"tolerance_sec must be positive, got {self.tolerance_sec}"
            )
        if self.min_onset_distance_ms < 0.0:
            raise ValueError(
                "min_onset_distance_ms must be non-negative, "
                f"got {self.min_onset_distance_ms}"
            )
        if self.early_stopping_patience < 0:
            raise ValueError(
                "early_stopping_patience must be non-negative, "
                f"got {self.early_stopping_patience}"
            )
        if self.checkpoint_metric and self.checkpoint_metric not in CHECKPOINT_METRICS:
            raise ValueError(
                "checkpoint_metric must be one of "
                f"{sorted(CHECKPOINT_METRICS)}, got {self.checkpoint_metric!r}"
            )
        if self.post_hoc_event_f1_export and not self.post_hoc_event_f1_thresholds:
            raise ValueError(
                "post_hoc_event_f1_thresholds must be non-empty when "
                "post_hoc_event_f1_export is enabled"
            )
        for threshold in self.post_hoc_event_f1_thresholds:
            if threshold < 0.0 or threshold > 1.0:
                raise ValueError(
                    "post_hoc_event_f1_thresholds values must be in [0, 1], "
                    f"got {threshold}"
                )


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
        config_path = pathlib.Path(path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with config_path.open("w") as f:
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
        with pathlib.Path(path).open() as f:
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
        config_path = pathlib.Path(path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with config_path.open("w") as f:
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
        with pathlib.Path(path).open() as f:
            data = json.load(f)
        return cls.from_dict(data)
