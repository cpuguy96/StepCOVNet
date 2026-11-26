"""Zero-TF configuration dataclasses and JSON I/O for PyTorch AR onset."""

from __future__ import annotations

import dataclasses
import json
import pathlib
from typing import Any

import numpy as np

from stepcovnet import constants
from stepcovnet.pytorch.onset_ar import targets

POINTER_HEAD_CONTENT = "content"
POINTER_HEAD_INDEX = "index"
POINTER_HEADS = (POINTER_HEAD_CONTENT, POINTER_HEAD_INDEX)

GAP_HEAD_CONTENT = "content"
GAP_HEAD_DENSE = "dense"
GAP_HEADS = (GAP_HEAD_CONTENT, GAP_HEAD_DENSE)

ALIGNMENT_POINTER_RESIDUAL = "pointer_residual"
ALIGNMENT_GAP_RESIDUAL = "gap_residual"
ALIGNMENTS = (ALIGNMENT_POINTER_RESIDUAL, ALIGNMENT_GAP_RESIDUAL)


def _coerce_json_values(value: object) -> object:
    if isinstance(value, pathlib.Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _coerce_json_values(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_coerce_json_values(item) for item in value]
    return value


class _DictSerializableMixin:
    def as_dict(self) -> dict[str, Any]:
        """Serialize dataclass fields to a JSON-compatible dictionary.

        Returns:
            Plain dictionary representation.
        """
        return _coerce_json_values(dataclasses.asdict(self))  # type: ignore[return-value,arg-type]

    @classmethod
    def from_dict(cls, data: dict[str, Any]):
        """Construct an instance from a dictionary, ignoring unknown keys.

        Args:
            data: Input key/value mapping.

        Returns:
            Initialized dataclass instance.
        """
        fields = {field.name for field in dataclasses.fields(cls)}  # type: ignore[arg-type]
        return cls(**{key: value for key, value in data.items() if key in fields})


@dataclasses.dataclass
class ArDatasetConfig(_DictSerializableMixin):
    """Dataset settings for PyTorch AR onset training.

    Attributes:
        data_dir: Training data directory.
        val_data_dir: Validation data directory.
        training_index_path: Path to training index JSON manifest.
        data_root: Root directory for resolving manifest relative paths.
        overfit_audio_path: Audio path for single-song overfit mode.
        overfit_chart_path: Chart path for single-song overfit mode.
        mert_features_dir: Directory containing cached MERT ``.npy`` files.
        batch_size: Mini-batch size.
        max_audio_seconds: Audio truncation horizon in seconds.
        max_steps_per_chart: Maximum allowed onset steps per chart.
        hop_sec: Seconds per audio frame hop.
        truncate_long_audio: Whether to truncate waveforms longer than horizon.
        normalize_mert_features: Whether to z-score MERT features per song.
        dynamic_padding: Whether to pad per batch rather than global max.
        length_bucket_boundaries: Bucket boundaries for sequence grouping.
        cache_in_memory: Whether to cache loaded samples in RAM.
        cache_max_samples: Maximum number of cached samples.
    """

    data_dir: str = ""
    val_data_dir: str = ""
    training_index_path: str = ""
    data_root: str = ""
    overfit_audio_path: str = ""
    overfit_chart_path: str = ""
    mert_features_dir: str = ""
    batch_size: int = 1
    max_audio_seconds: float = 300.0
    max_steps_per_chart: int = constants.MAX_STEPS
    hop_sec: float = constants.HOP_COEFF
    truncate_long_audio: bool = True
    normalize_mert_features: bool = False
    dynamic_padding: bool = False
    length_bucket_boundaries: list[int] = dataclasses.field(
        default_factory=lambda: [512, 768, 1024, 1536]
    )
    cache_in_memory: bool = True
    cache_max_samples: int = 64

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ArDatasetConfig:
        """Load dataset config, mapping legacy ``cache_overfit_batch`` if present.

        Args:
            data: Input dictionary.

        Returns:
            Populated :class:`ArDatasetConfig`.
        """
        fields = {field.name for field in dataclasses.fields(cls)}
        payload = dict(data)
        if "cache_in_memory" not in payload and "cache_overfit_batch" in payload:
            payload["cache_in_memory"] = bool(payload["cache_overfit_batch"])
        return cls(**{key: value for key, value in payload.items() if key in fields})


@dataclasses.dataclass
class ArModelConfig(_DictSerializableMixin):
    """Hyperparameters for the PyTorch AR onset encoder-decoder Transformer.

    Attributes:
        patch_frames: Hop frames grouped per audio patch.
        d_model: Transformer hidden width.
        n_enc_layers: Number of encoder blocks.
        n_dec_layers: Number of causal decoder blocks.
        token_scheme: Tokenization scheme name.
        alignment: Primary alignment head (``pointer_residual`` or ``gap_residual``).
        max_decode_steps: Maximum number of onset steps decoded.
        delta_max_dense: Dense delta vocabulary ceiling in hop frames.
        n_log_buckets: Overflow log delta bucket count.
        n_first_abs_bins: First-onset absolute bin count.
        patch_delta_max_dense: Dense patch-gap ceiling.
        patch_n_log_buckets: Overflow patch-gap log bucket count.
        keep_absolute_pointer_head: Keep absolute pointer head alongside gap head.
        gap_head: Gap scoring head (``content`` or ``dense``).
        num_heads: Multi-head attention head count.
        dropout_rate: Dropout probability.
        pointer_head: Pointer head type (``content`` or ``index``).
        legacy_inverted_attention_masks: Compatibility flag for legacy TF configs.
        density_conditioning: Global density mode (``none``, ``onset_density``, ``meter``).
        pointer_keys_pe_free: Key content pointers on PE-free encoder features.
        pointer_query_from_cross_attn: Project pointer queries via dedicated cross-attn.
        monotonic_pointer: Mask pointer logits before previous onset patch.
        decoder_cross_content_only: Restrict decoder cross-attn to PE-free content.
        pointer_qk_layernorm: Apply LayerNorm before pointer Q/K projections.
        density_meter_max: Meter normalization denominator.
        density_onset_hz_norm: Onsets/sec mapped to density ``1.0``.
        mert_hidden_size: Input MERT frame width.
    """

    patch_frames: int = 8
    d_model: int = 256
    n_enc_layers: int = 4
    n_dec_layers: int = 4
    token_scheme: str = "delta_bucketed"
    alignment: str = ALIGNMENT_POINTER_RESIDUAL
    max_decode_steps: int = constants.MAX_STEPS
    delta_max_dense: int = targets.DEFAULT_DELTA_MAX_DENSE
    n_log_buckets: int = targets.DEFAULT_N_LOG_BUCKETS
    n_first_abs_bins: int = targets.DEFAULT_N_FIRST_ABS_BINS
    patch_delta_max_dense: int = targets.DEFAULT_PATCH_DELTA_MAX_DENSE
    patch_n_log_buckets: int = targets.DEFAULT_PATCH_N_LOG_BUCKETS
    keep_absolute_pointer_head: bool = False
    gap_head: str = GAP_HEAD_CONTENT
    num_heads: int = 4
    dropout_rate: float = 0.1
    pointer_head: str = POINTER_HEAD_CONTENT
    legacy_inverted_attention_masks: bool = False
    density_conditioning: str = "none"
    pointer_keys_pe_free: bool = True
    pointer_query_from_cross_attn: bool = True
    monotonic_pointer: bool = True
    decoder_cross_content_only: bool = False
    pointer_qk_layernorm: bool = True
    density_meter_max: int = 32
    density_onset_hz_norm: float = 15.0
    mert_hidden_size: int = constants.MERT_HIDDEN_SIZE


@dataclasses.dataclass
class ArRunConfig(_DictSerializableMixin):
    """Training and eval run settings for PyTorch AR onset.

    Attributes:
        epochs: Total training epochs.
        overfit_one_song: Single-song overfit mode toggle.
        lambda_time: Weight on absolute onset time L1 loss.
        lambda_time_ramp_epochs: Linear ramp epochs for ``lambda_time``.
        lambda_residual: Weight on within-patch residual MSE loss.
        lambda_incremental_consistency: Weight on incremental consistency loss.
        incremental_consistency_max_steps: Max unroll steps for consistency loss.
        token_class_weight: Token CE class weighting scheme.
        use_soft_pointer_time: Soft expectation patch index for time loss.
        use_ste_pointer_time: Straight-through estimator for time loss.
        time_loss_correct_patch_only: Mask time loss to exact patch matches.
        pointer_local_ce_radius: Local pointer CE window radius.
        pointer_local_ce_anchor: Window anchor (``target`` or ``prev``).
        pointer_soft_distance_alpha: Soft distance penalty slope.
        pointer_soft_distance_alpha_hold_epochs: Hold epochs before annealing alpha.
        pointer_soft_distance_alpha_anneal_epochs: Linear anneal epochs for alpha.
        scheduled_sampling_max_p: Maximum scheduled sampling probability.
        scheduled_sampling_ramp_epochs: Ramp epochs for scheduled sampling.
        scheduled_sampling_warmup_epochs: Warmup epochs before scheduled sampling.
        pointer_loss_weight: Weight on pointer CE loss.
        gap_loss_weight: Weight on gap CE loss.
        eos_token_weight_scale: Multiplier on ``<EOS>`` token weight.
        init_model_path: Optional checkpoint path for warm-start initialization.
        length_normalize_ce: Divide token CE by active token count.
        tolerance_sec: Onset evaluation matching tolerance in seconds.
        min_onset_distance_ms: Minimum inter-onset distance in milliseconds.
        checkpoint_metric: Validation metric name for best checkpointing.
        early_stopping_patience: Early stopping patience epochs (0 disables).
        perfect_overfit_early_stop: Enable early stop on perfect overfit score.
        perfect_overfit_min_score: Threshold for perfect overfit early stop.
        perfect_overfit_patience: Consecutive epochs required for overfit stop.
        model_output_dir: Directory for saved checkpoints and eval JSONs.
        callback_root_dir: Root directory for TensorBoard logs.
        run_label: Human-readable run label.
        seed: Random seed.
        learning_rate: Adam learning rate.
        mixed_precision: Enable automatic mixed precision on CUDA.
        enable_xla: Legacy config field retained for JSON compatibility.
    """

    epochs: int = 300
    overfit_one_song: bool = False
    lambda_time: float = 0.0
    lambda_time_ramp_epochs: int = 0
    lambda_residual: float = 0.0
    lambda_incremental_consistency: float = 0.0
    incremental_consistency_max_steps: int = 0
    token_class_weight: str = "none"
    use_soft_pointer_time: bool = False
    use_ste_pointer_time: bool = False
    time_loss_correct_patch_only: bool = False
    pointer_local_ce_radius: int = 0
    pointer_local_ce_anchor: str = "target"
    pointer_soft_distance_alpha: float = 0.0
    pointer_soft_distance_alpha_hold_epochs: int = 0
    pointer_soft_distance_alpha_anneal_epochs: int = 0
    scheduled_sampling_max_p: float = 0.0
    scheduled_sampling_ramp_epochs: int = 0
    scheduled_sampling_warmup_epochs: int = 0
    pointer_loss_weight: float = 1.0
    gap_loss_weight: float = 1.0
    eos_token_weight_scale: float = 1.0
    init_model_path: str = ""
    length_normalize_ce: bool = True
    tolerance_sec: float = 0.02
    min_onset_distance_ms: float = 50.0
    checkpoint_metric: str = "val_event_onset_f1"
    early_stopping_patience: int = 0
    perfect_overfit_early_stop: bool = False
    perfect_overfit_min_score: float = 0.9999
    perfect_overfit_patience: int = 3
    model_output_dir: str = ""
    callback_root_dir: str = ""
    run_label: str = ""
    seed: int = 42
    learning_rate: float = 2e-3
    mixed_precision: bool = False
    enable_xla: bool = False


@dataclasses.dataclass
class ArExperimentConfig:
    """Complete configuration for a PyTorch AR onset experiment.

    Attributes:
        dataset: Dataset and batching configuration.
        model: Transformer architecture and head configuration.
        run: Optimization, loss weights, and logging settings.
    """

    dataset: ArDatasetConfig
    model: ArModelConfig
    run: ArRunConfig

    def as_dict(self) -> dict[str, Any]:
        """Serialize the experiment configuration to a plain dict.

        Returns:
            Nested JSON-compatible dictionary.
        """
        return {
            "dataset": self.dataset.as_dict(),
            "model": self.model.as_dict(),
            "run": self.run.as_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ArExperimentConfig:
        """Build an experiment config from a parsed JSON dictionary.

        Args:
            data: Dict with ``dataset``, ``model``, and ``run`` sections.

        Returns:
            Initialized :class:`ArExperimentConfig`.
        """
        return cls(
            dataset=ArDatasetConfig.from_dict(data["dataset"]),
            model=ArModelConfig.from_dict(data["model"]),
            run=ArRunConfig.from_dict(data["run"]),
        )

    def to_json(self, path: str | pathlib.Path) -> None:
        """Write configuration as formatted JSON.

        Args:
            path: Destination JSON file path.
        """
        config_path = pathlib.Path(path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(
            json.dumps(self.as_dict(), indent=2) + "\n", encoding="utf-8"
        )

    @classmethod
    def from_json(cls, path: str | pathlib.Path) -> ArExperimentConfig:
        """Load configuration from a JSON file.

        Args:
            path: Source JSON file path.

        Returns:
            Parsed :class:`ArExperimentConfig`.
        """
        data = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(data)

    def build_vocab(self) -> targets.DeltaBucketVocab:
        """Return the token-LM vocabulary implied by model settings.

        Returns:
            Configured :class:`targets.DeltaBucketVocab`.
        """
        return targets.DeltaBucketVocab(
            delta_max_dense=self.model.delta_max_dense,
            n_log_buckets=self.model.n_log_buckets,
            n_first_abs_bins=self.model.n_first_abs_bins,
            hop_sec=self.dataset.hop_sec,
        )

    def build_gap_vocab(self) -> targets.PatchGapVocab:
        """Return the patch-gap vocabulary implied by model settings.

        Returns:
            Configured :class:`targets.PatchGapVocab`.
        """
        return targets.PatchGapVocab(
            delta_max_dense=self.model.patch_delta_max_dense,
            n_log_buckets=self.model.patch_n_log_buckets,
        )

    def max_encoder_patches(self) -> int:
        """Return maximum patch count for padded encoder memory.

        Returns:
            Maximum integer number of patches.
        """
        max_frames = max(
            1, int(round(self.dataset.max_audio_seconds / self.dataset.hop_sec))
        )
        patch_frames = max(1, int(self.model.patch_frames))
        return (max_frames + patch_frames - 1) // patch_frames

    def patch_input_dim(self) -> int:
        """Return flattened MERT patch width ``patch_frames * mert_hidden_size``.

        Returns:
            Flattened patch feature width.
        """
        return int(self.model.patch_frames) * int(self.model.mert_hidden_size)

    def max_decoder_len(self) -> int:
        """Return padded decoder sequence length including ``<EOS>``.

        Returns:
            Maximum decoder token length.
        """
        return int(self.model.max_decode_steps) + 1


def density_conditioning_active(model_config: ArModelConfig) -> bool:
    """Return whether the model expects a ``density_scalar`` input.

    Args:
        model_config: AR model configuration.

    Returns:
        True when density conditioning is enabled.
    """
    return str(model_config.density_conditioning).strip().lower() not in ("", "none")


def gap_alignment_active(model_config: ArModelConfig) -> bool:
    """Return whether the primary alignment head is relative patch-gap CE.

    Args:
        model_config: AR model configuration.

    Returns:
        True when ``alignment == "gap_residual"``.

    Raises:
        ValueError: If ``alignment`` is unrecognized.
    """
    mode = str(model_config.alignment).strip().lower()
    if mode not in ALIGNMENTS:
        raise ValueError(f"model.alignment must be one of {ALIGNMENTS}, got {mode!r}")
    return mode == ALIGNMENT_GAP_RESIDUAL


def absolute_pointer_head_active(model_config: ArModelConfig) -> bool:
    """Return whether the model builds an absolute ``pointer_logits`` head.

    Args:
        model_config: AR model configuration.

    Returns:
        True when the absolute pointer head should be constructed.
    """
    if not gap_alignment_active(model_config):
        return True
    return bool(model_config.keep_absolute_pointer_head)


def content_pointer_active(model_config: ArModelConfig) -> bool:
    """Return whether patch logits score encoder content instead of absolute indices.

    Args:
        model_config: AR model configuration.

    Returns:
        True when ``pointer_head == "content"``.

    Raises:
        ValueError: If ``pointer_head`` is unrecognized.
    """
    mode = str(model_config.pointer_head).strip().lower()
    if mode not in POINTER_HEADS:
        raise ValueError(
            f"model.pointer_head must be one of {POINTER_HEADS}, got {mode!r}"
        )
    return mode == POINTER_HEAD_CONTENT


def content_gap_active(model_config: ArModelConfig) -> bool:
    """Return whether gap logits are content-scored against encoder memory.

    Args:
        model_config: AR model configuration.

    Returns:
        True when ``gap_alignment`` and ``gap_head == "content"``.

    Raises:
        ValueError: If ``gap_head`` is unrecognized.
    """
    if not gap_alignment_active(model_config):
        return False
    mode = str(model_config.gap_head).strip().lower()
    if mode not in GAP_HEADS:
        raise ValueError(f"model.gap_head must be one of {GAP_HEADS}, got {mode!r}")
    return mode == GAP_HEAD_CONTENT


def pointer_decode_max_ahead(run: ArRunConfig) -> int:
    """Return max patches ahead of ``prev`` at decode for prev-anchored local CE.

    Args:
        run: AR run configuration.

    Returns:
        Positive patch window width or ``0`` if disabled.
    """
    if int(run.pointer_local_ce_radius) <= 0:
        return 0
    if str(run.pointer_local_ce_anchor or "target").lower() != "prev":
        return 0
    return int(run.pointer_local_ce_radius)


def compute_density_scalar(
    *,
    n_onsets: int,
    duration_sec: float,
    mode: str,
    meter: int = 0,
    meter_max: int = 32,
    onset_hz_norm: float = 15.0,
) -> float:
    """Map chart metadata to a normalized density scalar in ``[0, 1]``.

    Args:
        n_onsets: Ground-truth onset count.
        duration_sec: Audio duration in seconds.
        mode: ``onset_density``, ``meter``, or ``none``.
        meter: Chart meter integer for ``meter`` mode.
        meter_max: Maximum meter normalization value.
        onset_hz_norm: Onsets/sec that maps to ``1.0``.

    Returns:
        Normalized float in ``[0, 1]``.

    Raises:
        ValueError: If ``mode`` is unsupported.
    """
    normalized = str(mode).strip().lower()
    if normalized in ("", "none"):
        return 0.0
    if normalized == "meter":
        denom = max(1, int(meter_max))
        return float(np.clip(int(meter), 0, denom)) / float(denom)
    if normalized == "onset_density":
        if duration_sec <= 0.0:
            return 0.0
        hz = float(n_onsets) / float(duration_sec)
        return float(np.clip(hz / max(1e-9, float(onset_hz_norm)), 0.0, 1.0))
    raise ValueError(f"unsupported density_conditioning mode: {mode!r}")
