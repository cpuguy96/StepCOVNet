"""Configuration dataclasses and JSON I/O for AR onset experiments."""

from __future__ import annotations

import dataclasses
import json
import pathlib

import numpy as np

from stepcovnet import constants
from stepcovnet.onset_ar import targets


def _coerce_json_values(value: object) -> object:
    if isinstance(value, pathlib.Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _coerce_json_values(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_coerce_json_values(item) for item in value]
    return value


class _DictSerializableMixin:
    def as_dict(self) -> dict:
        return _coerce_json_values(dataclasses.asdict(self))  # type: ignore[arg-type]

    @classmethod
    def from_dict(cls, data: dict):
        fields = {field.name for field in dataclasses.fields(cls)}
        return cls(**{key: value for key, value in data.items() if key in fields})


@dataclasses.dataclass
class ArDatasetConfig(_DictSerializableMixin):
    """Dataset settings for AR onset training."""

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
    normalize_mert_features: bool = (
        False  # default raw; see EXP-20260630-03 / NOTE-20260701-01
    )
    dynamic_padding: bool = False
    length_bucket_boundaries: list[int] = dataclasses.field(
        default_factory=lambda: [512, 768, 1024, 1536],
    )
    cache_in_memory: bool = True
    cache_max_samples: int = 64

    @classmethod
    def from_dict(cls, data: dict):
        """Load dataset config, mapping legacy ``cache_overfit_batch`` if present."""
        fields = {field.name for field in dataclasses.fields(cls)}
        payload = dict(data)
        if "cache_in_memory" not in payload and "cache_overfit_batch" in payload:
            payload["cache_in_memory"] = bool(payload["cache_overfit_batch"])
        return cls(**{key: value for key, value in payload.items() if key in fields})


@dataclasses.dataclass
class ArModelConfig(_DictSerializableMixin):
    """Locked v1 stack hyperparameters for ``gate-tide-overfit``."""

    patch_frames: int = 8
    d_model: int = 256
    n_enc_layers: int = 4
    n_dec_layers: int = 4
    token_scheme: str = "delta_bucketed"
    alignment: str = "pointer_residual"
    max_decode_steps: int = constants.MAX_STEPS
    delta_max_dense: int = targets.DEFAULT_DELTA_MAX_DENSE
    n_log_buckets: int = targets.DEFAULT_N_LOG_BUCKETS
    n_first_abs_bins: int = targets.DEFAULT_N_FIRST_ABS_BINS
    num_heads: int = 4
    dropout_rate: float = 0.1
    pointer_head: str = "content"
    # False = Keras keep-valid masks. True only for rebuilding pre-2026-07-16
    # checkpoints that were trained with inverted mask polarity.
    legacy_inverted_attention_masks: bool = False
    density_conditioning: str = "none"
    # Content-pointer: keys from pre-position patch embeddings (not PE-laden memory).
    pointer_keys_pe_free: bool = True
    # Content-pointer: force queries through a dedicated cross-attn over memory.
    pointer_query_from_cross_attn: bool = True
    # Mask pointer logits so patch_idx >= previous onset patch (train + decode).
    monotonic_pointer: bool = True
    # When pe-free keys are on, decoder cross-attn reads content only (no PE
    # residual). Default False: content-only regresses tide overfit (~0.67 vs
    # ~0.94); keep as an R2 A/B probe via config.
    decoder_cross_content_only: bool = False
    # LayerNorm on pointer query/key streams before the Dense projections.
    pointer_qk_layernorm: bool = True
    density_meter_max: int = 32
    density_onset_hz_norm: float = 15.0


@dataclasses.dataclass
class ArRunConfig(_DictSerializableMixin):
    """Training and eval run settings."""

    epochs: int = 300
    overfit_one_song: bool = False
    lambda_time: float = 0.0
    lambda_time_ramp_epochs: int = 0
    lambda_residual: float = 0.0
    lambda_incremental_consistency: float = 0.0
    incremental_consistency_max_steps: int = 0
    token_class_weight: str = "none"
    use_soft_pointer_time: bool = False
    # Hard argmax forward + soft expected backward for λ_time (NOTE-20260806-01).
    use_ste_pointer_time: bool = False
    # Apply λ_time only on steps where hard pointer matches the target patch.
    time_loss_correct_patch_only: bool = False
    # Restrict pointer CE to a local window; 0 = full patch axis.
    pointer_local_ce_radius: int = 0
    # "target": [target - r, target + r] CE-only (decode unrestricted — footgun).
    # "prev": [prev, prev + r] for CE + decode/metrics (decode-consistent).
    pointer_local_ce_anchor: str = "target"
    scheduled_sampling_max_p: float = 0.0
    scheduled_sampling_ramp_epochs: int = 0
    scheduled_sampling_warmup_epochs: int = 0
    pointer_loss_weight: float = 1.0
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


def pointer_decode_max_ahead(run: ArRunConfig) -> int:
    """Max patches ahead of ``prev`` at decode when local CE is prev-anchored."""
    if int(run.pointer_local_ce_radius) <= 0:
        return 0
    if str(run.pointer_local_ce_anchor or "target").lower() != "prev":
        return 0
    return int(run.pointer_local_ce_radius)


@dataclasses.dataclass
class ArExperimentConfig:
    """Complete configuration for an AR onset experiment."""

    dataset: ArDatasetConfig
    model: ArModelConfig
    run: ArRunConfig

    def as_dict(self) -> dict:
        return {
            "dataset": self.dataset.as_dict(),
            "model": self.model.as_dict(),
            "run": self.run.as_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> ArExperimentConfig:
        return cls(
            dataset=ArDatasetConfig.from_dict(data["dataset"]),
            model=ArModelConfig.from_dict(data["model"]),
            run=ArRunConfig.from_dict(data["run"]),
        )

    def to_json(self, path: str) -> None:
        config_path = pathlib.Path(path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with config_path.open("w", encoding="utf-8") as config_file:
            json.dump(self.as_dict(), config_file, indent=2)

    @classmethod
    def from_json(cls, path: str) -> ArExperimentConfig:
        with pathlib.Path(path).open(encoding="utf-8") as config_file:
            data = json.load(config_file)
        return cls.from_dict(data)

    def build_vocab(self) -> targets.DeltaBucketVocab:
        """Return the token vocabulary implied by model settings."""
        return targets.DeltaBucketVocab(
            delta_max_dense=self.model.delta_max_dense,
            n_log_buckets=self.model.n_log_buckets,
            n_first_abs_bins=self.model.n_first_abs_bins,
            hop_sec=self.dataset.hop_sec,
        )

    def max_encoder_patches(self) -> int:
        """Maximum patch count for padded encoder memory."""
        max_frames = max(
            1,
            int(round(self.dataset.max_audio_seconds / self.dataset.hop_sec)),
        )
        patch_frames = max(1, int(self.model.patch_frames))
        return (max_frames + patch_frames - 1) // patch_frames

    def patch_input_dim(self) -> int:
        """Flattened MERT patch feature width ``P * 1024``."""
        return int(self.model.patch_frames) * constants.MERT_HIDDEN_SIZE

    def max_decoder_len(self) -> int:
        """Padded decoder sequence length including ``<EOS>``."""
        return int(self.model.max_decode_steps) + 1


def density_conditioning_active(model_config: ArModelConfig) -> bool:
    """Return whether the model expects a ``density_scalar`` input."""
    mode = str(model_config.density_conditioning).strip().lower()
    return mode not in ("", "none")


POINTER_HEAD_CONTENT = "content"
POINTER_HEAD_INDEX = "index"
POINTER_HEADS = (POINTER_HEAD_CONTENT, POINTER_HEAD_INDEX)


def content_pointer_active(model_config: ArModelConfig) -> bool:
    """Whether patch logits score encoder content instead of absolute indices.

    ``index`` reproduces the pre-2026-08-04 ``Dense(max_patches)`` head, which was
    measured to ignore the audio entirely; it exists only to rebuild old runs.

    Raises:
        ValueError: If ``pointer_head`` is not a recognized mode.
    """
    mode = str(model_config.pointer_head).strip().lower()
    if mode not in POINTER_HEADS:
        msg = f"model.pointer_head must be one of {POINTER_HEADS}, got {mode!r}"
        raise ValueError(msg)
    return mode == POINTER_HEAD_CONTENT


def normalize_density_conditioning_mode(mode: str) -> str:
    """Return the lowercase density mode token."""
    return str(mode).strip().lower()


def compute_density_scalar(
    *,
    n_onsets: int,
    duration_sec: float,
    mode: str,
    meter: int = 0,
    meter_max: int = 32,
    onset_hz_norm: float = 15.0,
) -> float:
    """Map chart metadata to a unit density feature in ``[0, 1]``.

    Modes (``model.density_conditioning``):

    * ``onset_density`` (preferred) — ``clip(n_onsets / duration_sec / onset_hz_norm, 0, 1)``.
      Uses only clipped GT onset count and audio duration; no simfile fields.
    * ``meter`` — ``clip(meter, 0, meter_max) / meter_max`` from ``#METER``.
    * ``none`` — ``0.0`` (input omitted from the model).

    Args:
        n_onsets: Ground-truth onset count after clipping to audio duration.
        duration_sec: Audio duration in seconds.
        mode: ``onset_density``, ``meter``, or ``none``.
        meter: Raw ``#METER`` when ``mode`` is ``meter``.
        meter_max: Denominator for meter normalization.
        onset_hz_norm: Onsets-per-second that maps to feature ``1.0``.

    Returns:
        Deterministic density scalar for decoder conditioning.
    """
    normalized_mode = normalize_density_conditioning_mode(mode)
    if normalized_mode in ("", "none"):
        return 0.0
    if normalized_mode == "meter":
        denom = max(1, int(meter_max))
        return float(np.clip(int(meter), 0, denom)) / float(denom)
    if normalized_mode == "onset_density":
        if duration_sec <= 0.0:
            return 0.0
        hz = float(n_onsets) / float(duration_sec)
        return density_scalar_from_onsets_per_sec(hz, onset_hz_norm=onset_hz_norm)
    raise ValueError(f"unsupported density_conditioning mode: {mode!r}")


def density_scalar_from_onsets_per_sec(
    onsets_per_sec: float,
    *,
    onset_hz_norm: float = 15.0,
) -> float:
    """Map a target onset rate to ``density_scalar``.

    Args:
        onsets_per_sec: Desired onsets per second.
        onset_hz_norm: Rate that maps to feature ``1.0``.
    """
    norm = max(1e-9, float(onset_hz_norm))
    return float(np.clip(float(onsets_per_sec) / norm, 0.0, 1.0))


def target_density_scalar(
    target_onsets: int,
    duration_sec: float,
    model_config: ArModelConfig,
) -> float:
    """Density feature for decode when conditioning on a requested onset count.

    Args:
        target_onsets: Desired number of onsets over ``duration_sec``.
        duration_sec: Song duration in seconds.
        model_config: Supplies ``density_conditioning`` and normalization knobs.

    Returns:
        ``density_scalar`` passed to the decoder at generation time.
    """
    return compute_density_scalar(
        n_onsets=int(target_onsets),
        duration_sec=float(duration_sec),
        mode=model_config.density_conditioning,
        meter=0,
        meter_max=model_config.density_meter_max,
        onset_hz_norm=model_config.density_onset_hz_norm,
    )
