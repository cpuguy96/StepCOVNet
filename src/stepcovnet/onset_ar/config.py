"""Configuration dataclasses and JSON I/O for AR onset experiments."""

from __future__ import annotations

import dataclasses
import json
import pathlib

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
    cache_overfit_batch: bool = True


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
    legacy_inverted_attention_masks: bool = True


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
    perfect_overfit_early_stop: bool = False
    perfect_overfit_min_score: float = 0.9999
    perfect_overfit_patience: int = 3
    model_output_dir: str = ""
    callback_root_dir: str = ""
    seed: int = 42
    learning_rate: float = 2e-3


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
