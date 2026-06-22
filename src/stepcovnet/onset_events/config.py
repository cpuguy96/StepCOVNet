"""Configuration dataclasses and JSON I/O for event-based onset experiments."""

from __future__ import annotations

import dataclasses
import json
import pathlib

from stepcovnet import constants
from stepcovnet.onset_events import charts, targets
from stepcovnet.onset_events import frontend as audio_frontend


def _coerce_json_values(value: object) -> object:
    """Recursively convert Path values for JSON serialization."""
    if isinstance(value, pathlib.Path):
        return str(value)
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
class OnsetEventEncoderConfig(_DictSerializableMixin):
    """U-Net-style temporal encoder settings for event onset models.

    Attributes:
        initial_filters: Filter count in the first encoder level.
        depth: Number of U-Net down/up levels.
        dilation_rates: Dilation factors within each encoder block.
        kernel_size: Convolution kernel size.
        dropout_rate: Dropout rate for regularization.
    """

    initial_filters: int = 16
    depth: int = 2
    dilation_rates: list[int] = dataclasses.field(default_factory=lambda: [1, 2, 4, 8])
    kernel_size: int = 3
    dropout_rate: float = 0.0


@dataclasses.dataclass
class OnsetEventDatasetConfig(_DictSerializableMixin):
    """Dataset settings for event-based onset training and validation.

    Attributes:
        data_dir: Path to training audio/chart pairs.
        val_data_dir: Path to validation pairs.
        test_data_dir: Optional path for smoke or holdout evaluation.
        batch_size: Samples per batch.
        max_audio_seconds: Maximum waveform duration before truncation.
        n_max_onsets: Fixed length for padded ground-truth onset times.
        max_steps_per_chart: Skip charts with more than this many steps.
        target_sample_rate: Audio sample rate in Hz.
        truncate_long_audio: When True, truncate waveforms to the duration cap.
        apply_audio_augment: When True, apply training-time audio augmentation.
        overfit_audio_path: When set with ``overfit_chart_path``, train and validation
            use only this pair (single-song overfit).
        overfit_chart_path: Chart path paired with ``overfit_audio_path``.
        frontend: Pre-processing type passed to the dataset loader (``conv1d``,
            ``mel``, or ``mert``).
        mert_features_dir: Directory of precomputed ``.mert.npy`` files when
            ``frontend`` is ``mert``.
        data_root: Training data root for nested MERT feature paths.
    """

    data_dir: str = "data/v2/train"
    val_data_dir: str = "data/v2/val"
    test_data_dir: str = "data/v2/test"
    overfit_audio_path: str = ""
    overfit_chart_path: str = ""
    batch_size: int = 1
    max_audio_seconds: float = audio_frontend.MAX_AUDIO_SECONDS
    n_max_onsets: int = targets.N_MAX_ONSETS
    max_steps_per_chart: int = charts.MAX_STEPS_PER_CHART
    target_sample_rate: int = constants.TARGET_SR
    truncate_long_audio: bool = True
    apply_audio_augment: bool = False
    frontend: str = "conv1d"
    mert_features_dir: str = ""
    data_root: str = ""


@dataclasses.dataclass
class OnsetEventModelConfig(_DictSerializableMixin):
    """Model architecture settings for event-based onset detection.

    Attributes:
        frontend: Audio frontend type (``conv1d``, ``mel``, or ``mert``).
        encoder: Temporal encoder hyperparameters.
        num_queries: Fixed query slot count ``K``.
        embed_dim: Query embedding dimension for cross-attention.
        decoder_layers: Number of cross-attention decoder layers.
        target_sample_rate: Sample rate used to size the audio input tensor.
        max_audio_seconds: Fixed input duration cap in seconds.
        frame_hop_sec: Target frontend frame spacing in seconds.
        base_filters: Frontend Conv1D channel count.
        include_duration_input: When True, model takes ``duration`` and outputs
            ``pred_times`` in seconds via ``sigmoid * duration``.
    """

    frontend: str = "conv1d"
    encoder: OnsetEventEncoderConfig = dataclasses.field(
        default_factory=OnsetEventEncoderConfig
    )
    num_queries: int = 1024
    embed_dim: int = 256
    decoder_layers: int = 2
    target_sample_rate: int = constants.TARGET_SR
    max_audio_seconds: float = audio_frontend.MAX_AUDIO_SECONDS
    frame_hop_sec: float = audio_frontend.DEFAULT_FRAME_HOP_SEC
    base_filters: int = audio_frontend.DEFAULT_BASE_FILTERS
    include_duration_input: bool = True

    def as_dict(self) -> dict:
        """Convert to dict with nested encoder serialized."""
        d = dataclasses.asdict(self)
        d["encoder"] = self.encoder.as_dict()
        return d

    @classmethod
    def from_dict(cls, data: dict) -> OnsetEventModelConfig:
        """Create from dict; ``encoder`` may be a nested mapping."""
        kwargs = dict(data)
        encoder_data = kwargs.pop("encoder", {})
        kwargs["encoder"] = OnsetEventEncoderConfig.from_dict(encoder_data)
        return cls(**kwargs)


@dataclasses.dataclass
class OnsetEventRunConfig(_DictSerializableMixin):
    """Training and inference run settings for event onset experiments.

    Attributes:
        epochs: Number of training epochs.
        tolerance_sec: Hungarian matching slack in seconds.
        confidence_threshold: Minimum confidence for inference filtering.
        min_onset_distance_ms: Minimum gap between predicted onsets at inference and
            for the ``event_onset_f1_mingap`` validation metric.
        lambda_cls: Weight for slot classification loss.
        lambda_time: Weight for matched time L1 loss.
        model_output_dir: Directory for saved model checkpoints.
        callback_root_dir: Root directory for TensorBoard and checkpoints.
        seed: Random seed for reproducibility.
        overfit_one_song: When True, train and validation both use the first valid
            pair under ``dataset.data_dir`` (ignored when explicit overfit paths are set).
        pipeline_check_shortcuts: When True, initialize query times from ground
            truth and freeze time deltas (pipeline validation only, not real overfit).
        init_query_refs_from_gt: When True, anchor query slots to sorted chart onsets
            (first ``num_queries`` slots); ignored when ``pipeline_check_shortcuts`` is
            True.
        learn_time_delta: When False, predicted times follow reference anchors only.
    """

    epochs: int = 20
    overfit_one_song: bool = False
    pipeline_check_shortcuts: bool = False
    init_query_refs_from_gt: bool = False
    learn_time_delta: bool = True
    tolerance_sec: float = 0.02
    confidence_threshold: float = 0.5
    min_onset_distance_ms: float = 50.0
    lambda_cls: float = 1.0
    lambda_time: float = 5.0
    model_output_dir: str = ""
    callback_root_dir: str = ""
    seed: int = 42


@dataclasses.dataclass
class OnsetEventExperimentConfig:
    """Complete configuration for an event-based onset experiment.

    Attributes:
        dataset: Dataset paths and loading options.
        model: Model architecture options.
        run: Training and inference run options.
    """

    dataset: OnsetEventDatasetConfig
    model: OnsetEventModelConfig
    run: OnsetEventRunConfig

    def as_dict(self) -> dict:
        """Convert config to dictionary for JSON serialization.

        Returns:
            Nested dict with ``dataset``, ``model``, and ``run`` keys.
        """
        return {
            "dataset": self.dataset.as_dict(),
            "model": self.model.as_dict(),
            "run": self.run.as_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> OnsetEventExperimentConfig:
        """Create config from dictionary.

        Args:
            data: Mapping with ``dataset``, ``model``, and ``run`` sections.

        Returns:
            OnsetEventExperimentConfig instance.

        Raises:
            KeyError: If a required top-level key is missing.
        """
        return cls(
            dataset=OnsetEventDatasetConfig.from_dict(data["dataset"]),
            model=OnsetEventModelConfig.from_dict(data["model"]),
            run=OnsetEventRunConfig.from_dict(data["run"]),
        )

    def to_json(self, path: str) -> None:
        """Save config to a formatted JSON file.

        Args:
            path: Destination file path.
        """
        config_path = pathlib.Path(path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with config_path.open("w", encoding="utf-8") as config_file:
            json.dump(self.as_dict(), config_file, indent=2)

    @classmethod
    def from_json(cls, path: str) -> OnsetEventExperimentConfig:
        """Load config from a JSON file.

        Args:
            path: Path to the JSON config file.

        Returns:
            OnsetEventExperimentConfig loaded from disk.

        Raises:
            FileNotFoundError: If the file does not exist.
            json.JSONDecodeError: If the file is not valid JSON.
            KeyError: If required keys are missing.
        """
        with pathlib.Path(path).open(encoding="utf-8") as config_file:
            data = json.load(config_file)
        return cls.from_dict(data)
