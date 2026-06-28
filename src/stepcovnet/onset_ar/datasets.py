"""Dataset loading for AR onset training (MERT patches + token targets)."""

from __future__ import annotations

import dataclasses
import pathlib

import numpy as np

from stepcovnet import constants, ssl_features
from stepcovnet.onset_ar import config, targets
from stepcovnet.onset_events import (
    audio,
    charts,
)
from stepcovnet.onset_events import (
    datasets as event_datasets,
)
from stepcovnet.onset_events import (
    targets as event_targets,
)


@dataclasses.dataclass(frozen=True)
class ArSample:
    """One AR onset training example after PRE and target encoding."""

    mert_patches: np.ndarray
    n_patches: int
    n_frames: int
    duration_sec: float
    token_seq: targets.OnsetTokenSequence
    gt_times_sec: np.ndarray
    audio_path: str
    chart_path: str


def patch_mert_features(
    features: np.ndarray,
    patch_frames: int,
) -> tuple[np.ndarray, int, int]:
    """Non-overlapping patch of ``(T, D)`` MERT frames into ``(T', P*D)``."""
    if features.ndim != 2:
        raise ValueError(f"features must be 2D; got shape {features.shape!r}")
    patch_frames = max(1, int(patch_frames))
    n_frames = int(features.shape[0])
    n_patches = max(1, (n_frames + patch_frames - 1) // patch_frames)
    padded_frames = n_patches * patch_frames
    if padded_frames > n_frames:
        pad = np.zeros(
            (padded_frames - n_frames, features.shape[1]), dtype=features.dtype
        )
        features = np.concatenate([features, pad], axis=0)
    patches = features.reshape(n_patches, patch_frames * features.shape[1])
    return patches.astype(np.float32), n_patches, n_frames


def _resolve_overfit_pair(
    dataset_config: config.ArDatasetConfig,
    run_config: config.ArRunConfig,
) -> tuple[str, str]:
    if dataset_config.overfit_audio_path and dataset_config.overfit_chart_path:
        return dataset_config.overfit_audio_path, dataset_config.overfit_chart_path
    if run_config.overfit_one_song and dataset_config.data_dir:
        pair = event_datasets.first_valid_pair(dataset_config.data_dir)
        if pair is None:
            raise ValueError(
                f"no valid audio/chart pair under {dataset_config.data_dir!r}"
            )
        audio_path, chart_path, _ = pair
        return audio_path, chart_path
    raise ValueError(
        "overfit paths required: set dataset.overfit_audio_path and "
        "dataset.overfit_chart_path, or run.overfit_one_song with dataset.data_dir",
    )


def load_ar_sample(
    audio_path: str,
    chart_path: str,
    *,
    dataset_config: config.ArDatasetConfig,
    model_config: config.ArModelConfig,
    vocab: targets.DeltaBucketVocab | None = None,
    chart_index: int = 0,
) -> ArSample:
    """Load one audio/chart pair into patched MERT memory and token targets."""
    raw_times = charts.load_onset_times(
        chart_path,
        max_steps=dataset_config.max_steps_per_chart,
        chart_index=chart_index,
    )
    if raw_times is None:
        raise ValueError(f"chart exceeds step cap: {chart_path}")

    max_samples = audio.max_samples_for_cap(dataset_config.max_audio_seconds)
    waveform = audio.load_waveform(audio_path)
    if dataset_config.truncate_long_audio:
        waveform = audio.truncate_waveform(waveform, max_samples)
    duration_sec = float(waveform.size) / float(constants.TARGET_SR)

    mert_raw = ssl_features.load_mert_features(
        audio_path,
        dataset_config.mert_features_dir,
        dataset_config.data_root,
    )
    features = ssl_features.resample_features_to_hop_grid(
        mert_raw,
        duration_sec,
        hop_sec=dataset_config.hop_sec,
    )
    mert_patches, n_patches, n_frames = patch_mert_features(
        features,
        model_config.patch_frames,
    )

    vocab = vocab or targets.DeltaBucketVocab(hop_sec=dataset_config.hop_sec)
    gt_times_sec = event_targets.clip_times_to_duration(raw_times, duration_sec)
    token_seq = targets.encode_onset_times(
        gt_times_sec,
        duration_sec=duration_sec,
        hop_sec=dataset_config.hop_sec,
        patch_frames=model_config.patch_frames,
        vocab=vocab,
        max_steps=dataset_config.max_steps_per_chart,
    )

    return ArSample(
        mert_patches=mert_patches,
        n_patches=n_patches,
        n_frames=n_frames,
        duration_sec=duration_sec,
        token_seq=token_seq,
        gt_times_sec=np.asarray(gt_times_sec, dtype=np.float32),
        audio_path=audio_path,
        chart_path=chart_path,
    )


def verify_tide_assets(
    experiment_config: config.ArExperimentConfig,
) -> dict[str, object]:
    """Return a summary dict after checking tide overfit asset paths exist."""
    dataset_config = experiment_config.dataset
    audio_path = dataset_config.overfit_audio_path
    chart_path = dataset_config.overfit_chart_path
    missing = [
        label
        for label, path in (
            ("audio", audio_path),
            ("chart", chart_path),
        )
        if not pathlib.Path(path).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"missing tide overfit assets: {', '.join(missing)} "
            f"(audio={audio_path!r}, chart={chart_path!r})",
        )

    mert_path = ssl_features.mert_npy_path(
        audio_path,
        dataset_config.mert_features_dir,
        dataset_config.data_root,
    )
    if not pathlib.Path(mert_path).is_file():
        raise FileNotFoundError(
            f"MERT cache missing at {mert_path!r}; extract MERT for tide before gate-tide-overfit",
        )

    return {
        "audio_path": audio_path,
        "chart_path": chart_path,
        "mert_path": mert_path,
        "patch_frames": experiment_config.model.patch_frames,
        "vocab_size": experiment_config.build_vocab().vocab_size,
    }


def load_overfit_sample(experiment_config: config.ArExperimentConfig) -> ArSample:
    """Load the configured single-song overfit example."""
    audio_path, chart_path = _resolve_overfit_pair(
        experiment_config.dataset,
        experiment_config.run,
    )
    return load_ar_sample(
        audio_path,
        chart_path,
        dataset_config=experiment_config.dataset,
        model_config=experiment_config.model,
        vocab=experiment_config.build_vocab(),
    )


def verify_config_loads_one_batch(
    experiment_config: config.ArExperimentConfig,
) -> tuple[dict[str, object], ArSample]:
    """Verify tide assets and load one AR sample (Phase 0 smoke)."""
    summary = verify_tide_assets(experiment_config)
    sample = load_overfit_sample(experiment_config)
    summary.update(
        {
            "duration_sec": sample.duration_sec,
            "n_frames": sample.n_frames,
            "n_patches": sample.n_patches,
            "n_onsets": int(sample.gt_times_sec.size),
            "n_decoder_steps": int(sample.token_seq.decoder_target_ids.size),
            "mert_patch_shape": tuple(sample.mert_patches.shape),
        },
    )
    return summary, sample
