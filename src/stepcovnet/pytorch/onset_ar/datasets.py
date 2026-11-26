"""Pure-NumPy/PyTorch dataset loading and collation for AR onset training."""

from __future__ import annotations

import dataclasses
import pathlib
from typing import Any

import librosa
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from stepcovnet import ssl_features
from stepcovnet.dataset_prep import training_index, training_loader
from stepcovnet.pytorch.onset_ar import config, targets


@dataclasses.dataclass(frozen=True)
class ArSample:
    """One AR onset training/evaluation example after MERT patching and encoding.

    Attributes:
        mert_patches: Patched MERT array ``(n_patches, patch_dim)``.
        n_patches: Number of valid audio patches.
        n_frames: Number of valid hop frames.
        duration_sec: Audio duration in seconds.
        token_seq: Encoded :class:`targets.OnsetTokenSequence`.
        gt_times_sec: Clipped ground-truth onset times in seconds.
        audio_path: Source audio path (or synthetic identifier).
        chart_path: Source chart path (or synthetic identifier).
        meter: Chart difficulty meter integer.
        density_scalar: Normalized density conditioning scalar in ``[0, 1]``.
    """

    mert_patches: np.ndarray
    n_patches: int
    n_frames: int
    duration_sec: float
    token_seq: targets.OnsetTokenSequence
    gt_times_sec: np.ndarray
    audio_path: str
    chart_path: str
    meter: int = 0
    density_scalar: float = 0.0


def patch_mert_features(
    features: np.ndarray,
    patch_frames: int,
) -> tuple[np.ndarray, int, int]:
    """Group ``(T, D)`` MERT frames into non-overlapping ``(T', P * D)`` patches.

    Args:
        features: Frame-aligned MERT features ``(T, D)``.
        patch_frames: Number of frames per patch.

    Returns:
        Tuple ``(patches, n_patches, n_frames)``.

    Raises:
        ValueError: If ``features`` is not 2D.
    """
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


def _normalize_features(features: np.ndarray) -> np.ndarray:
    mean = np.mean(features, axis=0, keepdims=True)
    std = np.std(features, axis=0, keepdims=True)
    return ((features - mean) / np.maximum(std, 1e-6)).astype(np.float32)


def load_ar_sample(
    audio_path: str,
    chart_path: str,
    *,
    dataset_config: config.ArDatasetConfig,
    model_config: config.ArModelConfig,
    vocab: targets.DeltaBucketVocab | None = None,
    gap_vocab: targets.PatchGapVocab | None = None,
    chart_index: int = 0,
) -> ArSample:
    """Load one audio/chart pair into patched MERT memory and AR token targets.

    Args:
        audio_path: Path to audio file.
        chart_path: Path to chart JSON/simfile.
        dataset_config: AR dataset configuration.
        model_config: AR model configuration.
        vocab: Optional prebuilt token vocabulary.
        gap_vocab: Optional prebuilt gap vocabulary.
        chart_index: Chart index within the file.

    Returns:
        Loaded :class:`ArSample`.

    Raises:
        ValueError: If chart exceeds ``max_steps_per_chart``.
    """
    raw_times = np.asarray(
        training_loader.load_chart_times_sec(chart_path, chart_index),
        dtype=np.float64,
    )
    if raw_times.size > dataset_config.max_steps_per_chart:
        raise ValueError(f"chart exceeds step cap: {chart_path}")

    duration_sec = float(librosa.get_duration(path=audio_path))
    if dataset_config.truncate_long_audio:
        duration_sec = min(duration_sec, float(dataset_config.max_audio_seconds))

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
    if dataset_config.normalize_mert_features:
        features = _normalize_features(features)
    mert_patches, n_patches, n_frames = patch_mert_features(
        features, model_config.patch_frames
    )

    vocab = vocab or targets.DeltaBucketVocab(
        delta_max_dense=model_config.delta_max_dense,
        n_log_buckets=model_config.n_log_buckets,
        n_first_abs_bins=model_config.n_first_abs_bins,
        hop_sec=dataset_config.hop_sec,
    )
    gap_vocab = gap_vocab or targets.PatchGapVocab(
        delta_max_dense=model_config.patch_delta_max_dense,
        n_log_buckets=model_config.patch_n_log_buckets,
    )
    gt_times_sec = targets.clip_times_to_duration(raw_times, duration_sec)
    token_seq = targets.encode_onset_times(
        gt_times_sec,
        duration_sec=duration_sec,
        hop_sec=dataset_config.hop_sec,
        patch_frames=model_config.patch_frames,
        vocab=vocab,
        gap_vocab=gap_vocab,
        max_steps=dataset_config.max_steps_per_chart,
    )
    meter = training_loader.load_chart_meter(chart_path, chart_index)
    density_scalar = config.compute_density_scalar(
        n_onsets=int(gt_times_sec.size),
        duration_sec=duration_sec,
        mode=model_config.density_conditioning,
        meter=meter,
        meter_max=model_config.density_meter_max,
        onset_hz_norm=model_config.density_onset_hz_norm,
    )
    return ArSample(
        mert_patches=mert_patches,
        n_patches=n_patches,
        n_frames=n_frames,
        duration_sec=duration_sec,
        token_seq=token_seq,
        gt_times_sec=gt_times_sec.astype(np.float32),
        audio_path=audio_path,
        chart_path=chart_path,
        meter=meter,
        density_scalar=density_scalar,
    )


def load_split_samples(
    experiment_config: config.ArExperimentConfig,
    split: str,
) -> list[ArSample]:
    """Load AR samples for ``train`` or ``val`` from overfit path or manifest.

    Args:
        experiment_config: Full AR experiment config.
        split: ``train`` or ``val``.

    Returns:
        List of :class:`ArSample` objects.

    Raises:
        ValueError: If no samples can be loaded for the requested split.
    """
    ds_cfg = experiment_config.dataset
    vocab = experiment_config.build_vocab()
    gap_vocab = experiment_config.build_gap_vocab()

    if (
        ds_cfg.overfit_audio_path
        and ds_cfg.overfit_chart_path
        and not ds_cfg.training_index_path
    ):
        return [
            load_ar_sample(
                ds_cfg.overfit_audio_path,
                ds_cfg.overfit_chart_path,
                dataset_config=ds_cfg,
                model_config=experiment_config.model,
                vocab=vocab,
                gap_vocab=gap_vocab,
            )
        ]

    if not ds_cfg.training_index_path:
        raise ValueError(
            "Configure either dataset.training_index_path or "
            "dataset.overfit_audio_path + dataset.overfit_chart_path"
        )
    index = training_index.load_training_index(ds_cfg.training_index_path)
    root = pathlib.Path(ds_cfg.data_root) if ds_cfg.data_root else pathlib.Path()
    samples: list[ArSample] = []
    for entry in index.entries:
        if entry.split != split:
            continue
        audio_path = str(root / entry.audio_relpath)
        chart_path = str(root / entry.chart_relpath)
        try:
            samples.append(
                load_ar_sample(
                    audio_path,
                    chart_path,
                    dataset_config=ds_cfg,
                    model_config=experiment_config.model,
                    vocab=vocab,
                    gap_vocab=gap_vocab,
                    chart_index=entry.chart_index,
                )
            )
        except ValueError:
            continue
    if not samples:
        raise ValueError(f"No valid AR onset samples loaded for split={split!r}")
    return samples


def pack_ar_sample(
    sample: ArSample,
    *,
    pad_patches: int,
    pad_dec_len: int,
) -> dict[str, np.ndarray]:
    """Pad one :class:`ArSample` to ``(pad_patches, pad_dec_len)`` arrays.

    Args:
        sample: Unpadded AR sample.
        pad_patches: Target encoder patch length.
        pad_dec_len: Target decoder token length.

    Returns:
        Dictionary of padded NumPy arrays ready for batching.
    """
    patch_dim = int(sample.mert_patches.shape[1])
    n_p = min(sample.n_patches, pad_patches)
    mert = np.zeros((pad_patches, patch_dim), dtype=np.float32)
    mert[:n_p] = sample.mert_patches[:n_p]
    patch_mask = np.zeros((pad_patches,), dtype=np.float32)
    patch_mask[:n_p] = 1.0

    seq = sample.token_seq
    dec_len = min(int(seq.decoder_input_ids.size), pad_dec_len)
    n_onsets = min(seq.n_steps, max(0, pad_dec_len - 1))

    dec_in = np.full((pad_dec_len,), targets.PAD_ID, dtype=np.int64)
    dec_tgt = np.full((pad_dec_len,), targets.PAD_ID, dtype=np.int64)
    dec_mask = np.zeros((pad_dec_len,), dtype=np.float32)
    onset_mask = np.zeros((pad_dec_len,), dtype=np.float32)
    tgt_patches = np.zeros((pad_dec_len,), dtype=np.int64)
    tgt_gaps = np.zeros((pad_dec_len,), dtype=np.int64)
    tgt_res = np.zeros((pad_dec_len,), dtype=np.float32)
    tgt_times = np.zeros((pad_dec_len,), dtype=np.float32)

    dec_in[:dec_len] = seq.decoder_input_ids[:dec_len]
    dec_tgt[:dec_len] = seq.decoder_target_ids[:dec_len]
    dec_mask[:dec_len] = 1.0
    if n_onsets > 0:
        onset_mask[:n_onsets] = 1.0
        tgt_patches[:n_onsets] = np.clip(
            seq.patch_indices[:n_onsets], 0, pad_patches - 1
        )
        tgt_gaps[:n_onsets] = seq.gap_token_ids[:n_onsets]
        tgt_res[:n_onsets] = seq.residual_sec[:n_onsets]
        tgt_times[:n_onsets] = sample.gt_times_sec[:n_onsets]

    prev_patches = np.zeros((pad_dec_len,), dtype=np.int64)
    if pad_dec_len > 1:
        prev_patches[1:] = tgt_patches[:-1]

    return {
        "mert_patches": mert,
        "patch_mask": patch_mask,
        "decoder_input_ids": dec_in,
        "decoder_target_ids": dec_tgt,
        "decoder_mask": dec_mask,
        "onset_step_mask": onset_mask,
        "target_patch_indices": tgt_patches,
        "prev_patch_indices": prev_patches,
        "target_gap_ids": tgt_gaps,
        "target_residual_sec": tgt_res,
        "target_times": tgt_times,
        "density_scalar": np.asarray([sample.density_scalar], dtype=np.float32),
    }


def collate_ar_samples(
    samples: list[ArSample],
    *,
    dynamic_padding: bool = True,
    max_patches: int | None = None,
    max_decoder_len: int | None = None,
) -> dict[str, torch.Tensor]:
    """Collate a list of :class:`ArSample` objects into a PyTorch tensor batch.

    Args:
        samples: Batch of unpadded samples.
        dynamic_padding: Pad to batch maximums instead of fixed global maximums.
        max_patches: Fixed encoder patch cap when ``dynamic_padding=False``.
        max_decoder_len: Fixed decoder length cap when ``dynamic_padding=False``.

    Returns:
        Dictionary of batched PyTorch tensors.
    """
    if dynamic_padding:
        pad_p = max(1, max(s.n_patches for s in samples))
        pad_d = max(1, max(int(s.token_seq.decoder_input_ids.size) for s in samples))
    else:
        pad_p = int(max_patches or max(s.n_patches for s in samples))
        pad_d = int(
            max_decoder_len
            or max(int(s.token_seq.decoder_input_ids.size) for s in samples)
        )
    packed = [pack_ar_sample(s, pad_patches=pad_p, pad_dec_len=pad_d) for s in samples]
    return {
        key: torch.from_numpy(np.stack([item[key] for item in packed], axis=0))
        for key in packed[0]
    }


class ArOnsetDataset(Dataset[ArSample]):
    """Map-style PyTorch dataset over loaded :class:`ArSample` instances.

    Args:
        samples: Sequence of loaded AR samples.
    """

    def __init__(self, samples: list[ArSample]) -> None:
        self._samples = list(samples)

    def __len__(self) -> int:
        """Return number of samples.

        Returns:
            Dataset size.
        """
        return len(self._samples)

    def __getitem__(self, index: int) -> ArSample:
        """Return sample at ``index``.

        Args:
            index: Zero-based sample index.

        Returns:
            The :class:`ArSample` at ``index``.
        """
        return self._samples[index]


def build_dataloader(
    samples: list[ArSample],
    *,
    batch_size: int,
    shuffle: bool,
    dynamic_padding: bool = True,
    max_patches: int | None = None,
    max_decoder_len: int | None = None,
    seed: int = 42,
) -> DataLoader[Any]:
    """Create a PyTorch :class:`DataLoader` with dynamic or fixed AR padding.

    Args:
        samples: List of :class:`ArSample` items.
        batch_size: Mini-batch size.
        shuffle: Whether to shuffle each epoch.
        dynamic_padding: Pad each batch to its max length rather than global max.
        max_patches: Global encoder length cap when ``dynamic_padding=False``.
        max_decoder_len: Global decoder length cap when ``dynamic_padding=False``.
        seed: Generator seed for reproducible shuffling.

    Returns:
        Configured :class:`DataLoader`.
    """
    generator = torch.Generator()
    generator.manual_seed(int(seed))

    def _collate(batch: list[Any]) -> dict[str, torch.Tensor]:
        return collate_ar_samples(
            batch,
            dynamic_padding=dynamic_padding,
            max_patches=max_patches,
            max_decoder_len=max_decoder_len,
        )

    return DataLoader(
        ArOnsetDataset(samples),
        batch_size=max(1, int(batch_size)),
        shuffle=shuffle,
        collate_fn=_collate,
        generator=generator if shuffle else None,
    )
