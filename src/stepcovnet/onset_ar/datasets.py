"""Dataset loading for AR onset training (MERT patches + token targets)."""

from __future__ import annotations

import dataclasses
import logging
import pathlib

import numpy as np
import tensorflow as tf

from stepcovnet import constants, pairing, ssl_features
from stepcovnet.dataset_prep import training_index, training_loader
from stepcovnet.datasets import normalize_onset_spectrogram
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
    meter: int = 0
    density_scalar: float = 0.0


def compute_density_scalar_for_model(
    *,
    meter: int,
    n_onsets: int,
    duration_sec: float,
    model_config: config.ArModelConfig,
) -> float:
    """Return the density feature implied by ``model_config.density_conditioning``."""
    return config.compute_density_scalar(
        n_onsets=n_onsets,
        duration_sec=duration_sec,
        mode=model_config.density_conditioning,
        meter=meter,
        meter_max=model_config.density_meter_max,
        onset_hz_norm=model_config.density_onset_hz_norm,
    )


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
    if dataset_config.normalize_mert_features:
        features = normalize_onset_spectrogram(features)
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
    meter = training_loader.load_chart_meter(chart_path, chart_index)
    density_scalar = compute_density_scalar_for_model(
        meter=meter,
        n_onsets=int(gt_times_sec.size),
        duration_sec=duration_sec,
        model_config=model_config,
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
        meter=meter,
        density_scalar=density_scalar,
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


def _is_single_song_mode(experiment_config: config.ArExperimentConfig) -> bool:
    run_config = experiment_config.run
    dataset_config = experiment_config.dataset
    if run_config.overfit_one_song:
        return True
    return bool(
        dataset_config.overfit_audio_path
        and dataset_config.overfit_chart_path
        and not str(dataset_config.training_index_path).strip(),
    )


def _resolve_ar_data_root(
    dataset_config: config.ArDatasetConfig,
) -> str:
    data_root = str(dataset_config.data_root or dataset_config.data_dir).strip()
    index_ref = str(dataset_config.training_index_path).strip()
    if index_ref:
        index_path = pathlib.Path(index_ref)
        index = training_index.load_training_index(index_path)
        return str(training_index.resolve_output_dir(index, index_path))
    return data_root


def _filter_valid_ar_samples(
    samples: list[tuple[str, str, int]],
    dataset_config: config.ArDatasetConfig,
) -> list[tuple[str, str, int]]:
    """Keep samples with files, chart step cap, and cached MERT features."""
    data_root = _resolve_ar_data_root(dataset_config)
    valid: list[tuple[str, str, int]] = []
    for audio_path, chart_path, chart_index in samples:
        if (
            not pathlib.Path(audio_path).is_file()
            or not pathlib.Path(chart_path).is_file()
        ):
            continue
        if charts.chart_exceeds_step_cap(
            chart_path,
            max_steps=dataset_config.max_steps_per_chart,
            chart_index=chart_index,
        ):
            continue
        if (
            charts.load_onset_times(
                chart_path,
                max_steps=dataset_config.max_steps_per_chart,
                chart_index=chart_index,
            )
            is None
        ):
            continue
        mert_path = ssl_features.mert_npy_path(
            audio_path,
            dataset_config.mert_features_dir,
            data_root,
        )
        if not pathlib.Path(mert_path).is_file():
            continue
        valid.append((audio_path, chart_path, chart_index))
    return valid


def _normalize_training_samples(
    samples: list[tuple[str, str] | tuple[str, str, int]],
) -> list[tuple[str, str, int]]:
    normalized: list[tuple[str, str, int]] = []
    for sample in samples:
        if len(sample) == 2:
            normalized.append((sample[0], sample[1], 0))
        else:
            normalized.append((sample[0], sample[1], sample[2]))
    return normalized


def list_ar_training_samples(
    experiment_config: config.ArExperimentConfig,
    *,
    split: str | None = None,
) -> list[tuple[str, str, int]]:
    """Resolve manifest or directory samples for AR training."""
    dataset_config = experiment_config.dataset
    index_ref = str(dataset_config.training_index_path).strip()
    if index_ref:
        return pairing.list_training_samples(index_ref, split=split)
    data_ref = dataset_config.data_dir or dataset_config.val_data_dir
    if not data_ref:
        raise ValueError(
            "dataset.training_index_path or dataset.data_dir is required "
            "for multi-song AR training",
        )
    return pairing.list_training_samples(data_ref, split=split)


def verify_config_loads_one_batch(
    experiment_config: config.ArExperimentConfig,
) -> tuple[dict[str, object], ArSample]:
    """Verify assets and load one AR sample (tide overfit or manifest smoke)."""
    if _is_single_song_mode(experiment_config):
        summary = verify_tide_assets(experiment_config)
        sample = load_overfit_sample(experiment_config)
        summary.update(_sample_summary_fields(sample))
    else:
        summary = _verify_manifest_summary(experiment_config)
        sample = _load_first_valid_ar_sample(experiment_config)
        summary.update(_sample_summary_fields(sample))
    return summary, sample


def _sample_summary_fields(sample: ArSample) -> dict[str, object]:
    return {
        "duration_sec": sample.duration_sec,
        "n_frames": sample.n_frames,
        "n_patches": sample.n_patches,
        "n_onsets": int(sample.gt_times_sec.size),
        "n_decoder_steps": int(sample.token_seq.decoder_target_ids.size),
        "mert_patch_shape": tuple(sample.mert_patches.shape),
    }


def _load_first_valid_ar_sample(
    experiment_config: config.ArExperimentConfig,
) -> ArSample:
    samples = _filter_valid_ar_samples(
        list_ar_training_samples(
            experiment_config,
            split=training_index.SPLIT_TRAIN,
        ),
        experiment_config.dataset,
    )
    if not samples:
        raise ValueError("No valid AR training samples after filtering.")
    audio_path, chart_path, chart_index = samples[0]
    return load_ar_sample(
        audio_path,
        chart_path,
        dataset_config=experiment_config.dataset,
        model_config=experiment_config.model,
        vocab=experiment_config.build_vocab(),
        chart_index=chart_index,
    )


def _verify_manifest_summary(
    experiment_config: config.ArExperimentConfig,
) -> dict[str, object]:
    dataset_config = experiment_config.dataset
    index_ref = str(dataset_config.training_index_path).strip()
    if not index_ref:
        raise ValueError("dataset.training_index_path is required for manifest verify")
    if not pathlib.Path(index_ref).is_file():
        raise FileNotFoundError(f"training index not found: {index_ref!r}")

    data_root = _resolve_ar_data_root(dataset_config)
    train_samples = _filter_valid_ar_samples(
        list_ar_training_samples(
            experiment_config,
            split=training_index.SPLIT_TRAIN,
        ),
        dataset_config,
    )
    val_samples = _filter_valid_ar_samples(
        list_ar_training_samples(
            experiment_config,
            split=training_index.SPLIT_VAL,
        ),
        dataset_config,
    )
    if not train_samples:
        raise ValueError("No valid train samples in manifest after filtering.")

    train_ds, val_ds, _, _ = create_ar_training_datasets(experiment_config)
    n_train_batches = count_dataset_batches(train_ds)
    n_val_batches = count_dataset_batches(val_ds)
    return {
        "training_index_path": index_ref,
        "data_root": data_root,
        "n_train_samples": len(train_samples),
        "n_val_samples": len(val_samples),
        "n_train_batches": n_train_batches,
        "n_val_batches": n_val_batches,
        "patch_frames": experiment_config.model.patch_frames,
        "vocab_size": experiment_config.build_vocab().vocab_size,
    }


def count_dataset_batches(
    dataset: tf.data.Dataset,
    limit: int = -1,
) -> int:
    """Count batches yielded by a ``tf.data`` pipeline."""
    count = 0
    for _ in dataset:
        count += 1
        if limit > 0 and count >= limit:
            break
    return count


def sample_to_training_arrays(
    sample: ArSample,
    experiment_config: config.ArExperimentConfig,
    *,
    pad_to_configured_max: bool = True,
) -> dict[str, np.ndarray]:
    """Convert one :class:`ArSample` into padded per-sample numpy arrays."""
    model_config = experiment_config.model
    dataset_config = experiment_config.dataset
    patch_dim = experiment_config.patch_input_dim()
    n_patches = int(sample.n_patches)
    dec_len = int(sample.token_seq.decoder_target_ids.size)
    n_gt = int(sample.gt_times_sec.size)
    if pad_to_configured_max:
        max_patches = experiment_config.max_encoder_patches()
        max_dec = experiment_config.max_decoder_len()
        max_gt = int(model_config.max_decode_steps)
    else:
        max_patches = n_patches
        max_dec = dec_len
        max_gt = max(1, n_gt)

    patches = np.zeros((max_patches, patch_dim), dtype=np.float32)
    patch_mask = np.zeros((max_patches,), dtype=np.float32)
    patches[:n_patches] = sample.mert_patches
    patch_mask[:n_patches] = 1.0

    dec_in = sample.token_seq.decoder_input_ids
    dec_tgt = sample.token_seq.decoder_target_ids
    decoder_input_ids = np.zeros((max_dec,), dtype=np.int32)
    decoder_target_ids = np.zeros((max_dec,), dtype=np.int32)
    decoder_mask = np.zeros((max_dec,), dtype=np.float32)
    decoder_input_ids[:dec_len] = dec_in
    decoder_target_ids[:dec_len] = dec_tgt
    decoder_mask[:dec_len] = 1.0

    n_steps = sample.token_seq.n_steps
    target_patch_indices = np.zeros((max_dec,), dtype=np.int32)
    target_residual_sec = np.zeros((max_dec,), dtype=np.float32)
    target_times = np.zeros((max_dec,), dtype=np.float32)
    onset_step_mask = np.zeros((max_dec,), dtype=np.float32)
    if n_steps > 0:
        target_patch_indices[:n_steps] = sample.token_seq.patch_indices
        target_residual_sec[:n_steps] = sample.token_seq.residual_sec
        target_times[:n_steps] = targets.decode_pointer_residual_to_times(
            sample.token_seq.patch_indices,
            sample.token_seq.residual_sec,
            patch_frames=model_config.patch_frames,
            hop_sec=dataset_config.hop_sec,
        )
        onset_step_mask[:n_steps] = 1.0

    gt_times = np.zeros((max_gt,), dtype=np.float32)
    gt_mask = np.zeros((max_gt,), dtype=np.float32)
    gt_times[:n_gt] = sample.gt_times_sec
    gt_mask[:n_gt] = 1.0

    arrays = {
        "mert_patches": patches,
        "patch_mask": patch_mask,
        "decoder_input_ids": decoder_input_ids,
        "decoder_target_ids": decoder_target_ids,
        "decoder_mask": decoder_mask,
        "target_patch_indices": target_patch_indices,
        "target_residual_sec": target_residual_sec,
        "target_times": target_times,
        "onset_step_mask": onset_step_mask,
        "gt_times": gt_times,
        "gt_mask": gt_mask,
        "duration": np.asarray(sample.duration_sec, dtype=np.float32),
    }
    if config.density_conditioning_active(model_config):
        arrays["density_scalar"] = np.asarray(sample.density_scalar, dtype=np.float32)
    return arrays


def sample_to_training_batch(
    sample: ArSample,
    experiment_config: config.ArExperimentConfig,
) -> dict[str, np.ndarray]:
    """Convert one :class:`ArSample` into padded numpy batch arrays."""
    arrays = sample_to_training_arrays(sample, experiment_config)
    batch = {
        key: (
            np.asarray([value], dtype=value.dtype)
            if key in ("duration", "density_scalar")
            else value[np.newaxis, ...]
        )
        for key, value in arrays.items()
    }
    return batch


def _load_ar_sample_py_callback(
    audio_path_t: tf.Tensor,
    chart_path_t: tf.Tensor,
    chart_index_t: tf.Tensor,
    experiment_config: config.ArExperimentConfig,
) -> tuple[np.ndarray, ...]:
    audio_path = audio_path_t.numpy().decode()  # type: ignore[union-attr]
    chart_path = chart_path_t.numpy().decode()  # type: ignore[union-attr]
    chart_index = int(chart_index_t.numpy())  # type: ignore[union-attr]
    sample = load_ar_sample(
        audio_path,
        chart_path,
        dataset_config=experiment_config.dataset,
        model_config=experiment_config.model,
        vocab=experiment_config.build_vocab(),
        chart_index=chart_index,
    )
    arrays = sample_to_training_arrays(
        sample,
        experiment_config,
        pad_to_configured_max=not experiment_config.dataset.dynamic_padding,
    )
    outputs = (
        arrays["mert_patches"],
        arrays["patch_mask"],
        arrays["decoder_input_ids"],
        arrays["decoder_target_ids"],
        arrays["decoder_mask"],
        arrays["target_patch_indices"],
        arrays["target_residual_sec"],
        arrays["target_times"],
        arrays["onset_step_mask"],
        arrays["gt_times"],
        arrays["gt_mask"],
        arrays["duration"],
    )
    if config.density_conditioning_active(experiment_config.model):
        outputs = outputs + (arrays["density_scalar"],)
    return outputs


def _map_ar_sample_to_batch(
    audio_path_t: tf.Tensor,
    chart_path_t: tf.Tensor,
    chart_index_t: tf.Tensor,
    experiment_config: config.ArExperimentConfig,
) -> dict[str, tf.Tensor]:
    dynamic_padding = experiment_config.dataset.dynamic_padding
    max_patches = None if dynamic_padding else experiment_config.max_encoder_patches()
    patch_dim = experiment_config.patch_input_dim()
    max_dec = None if dynamic_padding else experiment_config.max_decoder_len()
    max_gt = None if dynamic_padding else int(experiment_config.model.max_decode_steps)
    use_density = config.density_conditioning_active(experiment_config.model)
    output_types = (
        tf.float32,
        tf.float32,
        tf.int32,
        tf.int32,
        tf.float32,
        tf.int32,
        tf.float32,
        tf.float32,
        tf.float32,
        tf.float32,
        tf.float32,
        tf.float32,
    )
    if use_density:
        output_types = output_types + (tf.float32,)
    mapped = tf.py_function(  # type: ignore[misc]
        lambda ap, cp, ci: _load_ar_sample_py_callback(ap, cp, ci, experiment_config),
        [audio_path_t, chart_path_t, chart_index_t],
        output_types,
    )
    if use_density:
        (
            mert_patches,
            patch_mask,
            decoder_input_ids,
            decoder_target_ids,
            decoder_mask,
            target_patch_indices,
            target_residual_sec,
            target_times,
            onset_step_mask,
            gt_times,
            gt_mask,
            duration,
            density_scalar,
        ) = mapped
    else:
        (
            mert_patches,
            patch_mask,
            decoder_input_ids,
            decoder_target_ids,
            decoder_mask,
            target_patch_indices,
            target_residual_sec,
            target_times,
            onset_step_mask,
            gt_times,
            gt_mask,
            duration,
        ) = mapped
    mert_patches.set_shape([max_patches, patch_dim])
    patch_mask.set_shape([max_patches])
    decoder_input_ids.set_shape([max_dec])
    decoder_target_ids.set_shape([max_dec])
    decoder_mask.set_shape([max_dec])
    target_patch_indices.set_shape([max_dec])
    target_residual_sec.set_shape([max_dec])
    target_times.set_shape([max_dec])
    onset_step_mask.set_shape([max_dec])
    gt_times.set_shape([max_gt])
    gt_mask.set_shape([max_gt])
    duration.set_shape([])
    batch = {
        "mert_patches": mert_patches,
        "patch_mask": patch_mask,
        "decoder_input_ids": decoder_input_ids,
        "decoder_target_ids": decoder_target_ids,
        "decoder_mask": decoder_mask,
        "target_patch_indices": target_patch_indices,
        "target_residual_sec": target_residual_sec,
        "target_times": target_times,
        "onset_step_mask": onset_step_mask,
        "gt_times": gt_times,
        "gt_mask": gt_mask,
        "duration": duration,
    }
    if use_density:
        density_scalar.set_shape([])
        batch["density_scalar"] = density_scalar
    return batch


def create_ar_tf_dataset_from_pairs(
    experiment_config: config.ArExperimentConfig,
    pairs: list[tuple[str, str] | tuple[str, str, int]],
    *,
    shuffle: bool = False,
    seed: int | None = None,
) -> tf.data.Dataset:
    """Build a ``tf.data`` pipeline from explicit audio/chart path pairs."""
    dataset_config = experiment_config.dataset
    samples = _filter_valid_ar_samples(
        _normalize_training_samples(pairs),
        dataset_config,
    )
    if not samples:
        raise ValueError("No valid AR audio-chart pairs found.")

    audio_paths, chart_paths, chart_indices = zip(*samples, strict=True)
    ds = tf.data.Dataset.from_tensor_slices(
        (
            list(audio_paths),
            list(chart_paths),
            list(chart_indices),
        ),
    )

    ds = ds.map(
        lambda audio_path, chart_path, chart_index: _map_ar_sample_to_batch(
            audio_path,
            chart_path,
            chart_index,
            experiment_config,
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    should_cache = dataset_config.cache_in_memory and (
        len(samples) <= int(dataset_config.cache_max_samples)
    )
    if should_cache:
        logging.info(
            "Caching %d AR samples in memory (cache_max_samples=%d)",
            len(samples),
            dataset_config.cache_max_samples,
        )
        ds = ds.cache()
        # Warm the cache once so training epochs do not re-enter py_function.
        for _ in ds:
            pass
    elif dataset_config.cache_in_memory:
        logging.warning(
            "Skipping AR in-memory cache: %d samples > cache_max_samples=%d",
            len(samples),
            dataset_config.cache_max_samples,
        )

    if shuffle and len(samples) > 1:
        ds = ds.shuffle(buffer_size=len(samples), seed=seed)

    if dataset_config.dynamic_padding:
        max_patches = experiment_config.max_encoder_patches()
        max_dec = experiment_config.max_decoder_len()

        def _normalized_sequence_length(batch: dict[str, tf.Tensor]) -> tf.Tensor:
            n_patches = tf.cast(tf.reduce_sum(batch["patch_mask"]), tf.int32)
            n_dec = tf.cast(tf.reduce_sum(batch["decoder_mask"]), tf.int32)
            encoder_length = tf.math.floordiv(
                n_patches * max_dec + max_patches - 1,
                max_patches,
            )
            return tf.maximum(n_dec, encoder_length)

        boundaries = sorted(
            {
                int(boundary)
                for boundary in dataset_config.length_bucket_boundaries
                if 0 < int(boundary) < max_dec
            },
        )
        ds = ds.bucket_by_sequence_length(
            element_length_func=_normalized_sequence_length,
            bucket_boundaries=boundaries,
            bucket_batch_sizes=[dataset_config.batch_size] * (len(boundaries) + 1),
            drop_remainder=False,
        )
    else:
        ds = ds.batch(dataset_config.batch_size, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def _should_cache_ar_samples(experiment_config: config.ArExperimentConfig) -> bool:
    """Return whether single-song overfit should use the in-memory batch helper."""
    return bool(experiment_config.dataset.cache_in_memory)


def create_ar_training_datasets(
    experiment_config: config.ArExperimentConfig,
) -> tuple[tf.data.Dataset, tf.data.Dataset, int, int]:
    """Build train/val datasets and return unbatched sample counts."""
    dataset_config = experiment_config.dataset
    run_config = experiment_config.run

    if _is_single_song_mode(experiment_config):
        audio_path, chart_path = _resolve_overfit_pair(dataset_config, run_config)
        logging.info("Single-song AR overfit mode: %s + %s", audio_path, chart_path)
        if _should_cache_ar_samples(experiment_config):
            logging.info("Caching single-song AR overfit batch in memory")
            overfit_dataset = create_overfit_tf_dataset(experiment_config)
        else:
            overfit_dataset = create_ar_tf_dataset_from_pairs(
                experiment_config,
                [(audio_path, chart_path)],
                shuffle=False,
            )
        return overfit_dataset, overfit_dataset, 1, 1

    index_ref = str(dataset_config.training_index_path).strip()
    train_split = None
    val_split = None

    if index_ref:
        data_root = _resolve_ar_data_root(dataset_config)
        dataset_config.data_root = data_root
        train_split = training_index.SPLIT_TRAIN
        val_split = training_index.SPLIT_VAL
        logging.info(
            "Using AR training index %s (data root %s)",
            index_ref,
            data_root,
        )
    elif training_index.manifest_split_enabled(
        dataset_config.data_dir,
        dataset_config.val_data_dir,
    ):
        train_split = training_index.SPLIT_TRAIN
        val_split = training_index.SPLIT_VAL
        logging.info(
            "Using training_index.json for AR train/val under %s",
            dataset_config.data_dir,
        )

    train_samples = _filter_valid_ar_samples(
        list_ar_training_samples(experiment_config, split=train_split),
        dataset_config,
    )
    val_samples = _filter_valid_ar_samples(
        list_ar_training_samples(experiment_config, split=val_split),
        dataset_config,
    )
    if not train_samples:
        raise ValueError("No valid AR train samples found.")

    train_dataset = create_ar_tf_dataset_from_pairs(
        experiment_config,
        train_samples,
        shuffle=True,
        seed=run_config.seed,
    )
    if val_samples:
        val_dataset = create_ar_tf_dataset_from_pairs(
            experiment_config,
            val_samples,
            shuffle=False,
        )
    else:
        val_dataset = train_dataset
    return train_dataset, val_dataset, len(train_samples), len(val_samples)


def create_overfit_tf_dataset(
    experiment_config: config.ArExperimentConfig,
) -> tf.data.Dataset:
    """Return an in-memory single-batch overfit dataset for ``model.fit``.

    Loads audio/MERT/chart once so each epoch reuses the same tensors instead of
    re-entering ``tf.py_function`` loaders.
    """
    sample = load_overfit_sample(experiment_config)
    arrays = sample_to_training_arrays(
        sample,
        experiment_config,
        pad_to_configured_max=not experiment_config.dataset.dynamic_padding,
    )
    batch = {
        key: (
            np.asarray([value], dtype=value.dtype)
            if key == "duration"
            else value[np.newaxis, ...]
        )
        for key, value in arrays.items()
    }
    return tf.data.Dataset.from_tensors(batch).prefetch(1)
