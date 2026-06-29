"""Benchmark free-running AR decode (prefix vs KV cache) on tide overfit batch."""

from __future__ import annotations

import argparse
import json
import time

import keras
import numpy as np

from stepcovnet.onset_ar import config, datasets, inference


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/ar/decode/v2.json",
        help="AR experiment config JSON.",
    )
    parser.add_argument(
        "--model_path",
        default="models_wsl/ar/gate_tide_overfit/ar_onset_model.keras",
        help="Checkpoint to decode with.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Warmup decode runs per path.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=2,
        help="Timed decode runs per path.",
    )
    return parser.parse_args()


def _time_decode(
    model: keras.Model,
    mert_patches: np.ndarray,
    patch_mask: np.ndarray,
    *,
    experiment_config: config.ArExperimentConfig,
    use_kv_cache: bool,
    warmup: int,
    runs: int,
) -> dict[str, float | int]:
    kwargs = {
        "max_decoder_len": experiment_config.max_decoder_len(),
        "patch_frames": experiment_config.model.patch_frames,
        "hop_sec": experiment_config.dataset.hop_sec,
        "experiment_config": experiment_config,
    }
    for _ in range(warmup):
        inference.decode_autoregressive_with_stats_numpy(
            model,
            mert_patches,
            patch_mask,
            use_kv_cache=use_kv_cache,
            **kwargs,
        )
    durations: list[float] = []
    last_stats = None
    for _ in range(runs):
        t0 = time.perf_counter()
        last_stats = inference.decode_autoregressive_with_stats_numpy(
            model,
            mert_patches,
            patch_mask,
            use_kv_cache=use_kv_cache,
            **kwargs,
        )
        durations.append(time.perf_counter() - t0)
    assert last_stats is not None
    return {
        "use_kv_cache": use_kv_cache,
        "mean_sec": float(np.mean(durations)),
        "min_sec": float(np.min(durations)),
        "n_forward_steps": int(last_stats.n_forward_steps),
        "n_onset_tokens": int(last_stats.n_onset_tokens),
    }


def main() -> None:
    args = _parse_args()
    experiment_config = config.ArExperimentConfig.from_json(args.config)
    sample = datasets.load_overfit_sample(experiment_config)
    batch = datasets.sample_to_training_batch(sample, experiment_config)
    mert_patches = np.asarray(batch["mert_patches"], dtype=np.float32)
    patch_mask = np.asarray(batch["patch_mask"], dtype=np.float32)
    model = keras.models.load_model(args.model_path, compile=False)

    prefix = _time_decode(
        model,
        mert_patches,
        patch_mask,
        experiment_config=experiment_config,
        use_kv_cache=False,
        warmup=args.warmup,
        runs=args.runs,
    )
    kv = _time_decode(
        model,
        mert_patches,
        patch_mask,
        experiment_config=experiment_config,
        use_kv_cache=True,
        warmup=args.warmup,
        runs=args.runs,
    )
    speedup = prefix["mean_sec"] / kv["mean_sec"] if kv["mean_sec"] > 0 else 0.0
    print(
        json.dumps(
            {
                "model_path": args.model_path,
                "prefix": prefix,
                "kv_cache": kv,
                "speedup_vs_prefix": speedup,
            },
            indent=2,
        ),
    )


if __name__ == "__main__":
    main()
