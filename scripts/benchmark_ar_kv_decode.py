"""Benchmark free-running PyTorch AR decode with O(L) KV cache."""

from __future__ import annotations

import argparse
import json
import time

from stepcovnet import wsl_gpu

wsl_gpu.bootstrap_gpu_script("scripts/benchmark_ar_kv_decode.py")

import numpy as np

from stepcovnet.onset_ar import config, datasets, inference, models, trainers
from stepcovnet.pytorch import runtime


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/ar/tide_overfit.json",
        help="AR experiment config JSON.",
    )
    parser.add_argument(
        "--model_path",
        default="models_wsl/ar/tide_overfit/ar_onset_model.pt",
        help="PyTorch checkpoint to decode with.",
    )
    parser.add_argument("--device", default="auto", help="Compute device.")
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Warmup decode runs.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=2,
        help="Timed decode runs.",
    )
    return parser.parse_args()


def _time_decode(
    model: models.ArOnsetModel,
    sample: datasets.ArSample,
    *,
    warmup: int,
    runs: int,
    device,
) -> dict[str, float | int]:
    patch_mask = np.ones((sample.n_patches,), dtype=np.float32)
    for _ in range(warmup):
        inference.decode_autoregressive_kv(
            model, sample.mert_patches, patch_mask, device=device
        )
    durations: list[float] = []
    last_stats = None
    for _ in range(runs):
        t0 = time.perf_counter()
        last_stats = inference.decode_autoregressive_kv(
            model, sample.mert_patches, patch_mask, device=device
        )
        durations.append(time.perf_counter() - t0)
    assert last_stats is not None
    return {
        "mean_sec": float(np.mean(durations)),
        "min_sec": float(np.min(durations)),
        "n_forward_steps": int(last_stats.n_forward_steps),
        "n_onset_tokens": int(last_stats.n_onset_tokens),
    }


def main() -> None:
    args = _parse_args()
    wsl_gpu.guard_pytorch_gpu_job(__file__)
    device = runtime.resolve_device(args.device)
    experiment_config = config.ArExperimentConfig.from_json(args.config)
    sample = datasets.load_split_samples(experiment_config, "train")[0]
    model = trainers.load_ar_onset_model(args.model_path, device=device)
    kv = _time_decode(
        model,
        sample,
        warmup=args.warmup,
        runs=args.runs,
        device=device,
    )
    print(
        json.dumps(
            {
                "model_path": args.model_path,
                "kv_cache": kv,
            },
            indent=2,
        ),
    )


if __name__ == "__main__":
    main()
