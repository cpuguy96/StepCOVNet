"""Threshold sweep and feature comparison for bad vs good val songs (EXP-21)."""

import argparse
import json
import os
import pathlib

import librosa
import numpy as np
import tensorflow as tf

from stepcovnet import config
from stepcovnet import datasets
from stepcovnet import dense_overfit_eval
from stepcovnet import pairing
from stepcovnet import ssl_features
from stepcovnet.onset_events import charts
from stepcovnet.onset_events import metrics

WORST_SONGS = (
    "1_2_fanclub",
    "dna",
    "hakurei_shrine_neighbourhood_association_marching_song",
    "intersect_thunderbolt",
    "the_purpose_song",
    "bridge_no_one_passes",
    "strobo_nights_ddrkirbys_summer_night_mix",
)
BEST_SONGS = (
    "totetatetoteta",
    "kuru_kuru_pa",
    "mitsuboshi_-happily_ever_after_remix-",
    "happy_x_2_days_lu-i_remix",
    "rebirth",
)
THRESHOLDS = (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6)
MIN_ONSET_DISTANCE_MS = 50.0
TOLERANCE_SEC = 0.02


def _pair_map(data_dir: str) -> dict[str, tuple[str, str]]:
    out: dict[str, tuple[str, str]] = {}
    for audio_path, chart_path in pairing.list_audio_chart_pairs(data_dir):
        song = pathlib.Path(audio_path).parent.name
        out[song] = (audio_path, chart_path)
    return out


def _eval_at_threshold(
    model: tf.keras.Model,
    audio_path: str,
    chart_path: str,
    experiment: config.OnsetExperimentConfig,
    val_dir: str,
    threshold: float,
) -> dict[str, float]:
    features = datasets.load_onset_features(
        audio_path,
        experiment.dataset.feature_source,
        experiment.dataset.mert_features_dir,
        val_dir,
    )
    features = datasets.normalize_onset_spectrogram(features)
    pred_probs = np.asarray(
        model.predict(np.expand_dims(features, 0), verbose=0)
    ).reshape(-1)
    pred_times, pred_conf = dense_overfit_eval.peak_times_and_confidence(
        pred_probs,
        confidence_threshold=threshold,
        min_onset_distance_ms=MIN_ONSET_DISTANCE_MS,
    )
    times = charts.load_onset_times(chart_path, max_steps=None)
    n_gt = int(times.size)
    n_max = max(n_gt, int(pred_times.size), 1)
    gt_times = np.zeros((1, n_max), dtype=np.float32)
    gt_mask = np.zeros((1, n_max), dtype=np.float32)
    gt_times[0, :n_gt] = times.astype(np.float32)
    gt_mask[0, :n_gt] = 1.0
    n_peaks = int(pred_times.size)
    pred_times_batch = np.zeros((1, n_max), dtype=np.float32)
    pred_conf_batch = np.zeros((1, n_max), dtype=np.float32)
    if n_peaks:
        pred_times_batch[0, :n_peaks] = pred_times
        pred_conf_batch[0, :n_peaks] = pred_conf
    _tp, _fp, _fn = metrics.count_event_onset_errors_numpy(
        pred_times_batch,
        pred_conf_batch,
        gt_times,
        gt_mask,
        TOLERANCE_SEC,
        threshold,
        MIN_ONSET_DISTANCE_MS,
    )
    _p, _r, f1 = metrics.event_onset_f1_numpy(
        pred_times_batch,
        pred_conf_batch,
        gt_times,
        gt_mask,
        TOLERANCE_SEC,
        threshold,
        MIN_ONSET_DISTANCE_MS,
    )
    return {
        "event_f1": float(f1),
        "event_tp": float(_tp),
        "event_fp": float(_fp),
        "event_fn": float(_fn),
        "num_peaks": float(n_peaks),
    }


def _gt_frame_mask(times: np.ndarray, n_frames: int, hop_sec: float) -> np.ndarray:
    mask = np.zeros((n_frames,), dtype=np.float32)
    for t in times:
        idx = int(round(float(t) / hop_sec))
        if 0 <= idx < n_frames:
            mask[idx] = 1.0
    return mask


def _frame_recall_at_threshold(
    pred_probs: np.ndarray, gt_mask: np.ndarray, threshold: float
) -> float:
    tol_frames = max(1, int(round(TOLERANCE_SEC / datasets.HOP_COEFF)))
    pred_binary = pred_probs >= threshold
    window = 2 * tol_frames + 1
    kernel = np.ones((window,), dtype=np.float32)
    gt_windows = np.convolve(gt_mask, kernel, mode="same") > 0
    hit = np.logical_and(gt_mask > 0.5, gt_windows & pred_binary).sum()
    n_gt = int(gt_mask.sum())
    return float(hit / n_gt) if n_gt else 0.0


def _peak_timing_offset_ms(
    pred_times: np.ndarray,
    gt_times: np.ndarray,
    tolerance_sec: float,
) -> dict[str, float]:
    if pred_times.size == 0 or gt_times.size == 0:
        return {
            "matched_frac": 0.0,
            "mean_abs_offset_ms": float("nan"),
            "median_abs_offset_ms": float("nan"),
        }
    used_gt: set[int] = set()
    offsets: list[float] = []
    for pt in pred_times:
        best_j = -1
        best_d = tolerance_sec + 1.0
        for j, gt in enumerate(gt_times):
            if j in used_gt:
                continue
            d = abs(float(pt) - float(gt))
            if d <= tolerance_sec and d < best_d:
                best_d = d
                best_j = j
        if best_j >= 0:
            used_gt.add(best_j)
            offsets.append(best_d * 1000.0)
    matched_frac = len(offsets) / float(pred_times.size)
    if not offsets:
        return {
            "matched_frac": matched_frac,
            "mean_abs_offset_ms": float("nan"),
            "median_abs_offset_ms": float("nan"),
        }
    arr = np.asarray(offsets, dtype=np.float64)
    return {
        "matched_frac": matched_frac,
        "mean_abs_offset_ms": float(arr.mean()),
        "median_abs_offset_ms": float(np.median(arr)),
    }


def _feature_profile(
    model: tf.keras.Model,
    audio_path: str,
    chart_path: str,
    experiment: config.OnsetExperimentConfig,
    val_dir: str,
) -> dict[str, float | int | bool]:
    features = datasets.load_onset_features(
        audio_path,
        experiment.dataset.feature_source,
        experiment.dataset.mert_features_dir,
        val_dir,
    )
    features_norm = datasets.normalize_onset_spectrogram(features)
    pred_probs = np.asarray(
        model.predict(np.expand_dims(features_norm, 0), verbose=0)
    ).reshape(-1)
    times = charts.load_onset_times(chart_path, max_steps=None)
    duration_audio = float(librosa.get_duration(path=audio_path))
    mert_path = ssl_features.mert_npy_path(
        audio_path,
        experiment.dataset.mert_features_dir,
        val_dir,
    )
    n_mert_raw = int(np.load(mert_path).shape[0]) if os.path.isfile(mert_path) else -1
    gt_mask = _gt_frame_mask(times, pred_probs.size, datasets.HOP_COEFF)
    pred_times, _ = dense_overfit_eval.peak_times_and_confidence(
        pred_probs,
        confidence_threshold=0.5,
        min_onset_distance_ms=MIN_ONSET_DISTANCE_MS,
    )
    timing = _peak_timing_offset_ms(pred_times, times, TOLERANCE_SEC)
    pos_rate = float((pred_probs >= 0.5).mean())
    gt_pos_rate = float(gt_mask.mean())
    return {
        "n_frames": int(pred_probs.size),
        "n_mert_raw_frames": n_mert_raw,
        "audio_duration_sec": duration_audio,
        "n_gt_steps": int(times.size),
        "step_density_hz": float(times.size / duration_audio)
        if duration_audio > 0
        else 0.0,
        "feature_dim": int(features.shape[1]),
        "feature_mean_raw": float(features.mean()),
        "feature_std_raw": float(features.std()),
        "feature_mean_norm": float(features_norm.mean()),
        "feature_std_norm": float(features_norm.std()),
        "pred_prob_mean": float(pred_probs.mean()),
        "pred_prob_max": float(pred_probs.max()),
        "pred_prob_p95": float(np.percentile(pred_probs, 95)),
        "pred_prob_at_gt_frames_mean": float(pred_probs[gt_mask > 0.5].mean())
        if gt_mask.any()
        else 0.0,
        "pred_prob_at_non_gt_mean": float(pred_probs[gt_mask <= 0.5].mean())
        if (~(gt_mask > 0.5)).any()
        else 0.0,
        "frame_pos_rate_pred_0_5": pos_rate,
        "frame_pos_rate_gt": gt_pos_rate,
        "frame_recall_at_0_5": _frame_recall_at_threshold(pred_probs, gt_mask, 0.5),
        "frame_recall_at_0_2": _frame_recall_at_threshold(pred_probs, gt_mask, 0.2),
        "chart_exceeds_1024": bool(times.size > 1024),
        "peak_matched_frac_at_0_5": timing["matched_frac"],
        "peak_mean_abs_offset_ms_at_0_5": timing["mean_abs_offset_ms"],
    }


def _threshold_sweep(
    model: tf.keras.Model,
    experiment: config.OnsetExperimentConfig,
    val_pairs: dict[str, tuple[str, str]],
    songs: tuple[str, ...],
) -> dict[str, dict]:
    val_dir = experiment.dataset.val_data_dir
    out: dict[str, dict] = {}
    for song in songs:
        audio_path, chart_path = val_pairs[song]
        curve = []
        best_f1 = -1.0
        best_thr = THRESHOLDS[0]
        for threshold in THRESHOLDS:
            metrics = _eval_at_threshold(
                model, audio_path, chart_path, experiment, val_dir, threshold
            )
            curve.append({"threshold": threshold, **metrics})
            if metrics["event_f1"] > best_f1:
                best_f1 = metrics["event_f1"]
                best_thr = threshold
        out[song] = {
            "curve": curve,
            "best_threshold": best_thr,
            "best_event_f1": best_f1,
            "f1_at_0_5": next(
                row["event_f1"] for row in curve if row["threshold"] == 0.5
            ),
        }
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Investigate bad val onset cases.")
    parser.add_argument(
        "--config",
        default="configs/dense_mert_v2_75train_200ep.json",
    )
    parser.add_argument(
        "--output",
        default="models_wsl/dense_mert_v2_75train_200ep/investigate_bad_val.json",
    )
    args = parser.parse_args()

    experiment = config.OnsetExperimentConfig.from_json(args.config)
    model_dir = experiment.run.model_output_dir
    keras_files = [name for name in os.listdir(model_dir) if name.endswith(".keras")]
    model_path = os.path.join(model_dir, keras_files[0])
    model = tf.keras.models.load_model(model_path, compile=False)

    val_pairs = _pair_map(experiment.dataset.val_data_dir)
    sweep_worst = _threshold_sweep(model, experiment, val_pairs, WORST_SONGS)
    sweep_best = _threshold_sweep(model, experiment, val_pairs, BEST_SONGS)

    profiles: dict[str, dict] = {}
    for song in WORST_SONGS + BEST_SONGS:
        audio_path, chart_path = val_pairs[song]
        profiles[song] = _feature_profile(
            model, audio_path, chart_path, experiment, experiment.dataset.val_data_dir
        )
        profiles[song]["group"] = "worst" if song in WORST_SONGS else "best"

    def group_mean(keys: tuple[str, ...], group: str, field: str) -> float:
        vals = [profiles[s][field] for s in keys if profiles[s]["group"] == group]
        return float(np.mean(vals)) if vals else float("nan")

    compare_fields = (
        "pred_prob_at_gt_frames_mean",
        "pred_prob_at_non_gt_mean",
        "frame_recall_at_0_5",
        "frame_recall_at_0_2",
        "peak_matched_frac_at_0_5",
        "step_density_hz",
        "n_gt_steps",
    )
    group_compare = {
        field: {
            "worst_mean": group_mean(WORST_SONGS + BEST_SONGS, "worst", field),
            "best_mean": group_mean(WORST_SONGS + BEST_SONGS, "best", field),
        }
        for field in compare_fields
    }

    report = {
        "model_path": model_path,
        "threshold_sweep": {
            "thresholds": list(THRESHOLDS),
            "min_onset_distance_ms": MIN_ONSET_DISTANCE_MS,
            "tolerance_sec": TOLERANCE_SEC,
            "worst": sweep_worst,
            "best": sweep_best,
        },
        "feature_profiles": profiles,
        "group_compare": group_compare,
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as out_file:
        json.dump(report, out_file, indent=2)
    print(f"wrote {args.output}")
    print("\n=== threshold sweep best F1 (worst) ===")
    for song in WORST_SONGS:
        row = sweep_worst[song]
        print(
            f"  {song}: best F1={row['best_event_f1']:.3f} @ thr={row['best_threshold']:.2f} "
            f"(F1@0.5={row['f1_at_0_5']:.3f})"
        )
    print("\n=== threshold sweep best F1 (best) ===")
    for song in BEST_SONGS:
        row = sweep_best[song]
        print(
            f"  {song}: best F1={row['best_event_f1']:.3f} @ thr={row['best_threshold']:.2f} "
            f"(F1@0.5={row['f1_at_0_5']:.3f})"
        )
    print("\n=== feature group compare (worst vs best means) ===")
    for field, vals in group_compare.items():
        print(f"  {field}: worst={vals['worst_mean']:.4f} best={vals['best_mean']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
