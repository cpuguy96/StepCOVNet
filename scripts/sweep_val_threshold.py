"""Global and per-song oracle threshold sweep on full val set (EXP-21)."""

import argparse
import json
import pathlib

import librosa
import numpy as np
import tensorflow as tf

from stepcovnet import config, datasets, dense_overfit_eval, pairing, ssl_features
from stepcovnet.onset_events import charts, metrics

THRESHOLDS = (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6)
MIN_ONSET_DISTANCE_MS = 50.0
TOLERANCE_SEC = 0.02
BOTTOM_3_SONGS = (
    "1_2_fanclub",
    "dna",
    "hakurei_shrine_neighbourhood_association_marching_song",
)


def _pair_map(data_dir: str) -> dict[str, tuple[str, str]]:
    out: dict[str, tuple[str, str]] = {}
    for audio_path, chart_path in pairing.list_audio_chart_pairs(data_dir):
        song = pathlib.Path(audio_path).parent.name
        out[song] = (audio_path, chart_path)
    return out


def _chart_bpm(chart_path: str) -> float:
    with pathlib.Path(chart_path).open(encoding="utf-8") as chart_file:
        chart_file.readline()
        bpm_line = chart_file.readline()
    return float(bpm_line.removeprefix("BPM").strip())


def _micro_f1(tp: float, fp: float, fn: float) -> float:
    denom = 2.0 * tp + fp + fn
    return float(2.0 * tp / denom) if denom > 0 else 0.0


def _predict_probs(
    model: tf.keras.Model,
    audio_path: str,
    experiment: config.OnsetExperimentConfig,
    data_root: str,
) -> np.ndarray:
    features = datasets.load_onset_features(
        audio_path,
        experiment.dataset.feature_source,
        experiment.dataset.mert_features_dir,
        data_root,
    )
    features = datasets.normalize_onset_spectrogram(features)
    pred_probs = np.asarray(
        model.predict(np.expand_dims(features, 0), verbose=0)
    ).reshape(-1)
    return pred_probs


def _gt_batch(chart_path: str) -> tuple[np.ndarray, np.ndarray, int]:
    times = charts.load_onset_times(chart_path, max_steps=None)
    n_gt = int(times.size)
    n_max = max(n_gt, 1)
    gt_times = np.zeros((1, n_max), dtype=np.float32)
    gt_mask = np.zeros((1, n_max), dtype=np.float32)
    gt_times[0, :n_gt] = times.astype(np.float32)
    gt_mask[0, :n_gt] = 1.0
    return gt_times, gt_mask, n_gt


def _metrics_at_threshold(
    pred_probs: np.ndarray,
    gt_times: np.ndarray,
    gt_mask: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    pred_times, pred_conf = dense_overfit_eval.peak_times_and_confidence(
        pred_probs,
        confidence_threshold=threshold,
        min_onset_distance_ms=MIN_ONSET_DISTANCE_MS,
    )
    n_peaks = int(pred_times.size)
    n_max = max(int(gt_mask.sum()), n_peaks, 1)
    pred_times_batch = np.zeros((1, n_max), dtype=np.float32)
    pred_conf_batch = np.zeros((1, n_max), dtype=np.float32)
    if n_peaks:
        pred_times_batch[0, :n_peaks] = pred_times
        pred_conf_batch[0, :n_peaks] = pred_conf
    gt_times_pad = np.zeros((1, n_max), dtype=np.float32)
    gt_mask_pad = np.zeros((1, n_max), dtype=np.float32)
    n_gt = int(gt_mask.sum())
    gt_times_pad[0, :n_gt] = gt_times[0, :n_gt]
    gt_mask_pad[0, :n_gt] = gt_mask[0, :n_gt]
    tp, fp, fn = metrics.count_event_onset_errors_numpy(
        pred_times_batch,
        pred_conf_batch,
        gt_times_pad,
        gt_mask_pad,
        TOLERANCE_SEC,
        threshold,
        MIN_ONSET_DISTANCE_MS,
    )
    _p, _r, f1 = metrics.event_onset_f1_numpy(
        pred_times_batch,
        pred_conf_batch,
        gt_times_pad,
        gt_mask_pad,
        TOLERANCE_SEC,
        threshold,
        MIN_ONSET_DISTANCE_MS,
    )
    return {
        "event_f1": float(f1),
        "event_tp": float(tp),
        "event_fp": float(fp),
        "event_fn": float(fn),
        "num_peaks": float(n_peaks),
    }


def _mert_feature_stats(
    audio_path: str,
    experiment: config.OnsetExperimentConfig,
    data_root: str,
) -> dict[str, float]:
    mert_path = ssl_features.mert_npy_path(
        audio_path,
        experiment.dataset.mert_features_dir,
        data_root,
    )
    raw = np.load(mert_path).astype(np.float64)
    per_dim_var = raw.var(axis=0)
    return {
        "feature_mean_raw": float(raw.mean()),
        "feature_std_raw": float(raw.std()),
        "per_dim_var_mean": float(per_dim_var.mean()),
        "per_dim_var_max": float(per_dim_var.max()),
        "n_frames": int(raw.shape[0]),
        "feature_dim": int(raw.shape[1]),
    }


def _song_metadata(
    song: str,
    audio_path: str,
    chart_path: str,
    experiment: config.OnsetExperimentConfig,
    data_root: str,
) -> dict[str, float | int | bool]:
    times = charts.load_onset_times(chart_path, max_steps=None)
    duration = float(librosa.get_duration(path=audio_path))
    n_gt = int(times.size)
    mert_stats = _mert_feature_stats(audio_path, experiment, data_root)
    return {
        "audio_duration_sec": duration,
        "n_gt_steps": n_gt,
        "step_density_hz": float(n_gt / duration) if duration > 0 else 0.0,
        "chart_bpm": _chart_bpm(chart_path),
        "chart_exceeds_1024": bool(n_gt > 1024),
        **mert_stats,
    }


def _train_song_names(experiment: config.OnsetExperimentConfig) -> set[str]:
    all_pairs = pairing.list_audio_chart_pairs(experiment.dataset.data_dir)
    selected = datasets.select_song_pairs(
        all_pairs,
        max_songs=experiment.dataset.max_train_songs,
        seed=experiment.run.seed,
    )
    return {pathlib.Path(audio_path).parent.name for audio_path, _ in selected}


def _distribution_summary(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {}
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "p25": float(np.percentile(arr, 25)),
        "median": float(np.median(arr)),
        "p75": float(np.percentile(arr, 75)),
        "max": float(arr.max()),
    }


def _ood_analysis(
    experiment: config.OnsetExperimentConfig,
    val_pairs: dict[str, tuple[str, str]],
    train_songs: set[str],
    investigate_path: str,
) -> dict:
    val_dir = experiment.dataset.val_data_dir
    train_dir = experiment.dataset.data_dir
    train_meta: list[dict] = []
    for audio_path, chart_path in pairing.list_audio_chart_pairs(train_dir):
        song = pathlib.Path(audio_path).parent.name
        if song not in train_songs:
            continue
        train_meta.append(
            {
                "song": song,
                **_song_metadata(song, audio_path, chart_path, experiment, train_dir),
            }
        )

    train_dists = {
        field: _distribution_summary([row[field] for row in train_meta])
        for field in (
            "feature_mean_raw",
            "feature_std_raw",
            "per_dim_var_mean",
            "audio_duration_sec",
            "step_density_hz",
            "chart_bpm",
            "n_gt_steps",
        )
    }

    investigate_profiles: dict[str, dict] = {}
    if pathlib.Path(investigate_path).is_file():
        with pathlib.Path(investigate_path).open(encoding="utf-8") as inv_file:
            inv_data = json.load(inv_file)
        investigate_profiles = inv_data.get("feature_profiles", {})

    bottom_3: dict[str, dict] = {}
    for song in BOTTOM_3_SONGS:
        audio_path, chart_path = val_pairs[song]
        meta = _song_metadata(song, audio_path, chart_path, experiment, val_dir)
        z_scores: dict[str, float] = {}
        for field in train_dists:
            train_mean = train_dists[field]["mean"]
            train_std = train_dists[field]["std"]
            if train_std > 0:
                z_scores[field] = float((meta[field] - train_mean) / train_std)
        pred_profile = investigate_profiles.get(song, {})
        bottom_3[song] = {
            "in_train_set": song in train_songs,
            "metadata": meta,
            "z_score_vs_train": z_scores,
            "pred_calibration_at_0_5": {
                key: pred_profile[key]
                for key in (
                    "pred_prob_mean",
                    "pred_prob_max",
                    "pred_prob_p95",
                    "pred_prob_at_gt_frames_mean",
                    "pred_prob_at_non_gt_mean",
                    "frame_recall_at_0_5",
                    "peak_matched_frac_at_0_5",
                )
                if key in pred_profile
            },
        }

    return {
        "train_song_count": len(train_songs),
        "train_metadata_distribution": train_dists,
        "bottom_3": bottom_3,
        "verdict_notes": {
            "in_train": {
                song: bottom_3[song]["in_train_set"] for song in BOTTOM_3_SONGS
            },
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Sweep val peak-pick thresholds.")
    parser.add_argument(
        "--config",
        default="configs/dense_mert_v2_75train_200ep.json",
    )
    parser.add_argument(
        "--output",
        default="models_wsl/dense_mert_v2_75train_200ep/val_threshold_sweep.json",
    )
    parser.add_argument(
        "--investigate-json",
        default="models_wsl/dense_mert_v2_75train_200ep/investigate_bad_val.json",
    )
    args = parser.parse_args()

    experiment = config.OnsetExperimentConfig.from_json(args.config)
    model_dir = pathlib.Path(experiment.run.model_output_dir)
    keras_files = sorted(
        path.name for path in model_dir.iterdir() if path.name.endswith(".keras")
    )
    model_path = str(model_dir / keras_files[0])
    model = tf.keras.models.load_model(model_path, compile=False)

    val_pairs = _pair_map(experiment.dataset.val_data_dir)
    val_songs = sorted(val_pairs)
    val_dir = experiment.dataset.val_data_dir

    song_cache: dict[str, dict] = {}
    for song in val_songs:
        audio_path, chart_path = val_pairs[song]
        pred_probs = _predict_probs(model, audio_path, experiment, val_dir)
        gt_times, gt_mask, _n_gt = _gt_batch(chart_path)
        song_cache[song] = {
            "pred_probs": pred_probs,
            "gt_times": gt_times,
            "gt_mask": gt_mask,
        }

    per_song_sweep: dict[str, dict] = {}
    oracle_f1_values: list[float] = []
    oracle_tp = oracle_fp = oracle_fn = 0.0
    oracle_thresholds: list[float] = []

    for song in val_songs:
        cache = song_cache[song]
        curve = []
        best_f1 = -1.0
        best_thr = THRESHOLDS[0]
        best_row: dict[str, float] = {}
        for threshold in THRESHOLDS:
            row = _metrics_at_threshold(
                cache["pred_probs"],
                cache["gt_times"],
                cache["gt_mask"],
                threshold,
            )
            curve.append({"threshold": threshold, **row})
            if row["event_f1"] > best_f1:
                best_f1 = row["event_f1"]
                best_thr = threshold
                best_row = row
        f1_at_0_5 = next(row["event_f1"] for row in curve if row["threshold"] == 0.5)
        per_song_sweep[song] = {
            "curve": curve,
            "best_threshold": best_thr,
            "best_event_f1": best_f1,
            "f1_at_0_5": f1_at_0_5,
            "best_metrics": best_row,
        }
        oracle_f1_values.append(best_f1)
        oracle_thresholds.append(best_thr)
        oracle_tp += best_row["event_tp"]
        oracle_fp += best_row["event_fp"]
        oracle_fn += best_row["event_fn"]

    global_curve = []
    best_global_f1 = -1.0
    best_global_thr = THRESHOLDS[0]
    best_global_row: dict[str, float] = {}
    for threshold in THRESHOLDS:
        f1_values: list[float] = []
        total_tp = total_fp = total_fn = 0.0
        for song in val_songs:
            cache = song_cache[song]
            row = _metrics_at_threshold(
                cache["pred_probs"],
                cache["gt_times"],
                cache["gt_mask"],
                threshold,
            )
            f1_values.append(row["event_f1"])
            total_tp += row["event_tp"]
            total_fp += row["event_fp"]
            total_fn += row["event_fn"]
        mean_f1 = float(np.mean(f1_values)) if f1_values else 0.0
        micro_f1 = _micro_f1(total_tp, total_fp, total_fn)
        global_row = {
            "threshold": threshold,
            "mean_event_f1": mean_f1,
            "micro_event_f1": micro_f1,
            "micro_tp": total_tp,
            "micro_fp": total_fp,
            "micro_fn": total_fn,
        }
        global_curve.append(global_row)
        if micro_f1 > best_global_f1:
            best_global_f1 = micro_f1
            best_global_thr = threshold
            best_global_row = global_row

    row_at_0_5 = next(row for row in global_curve if row["threshold"] == 0.5)
    train_songs = _train_song_names(experiment)
    ood = _ood_analysis(experiment, val_pairs, train_songs, args.investigate_json)

    report = {
        "model_path": model_path,
        "config_path": args.config,
        "num_val_songs": len(val_songs),
        "eval_kwargs": {
            "thresholds": list(THRESHOLDS),
            "min_onset_distance_ms": MIN_ONSET_DISTANCE_MS,
            "tolerance_sec": TOLERANCE_SEC,
        },
        "global_sweep": {
            "curve": global_curve,
            "best_threshold": best_global_thr,
            "best_mean_event_f1": best_global_row["mean_event_f1"],
            "best_micro_event_f1": best_global_row["micro_event_f1"],
            "at_0_5_mean_event_f1": row_at_0_5["mean_event_f1"],
            "at_0_5_micro_event_f1": row_at_0_5["micro_event_f1"],
            "best_metrics": best_global_row,
        },
        "oracle_per_song": {
            "mean_event_f1": float(np.mean(oracle_f1_values)),
            "micro_event_f1": _micro_f1(oracle_tp, oracle_fp, oracle_fn),
            "micro_tp": oracle_tp,
            "micro_fp": oracle_fp,
            "micro_fn": oracle_fn,
            "threshold_distribution": _distribution_summary(oracle_thresholds),
            "threshold_histogram": {
                str(thr): int(sum(1 for t in oracle_thresholds if t == thr))
                for thr in THRESHOLDS
            },
            "per_song": {
                song: {
                    "best_threshold": per_song_sweep[song]["best_threshold"],
                    "best_event_f1": per_song_sweep[song]["best_event_f1"],
                    "f1_at_0_5": per_song_sweep[song]["f1_at_0_5"],
                }
                for song in val_songs
            },
        },
        "per_song_sweep": per_song_sweep,
        "ood_bottom_3": ood,
    }

    output_file = pathlib.Path(args.output)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as out_file:
        json.dump(report, out_file, indent=2)

    print(f"wrote {args.output}")
    print("\n=== global threshold sweep ===")
    print(
        f"  best thr={best_global_thr:.2f}: "
        f"mean F1={best_global_row['mean_event_f1']:.4f} "
        f"micro F1={best_global_row['micro_event_f1']:.4f}"
    )
    print(
        f"  @ thr=0.50: mean F1={row_at_0_5['mean_event_f1']:.4f} "
        f"micro F1={row_at_0_5['micro_event_f1']:.4f}"
    )
    print("\n=== oracle per-song thresholds ===")
    print(
        f"  mean F1={report['oracle_per_song']['mean_event_f1']:.4f} "
        f"micro F1={report['oracle_per_song']['micro_event_f1']:.4f}"
    )
    print(f"  threshold dist: {report['oracle_per_song']['threshold_distribution']}")
    print("\n=== OOD bottom 3 (in train?) ===")
    for song in BOTTOM_3_SONGS:
        row = ood["bottom_3"][song]
        print(f"  {song}: in_train={row['in_train_set']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
