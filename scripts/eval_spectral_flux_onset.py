"""Classical spectral-flux onset baseline on val audio (peak-pick event F1)."""

import argparse
import json
import os
import pathlib

import librosa
import numpy as np

from stepcovnet import constants
from stepcovnet import dense_overfit_eval
from stepcovnet import pairing
from stepcovnet.onset_events import charts
from stepcovnet.onset_events import metrics

DEFAULT_THRESHOLDS = (
    0.05,
    0.1,
    0.15,
    0.2,
    0.25,
    0.3,
    0.35,
    0.4,
    0.45,
    0.5,
    0.55,
    0.6,
)
DEFAULT_MIN_ONSET_DISTANCE_MS = 50.0
DEFAULT_TOLERANCE_SEC = 0.02


def spectral_flux_envelope(audio_path: str) -> np.ndarray:
    """Return min-max normalized onset-strength envelope at the dense hop rate."""
    hop_length = int(round(constants.TARGET_SR * constants.HOP_COEFF))
    audio, _sr = librosa.load(audio_path, sr=constants.TARGET_SR, mono=True)
    strength = librosa.onset.onset_strength(
        y=audio,
        sr=constants.TARGET_SR,
        hop_length=hop_length,
    )
    strength = np.asarray(strength, dtype=np.float64)
    peak = float(strength.max()) if strength.size else 0.0
    if peak <= 0.0:
        return np.zeros_like(strength, dtype=np.float32)
    return (strength / peak).astype(np.float32)


def _micro_f1(tp: float, fp: float, fn: float) -> float:
    denom = 2.0 * tp + fp + fn
    return float(2.0 * tp / denom) if denom > 0 else 0.0


def _event_metrics_for_song(
    envelope: np.ndarray,
    chart_path: str,
    *,
    confidence_threshold: float,
    min_onset_distance_ms: float,
    tolerance_sec: float,
) -> dict[str, float]:
    pred_times, pred_conf = dense_overfit_eval.peak_times_and_confidence(
        envelope,
        confidence_threshold=confidence_threshold,
        min_onset_distance_ms=min_onset_distance_ms,
        hop_sec=constants.HOP_COEFF,
    )
    gt_times, gt_mask = dense_overfit_eval.build_gt_batch(chart_path)
    pred_times_batch, pred_conf_batch, gt_times_batch, gt_mask_batch = (
        dense_overfit_eval._align_event_batches(pred_times, pred_conf, gt_times, gt_mask)
    )
    tp, fp, fn = metrics.count_event_onset_errors_numpy(
        pred_times_batch,
        pred_conf_batch,
        gt_times_batch,
        gt_mask_batch,
        tolerance_sec,
        confidence_threshold,
        min_onset_distance_ms,
    )
    _p, _r, f1 = metrics.event_onset_f1_numpy(
        pred_times_batch,
        pred_conf_batch,
        gt_times_batch,
        gt_mask_batch,
        tolerance_sec,
        confidence_threshold,
        min_onset_distance_ms,
    )
    return {
        "event_f1": float(f1),
        "event_tp": float(tp),
        "event_fp": float(fp),
        "event_fn": float(fn),
        "num_peaks": float(pred_times.size),
    }


def eval_spectral_flux_val(
    val_data_dir: str,
    *,
    confidence_threshold: float,
    min_onset_distance_ms: float,
    tolerance_sec: float,
    max_songs: int | None = None,
    song_names: tuple[str, ...] | None = None,
) -> dict:
    """Evaluate spectral-flux peak-pick event F1 on val audio/chart pairs."""
    pair_map: dict[str, tuple[str, str]] = {}
    for audio_path, chart_path in pairing.list_audio_chart_pairs(val_data_dir):
        song = pathlib.Path(audio_path).parent.name
        pair_map[song] = (audio_path, chart_path)

    if song_names:
        songs = [name for name in song_names if name in pair_map]
    else:
        songs = sorted(pair_map)
    if max_songs is not None:
        songs = songs[: max_songs]

    per_song: dict[str, dict[str, float]] = {}
    total_tp = total_fp = total_fn = 0.0
    f1_values: list[float] = []

    for song in songs:
        audio_path, chart_path = pair_map[song]
        envelope = spectral_flux_envelope(audio_path)
        row = _event_metrics_for_song(
            envelope,
            chart_path,
            confidence_threshold=confidence_threshold,
            min_onset_distance_ms=min_onset_distance_ms,
            tolerance_sec=tolerance_sec,
        )
        per_song[song] = row
        total_tp += row["event_tp"]
        total_fp += row["event_fp"]
        total_fn += row["event_fn"]
        f1_values.append(row["event_f1"])

    return {
        "eval_split": val_data_dir,
        "num_songs": len(songs),
        "mean_event_f1": float(np.mean(f1_values)) if f1_values else 0.0,
        "micro_event_f1": _micro_f1(total_tp, total_fp, total_fn),
        "micro_tp": total_tp,
        "micro_fp": total_fp,
        "micro_fn": total_fn,
        "eval_kwargs": {
            "confidence_threshold": confidence_threshold,
            "min_onset_distance_ms": min_onset_distance_ms,
            "tolerance_sec": tolerance_sec,
            "method": "librosa.onset.onset_strength",
        },
        "per_song": per_song,
    }


def sweep_thresholds(
    val_data_dir: str,
    thresholds: tuple[float, ...],
    *,
    min_onset_distance_ms: float,
    tolerance_sec: float,
    max_songs: int | None = None,
    song_names: tuple[str, ...] | None = None,
) -> list[dict]:
    """Return global micro/mean F1 for each threshold."""
    curve: list[dict] = []
    for threshold in thresholds:
        report = eval_spectral_flux_val(
            val_data_dir,
            confidence_threshold=threshold,
            min_onset_distance_ms=min_onset_distance_ms,
            tolerance_sec=tolerance_sec,
            max_songs=max_songs,
            song_names=song_names,
        )
        curve.append(
            {
                "threshold": threshold,
                "mean_event_f1": report["mean_event_f1"],
                "micro_event_f1": report["micro_event_f1"],
                "micro_tp": report["micro_tp"],
                "micro_fp": report["micro_fp"],
                "micro_fn": report["micro_fn"],
            },
        )
    return curve


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Spectral-flux onset baseline on val split.",
    )
    parser.add_argument(
        "--val_data_dir",
        type=str,
        default="data/v2/val",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.2,
        help="Peak height threshold on normalized envelope; ignored when --sweep.",
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Sweep DEFAULT_THRESHOLDS and report best micro F1.",
    )
    parser.add_argument(
        "--max_songs",
        type=int,
        default=0,
        help="Limit to first N songs (0 = all).",
    )
    parser.add_argument(
        "--songs",
        type=str,
        default="",
        help="Comma-separated song folder names (overrides max_songs ordering).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models_wsl/research/spectral_flux_val_eval.json",
    )
    args = parser.parse_args(argv)

    max_songs = args.max_songs if args.max_songs > 0 else None
    song_names = tuple(s.strip() for s in args.songs.split(",") if s.strip()) or None

    if args.sweep:
        curve = sweep_thresholds(
            args.val_data_dir,
            DEFAULT_THRESHOLDS,
            min_onset_distance_ms=DEFAULT_MIN_ONSET_DISTANCE_MS,
            tolerance_sec=DEFAULT_TOLERANCE_SEC,
            max_songs=max_songs,
            song_names=song_names,
        )
        best_row = max(curve, key=lambda row: row["micro_event_f1"])
        report = {
            "eval_split": args.val_data_dir,
            "num_songs": max_songs,
            "sweep_curve": curve,
            "best_threshold": best_row["threshold"],
            "best_micro_event_f1": best_row["micro_event_f1"],
            "best_mean_event_f1": best_row["mean_event_f1"],
            "eval_kwargs": {
                "thresholds": list(DEFAULT_THRESHOLDS),
                "min_onset_distance_ms": DEFAULT_MIN_ONSET_DISTANCE_MS,
                "tolerance_sec": DEFAULT_TOLERANCE_SEC,
                "method": "librosa.onset.onset_strength",
            },
        }
    else:
        report = eval_spectral_flux_val(
            args.val_data_dir,
            confidence_threshold=args.threshold,
            min_onset_distance_ms=DEFAULT_MIN_ONSET_DISTANCE_MS,
            tolerance_sec=DEFAULT_TOLERANCE_SEC,
            max_songs=max_songs,
            song_names=song_names,
        )

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as out_file:
        json.dump(report, out_file, indent=2)

    print(f"wrote {args.output}")
    if args.sweep:
        print(
            f"  best thr={report['best_threshold']:.2f}: "
            f"micro F1={report['best_micro_event_f1']:.4f}",
        )
    else:
        print(
            f"  songs={report['num_songs']} "
            f"micro F1={report['micro_event_f1']:.4f} "
            f"@ threshold={args.threshold}",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
