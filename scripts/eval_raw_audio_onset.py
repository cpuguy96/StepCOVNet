#!/usr/bin/env python
"""End-to-end raw-audio onset evaluation: Beat This! vs Oracle GT beat grids.

Evaluates a trained ITGPT-MERT placement checkpoint directly against ground-truth
human onset timestamps across one or more beat-tracking modes:

* ``oracle`` — ground-truth simfile beat grid (upper bound)
* ``beat_this_raw`` — built-in Beat This! without lattice regularization or snap
* ``beat_this_reg`` — Beat This! + constant-BPM regularization / integer-BPM lock
* ``beat_this_full`` — shipped end-to-end pipeline (+ zero-lag attack snapping)
* ``librosa`` — offline librosa fallback (+ regularization & attack snapping)

Usage::

    python scripts/eval_raw_audio_onset.py \
      --training_index_path data/literature_fraxtil_orig/training_index_standard.json \
      --model_path models/best_slot_f1.pt \
      --split val \
      --output eval_raw_audio.json
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import json
import pathlib
from typing import Any

import numpy as np

from stepcovnet import beat_tracking, wsl_gpu

wsl_gpu.bootstrap_gpu_script("scripts/eval_raw_audio_onset.py")

from stepcovnet.dataset_prep import models as prep_models
from stepcovnet.dataset_prep import training_index
from stepcovnet.ddc import peak_pick
from stepcovnet.ddcl import features as ddcl_features
from stepcovnet.ddcl import slots as ddcl_slots
from stepcovnet.pytorch import runtime
from stepcovnet.pytorch.itgpt import trainers as itgpt_trainers

VALID_MODES: tuple[str, ...] = (
    "oracle",
    "beat_this_raw",
    "beat_this_reg",
    "beat_this_full",
    "librosa",
)
DEFAULT_MODES: tuple[str, ...] = (
    "oracle",
    "beat_this_raw",
    "beat_this_reg",
    "beat_this_full",
    "librosa",
)


@dataclasses.dataclass(frozen=True)
class SongEvalChart:
    """One ground-truth chart belonging to a song in the manifest.

    Attributes:
        song_key: Unique ``bundle/id`` song identifier.
        difficulty: Chart difficulty label.
        meter: Numeric StepMania meter.
        gt_times_sec: Sorted unique ground-truth onset timestamps in seconds.
        n_gt_beats: Number of integer beats covered by the ground-truth grid.
        target_density: Ground-truth steps per beat used for density conditioning.
    """

    song_key: str
    difficulty: str
    meter: int
    gt_times_sec: np.ndarray
    n_gt_beats: int
    target_density: float


@dataclasses.dataclass(frozen=True)
class SongEvalGroup:
    """One song and all its evaluation charts from the manifest.

    Attributes:
        song_key: Unique ``bundle/id`` song identifier.
        audio_path: Path to the raw audio file.
        oracle_grid: Ground-truth simfile :class:`stepcovnet.beat_tracking.BeatGrid`.
        gt_bpm: Ground-truth initial BPM.
        charts: Tuple of charts belonging to this song.
    """

    song_key: str
    audio_path: str
    oracle_grid: beat_tracking.BeatGrid
    gt_bpm: float
    charts: tuple[SongEvalChart, ...]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="End-to-end raw-audio onset evaluation (Beat This! vs Oracle GT beats)."
    )
    parser.add_argument(
        "--training_index_path",
        required=True,
        help="Path to training_index*.json manifest.",
    )
    parser.add_argument(
        "--model_path",
        required=True,
        help="Path to a trained ITGPT-MERT .pt checkpoint.",
    )
    parser.add_argument(
        "--split",
        choices=(training_index.SPLIT_TRAIN, training_index.SPLIT_VAL),
        default=training_index.SPLIT_VAL,
        help="Manifest split to evaluate (default: val).",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=list(VALID_MODES),
        default=list(DEFAULT_MODES),
        help="Beat-tracking modes to compare.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Slot occupancy probability threshold (default: 0.5).",
    )
    parser.add_argument(
        "--allowed_slots",
        choices=("raw", "musical"),
        default="musical",
        help=(
            "Slot filter applied during decoding: 'musical' (16th + 24th "
            "subdivisions, matching shipped generation) or 'raw' (all 48 slots)."
        ),
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Compute device ('auto', 'cuda', 'mps', 'cpu').",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Optional path to save the full JSON evaluation report.",
    )
    return parser


def _load_song_groups(
    training_index_path: str | pathlib.Path,
    *,
    split: training_index.SplitName,
) -> list[SongEvalGroup]:
    """Load and group ground-truth charts by song from a training manifest."""
    idx_path = pathlib.Path(training_index_path)
    idx = training_index.load_training_index(idx_path)
    root = training_index.resolve_output_dir(idx, idx_path)
    rows = training_index.rows_from_index(idx, root, split=split)

    by_song: dict[str, list[Any]] = collections.defaultdict(list)
    for row in rows:
        song_key = f"{pathlib.Path(row.audio_path).parent.parent.name}/{pathlib.Path(row.audio_path).parent.name}"
        by_song[song_key].append(row)

    groups: list[SongEvalGroup] = []
    for song_key in sorted(by_song):
        song_rows = by_song[song_key]
        audio_path = str(song_rows[0].audio_path)
        chart_path = pathlib.Path(song_rows[0].chart_json_path)
        doc = prep_models.ParsedSongPack.from_dict(
            json.loads(chart_path.read_text(encoding="utf-8"))
        )
        segments = doc.metadata.bpm_segments
        offset_sec = float(doc.metadata.offset_sec)
        gt_bpm = float(doc.metadata.initial_bpm)

        max_time = 0.0
        chart_items: list[tuple[str, int, np.ndarray]] = []
        for row in song_rows:
            chart = doc.charts[int(row.chart_index)]
            gt_times = np.unique(np.sort(np.asarray(chart.times_sec, dtype=np.float64)))
            if gt_times.size == 0:
                continue
            max_time = max(max_time, float(gt_times[-1]))
            chart_items.append(
                (str(chart.summary.difficulty), int(chart.summary.meter), gt_times)
            )
        if not chart_items or not segments:
            continue

        last_beat = ddcl_slots.time_to_beat(max_time, offset_sec, segments)
        n_gt_beats = max(2, int(np.ceil(last_beat)) + 1)
        beat_edges = ddcl_slots.beat_times_sec(n_gt_beats, offset_sec, segments)
        gt_beats = beat_edges[:-1]
        oracle_grid = beat_tracking.BeatGrid(
            beat_times=gt_beats,
            downbeat_times=gt_beats[::4],
            bpm=gt_bpm,
            tracker="oracle",
        )

        song_charts = tuple(
            SongEvalChart(
                song_key=song_key,
                difficulty=diff,
                meter=meter,
                gt_times_sec=gt_times,
                n_gt_beats=n_gt_beats,
                target_density=float(gt_times.size) / float(n_gt_beats),
            )
            for diff, meter, gt_times in chart_items
        )
        groups.append(
            SongEvalGroup(
                song_key=song_key,
                audio_path=audio_path,
                oracle_grid=oracle_grid,
                gt_bpm=gt_bpm,
                charts=song_charts,
            )
        )
    return groups


def _estimate_grid_for_mode(
    song: SongEvalGroup,
    mode: str,
    *,
    device: Any,
) -> beat_tracking.BeatGrid:
    """Produce a :class:`BeatGrid` for one song under the requested mode."""
    if mode == "oracle":
        return song.oracle_grid
    if mode == "beat_this_raw":
        return beat_tracking.estimate_beats(
            song.audio_path,
            tracker=beat_tracking.TRACKER_BEAT_THIS,
            device=device,
            regularize_tempo=False,
            snap_to_onsets=False,
        )
    if mode == "beat_this_reg":
        return beat_tracking.estimate_beats(
            song.audio_path,
            tracker=beat_tracking.TRACKER_BEAT_THIS,
            device=device,
            regularize_tempo=True,
            snap_to_onsets=False,
        )
    if mode == "beat_this_full":
        return beat_tracking.estimate_beats(
            song.audio_path,
            tracker=beat_tracking.TRACKER_BEAT_THIS,
            device=device,
            regularize_tempo=True,
            snap_to_onsets=True,
        )
    if mode == "librosa":
        return beat_tracking.estimate_beats(
            song.audio_path,
            tracker=beat_tracking.TRACKER_LIBROSA,
            device=device,
            regularize_tempo=True,
            snap_to_onsets=True,
        )
    raise ValueError(f"Unsupported mode: {mode!r}")


def _signed_timing_stats_ms(
    pred_times: np.ndarray,
    gt_times: np.ndarray,
    *,
    max_tol_sec: float = 0.070,
) -> tuple[float, float, float]:
    """Compute median signed offset, std, and early-to-late drift in ms."""
    pred = np.sort(np.asarray(pred_times, dtype=np.float64).reshape(-1))
    gt = np.sort(np.asarray(gt_times, dtype=np.float64).reshape(-1))
    if pred.size == 0 or gt.size == 0:
        return 0.0, 0.0, 0.0
    idx = np.searchsorted(gt, pred)
    pairs: list[tuple[float, float]] = []
    for t_p, k in zip(pred, idx, strict=True):
        cands: list[float] = []
        if k > 0:
            cands.append(float(t_p - gt[k - 1]))
        if k < gt.size:
            cands.append(float(t_p - gt[k]))
        if not cands:
            continue
        best = min(cands, key=abs)
        if abs(best) <= max_tol_sec:
            pairs.append((float(t_p), best * 1000.0))
    if not pairs:
        return 0.0, 0.0, 0.0
    errs = np.asarray([e for _, e in pairs], dtype=np.float64)
    q_n = max(1, errs.size // 4)
    drift = float(np.median(errs[-q_n:])) - float(np.median(errs[:q_n]))
    return float(np.median(errs)), float(np.std(errs)), drift


def evaluate_raw_audio_modes(
    song_groups: list[SongEvalGroup],
    model: Any,
    *,
    modes: tuple[str, ...] = DEFAULT_MODES,
    threshold: float = 0.5,
    allowed_slots: frozenset[int] | None = beat_tracking.SLOTS_16TH_24TH,
    device: Any = "cpu",
) -> dict[str, Any]:
    """Run end-to-end onset placement evaluation across beat-tracking modes.

    Args:
        song_groups: List of :class:`SongEvalGroup` objects to evaluate.
        model: Loaded PyTorch ITGPT placement model.
        modes: Beat-tracking modes to compare.
        threshold: Slot probability threshold.
        allowed_slots: Optional allowed 48-slot beat subdivisions.
        device: Target compute device.

    Returns:
        Structured report dict with per-mode aggregate metrics and per-song
        breakdowns.
    """
    target_device = runtime.resolve_device(device)
    n_frames = int(getattr(model, "n_frames_per_beat", 24))
    max_beats = int(getattr(model, "max_beats", 1024))

    # Preload per-song MERT once across all modes.
    mert_by_song: dict[str, np.ndarray] = {}
    for song in song_groups:
        mert_by_song[song.song_key] = beat_tracking._load_or_extract_mert_for_audio(
            song.audio_path,
            device="cuda" if target_device.type == "cuda" else "cpu",
        )

    # Cache Beat This! raw grid so beat_this_raw / reg / full only run neural
    # inference ONCE per song!
    raw_bt_cache: dict[str, beat_tracking.BeatGrid] = {}
    need_bt = any(m.startswith("beat_this") for m in modes)
    if need_bt:
        for song in song_groups:
            raw_bt_cache[song.song_key] = beat_tracking.estimate_beats(
                song.audio_path,
                tracker=beat_tracking.TRACKER_BEAT_THIS,
                device=target_device,
                regularize_tempo=False,
                snap_to_onsets=False,
            )

    mode_reports: dict[str, Any] = {}
    for mode in modes:
        counts_20 = peak_pick.OnsetMatchCounts(0, 0, 0)
        counts_35 = peak_pick.OnsetMatchCounts(0, 0, 0)
        counts_50 = peak_pick.OnsetMatchCounts(0, 0, 0)
        beat_counts_70 = peak_pick.OnsetMatchCounts(0, 0, 0)
        chart_f20: list[float] = []
        chart_f50: list[float] = []
        song_details: list[dict[str, Any]] = []

        for song in song_groups:
            if mode == "beat_this_raw":
                grid = raw_bt_cache[song.song_key]
            elif mode == "beat_this_reg":
                grid = beat_tracking.regularize_beat_grid(
                    raw_bt_cache[song.song_key],
                    audio_duration_sec=float(mert_by_song[song.song_key].shape[0])
                    / 100.0,
                )
            elif mode == "beat_this_full":
                reg = beat_tracking.regularize_beat_grid(
                    raw_bt_cache[song.song_key],
                    audio_duration_sec=float(mert_by_song[song.song_key].shape[0])
                    / 100.0,
                )
                grid = beat_tracking.snap_beats_to_onsets(reg, song.audio_path)
            else:
                grid = _estimate_grid_for_mode(song, mode, device=target_device)

            b_cnt = peak_pick.match_onsets(
                grid.beat_times, song.oracle_grid.beat_times, tolerance_sec=0.070
            )
            beat_counts_70 = peak_pick.add_counts(beat_counts_70, b_cnt)
            beat_off_ms, beat_std_ms, beat_drift_ms = _signed_timing_stats_ms(
                grid.beat_times, song.oracle_grid.beat_times, max_tol_sec=0.070
            )

            beat_mert = ddcl_features.beats_to_mert_tensor(
                mert_by_song[song.song_key],
                grid.beat_times_with_end,
                n_frames=n_frames,
            ).astype(np.float32)

            song_c20 = peak_pick.OnsetMatchCounts(0, 0, 0)
            song_c50 = peak_pick.OnsetMatchCounts(0, 0, 0)
            for chart in song.charts:
                # Adjust density target if the beat tracker detected a different
                # beat count (e.g. half/double-time) so total target steps match!
                eff_density = float(chart.gt_times_sec.size) / float(
                    max(1, beat_mert.shape[0])
                )
                pred_times = beat_tracking._predict_itgpt_onsets_for_mert(
                    song.song_key,
                    beat_mert,
                    grid,
                    model,
                    difficulty=chart.difficulty,
                    target_density=eff_density,
                    meter=chart.meter,
                    threshold=threshold,
                    max_beats=max_beats,
                    device=target_device,
                    fallback_top_k=False,
                    allowed_slots=allowed_slots,
                )
                c20 = peak_pick.match_onsets(
                    pred_times, chart.gt_times_sec, tolerance_sec=0.020
                )
                c35 = peak_pick.match_onsets(
                    pred_times, chart.gt_times_sec, tolerance_sec=0.035
                )
                c50 = peak_pick.match_onsets(
                    pred_times, chart.gt_times_sec, tolerance_sec=0.050
                )
                counts_20 = peak_pick.add_counts(counts_20, c20)
                counts_35 = peak_pick.add_counts(counts_35, c35)
                counts_50 = peak_pick.add_counts(counts_50, c50)
                song_c20 = peak_pick.add_counts(song_c20, c20)
                song_c50 = peak_pick.add_counts(song_c50, c50)
                chart_f20.append(c20.f_score)
                chart_f50.append(c50.f_score)

            song_details.append(
                {
                    "song_key": song.song_key,
                    "n_charts": len(song.charts),
                    "gt_bpm": round(song.gt_bpm, 3),
                    "pred_bpm": round(float(grid.bpm), 3),
                    "beat_f1_70ms": round(b_cnt.f_score, 4),
                    "beat_offset_ms": round(beat_off_ms, 2),
                    "beat_std_ms": round(beat_std_ms, 2),
                    "beat_drift_ms": round(beat_drift_ms, 2),
                    "ev_f1_20ms": round(song_c20.f_score, 4),
                    "ev_f1_50ms": round(song_c50.f_score, 4),
                    "n_pred": song_c20.true_positives + song_c20.false_positives,
                    "n_gt": song_c20.true_positives + song_c20.false_negatives,
                }
            )

        mode_reports[mode] = {
            "ev_f1_20ms": counts_20.f_score,
            "ev_p_20ms": counts_20.precision,
            "ev_r_20ms": counts_20.recall,
            "ev_f1_35ms": counts_35.f_score,
            "ev_f1_50ms": counts_50.f_score,
            "ev_p_50ms": counts_50.precision,
            "ev_r_50ms": counts_50.recall,
            "macro_ev_f1_20ms": peak_pick.f_score_c(chart_f20),
            "macro_ev_f1_50ms": peak_pick.f_score_c(chart_f50),
            "beat_f1_70ms": beat_counts_70.f_score,
            "n_pred": counts_20.true_positives + counts_20.false_positives,
            "n_gt": counts_20.true_positives + counts_20.false_negatives,
            "n_songs": len(song_groups),
            "n_charts": len(chart_f20),
            "songs": song_details,
        }

    return {
        "n_songs": len(song_groups),
        "n_charts": sum(len(s.charts) for s in song_groups),
        "threshold": float(threshold),
        "modes": mode_reports,
    }


def _print_summary(report: dict[str, Any]) -> None:
    print(
        f"\n=== End-to-End Raw-Audio Onset Evaluation "
        f"({report['n_songs']} songs / {report['n_charts']} charts, thr={report['threshold']:.2f}) ==="
    )
    print(
        f"{'Mode':16s} | {'BeatF1@70':>9s} | {'evF1@20ms':>9s} | {'evF1@35ms':>9s} | "
        f"{'evF1@50ms':>9s} | {'P@20':>6s} | {'R@20':>6s} | {'Pred/GT':>11s}"
    )
    print("-" * 92)
    for mode, stats in report["modes"].items():
        print(
            f"{mode:16s} | {stats['beat_f1_70ms']:9.4f} | {stats['ev_f1_20ms']:9.4f} | "
            f"{stats['ev_f1_35ms']:9.4f} | {stats['ev_f1_50ms']:9.4f} | "
            f"{stats['ev_p_20ms']:6.3f} | {stats['ev_r_20ms']:6.3f} | "
            f"{stats['n_pred']:5d}/{stats['n_gt']:<5d}"
        )

    focus = "beat_this_full" if "beat_this_full" in report["modes"] else None
    if focus is not None:
        oracle_map = (
            {s["song_key"]: s for s in report["modes"]["oracle"]["songs"]}
            if "oracle" in report["modes"]
            else {}
        )
        print(f"\n=== Per-Song Breakdown ({focus} vs oracle) ===")
        print(
            f"{'Song':38s} | {'BPM (GT->Pred)':>16s} | {'Off(ms)':>7s} | {'Drift':>6s} | "
            f"{'evF1@20':>7s} | {'Oracle20':>8s} | {'evF1@50':>7s}"
        )
        print("-" * 102)
        for s in report["modes"][focus]["songs"]:
            o20 = oracle_map.get(s["song_key"], {}).get("ev_f1_20ms", float("nan"))
            print(
                f"{s['song_key'][:38]:38s} | {s['gt_bpm']:6.1f} -> {s['pred_bpm']:6.2f} | "
                f"{s['beat_offset_ms']:+7.1f} | {s['beat_drift_ms']:+6.1f} | "
                f"{s['ev_f1_20ms']:7.4f} | {o20:8.4f} | {s['ev_f1_50ms']:7.4f}"
            )


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for end-to-end raw-audio onset evaluation.

    Args:
        argv: Optional CLI argument list.

    Returns:
        Exit code ``0`` on success.
    """
    args = _build_parser().parse_args(argv)
    wsl_gpu.guard_pytorch_gpu_job(__file__)
    device = runtime.resolve_device(args.device)
    model = itgpt_trainers.load_placement_model(args.model_path, device=device)
    groups = _load_song_groups(args.training_index_path, split=args.split)
    allowed = beat_tracking.SLOTS_16TH_24TH if args.allowed_slots == "musical" else None
    report = evaluate_raw_audio_modes(
        groups,
        model,
        modes=tuple(args.modes),
        threshold=args.threshold,
        allowed_slots=allowed,
        device=device,
    )
    _print_summary(report)
    if args.output:
        out_path = pathlib.Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"\nSaved report to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
