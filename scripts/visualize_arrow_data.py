r"""Visualize arrow training data: aggregate stats and plots for a training data directory.

Usage:
    python scripts/visualize_arrow_data.py --data_dir=data/v2/train
    python scripts/visualize_arrow_data.py --data_dir=data/v2/train --output_dir=output/viz --no_show
    python scripts/visualize_arrow_data.py --data_dir=data/v2/train --top_arrows=15
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

from stepcovnet import constants, datasets, generator, metrics

# Note-kind labels: 0=empty, 1=single, 2=chord, 3=hold_start, 4=hold_end, 5=hold_both
NOTE_KIND_LABELS = [
    "Empty",
    "Single",
    "Chord",
    "Hold start",
    "Hold end",
    "Hold both",
]
COLUMN_LABELS = ["Left", "Down", "Up", "Right"]

# Bins for timing correlation plots
N_TIME_BINS = 10
TIME_BIN_EDGES = np.linspace(0.0, 1.0, N_TIME_BINS + 1)
# Inter-step interval bins (seconds): 0-0.1, 0.1-0.2, ..., 0.9-1.0, 1.0+
INTERVAL_BIN_EDGES = np.concatenate([np.arange(0.0, 1.05, 0.1), [np.inf]])
N_INTERVAL_BINS = len(INTERVAL_BIN_EDGES) - 1

PARSER = argparse.ArgumentParser(
    description="Visualize arrow training data: stats and plots for a training data directory."
)
PARSER.add_argument(
    "--data_dir",
    type=str,
    required=True,
    help="Directory containing chart (.txt) and audio files (same layout as arrow training data).",
)
PARSER.add_argument(
    "--output_dir",
    type=str,
    default=None,
    help="Directory to save summary.txt and PNG figures. If not set, figures are only shown.",
)
PARSER.add_argument(
    "--no_show",
    action="store_true",
    help="Do not call plt.show(); only save files when --output_dir is set.",
)
PARSER.add_argument(
    "--top_arrows",
    type=int,
    default=25,
    help="Number of top arrow types to show in bar chart (default 25).",
)


def _build_note_kind_lookup() -> np.ndarray:
    """Build length-256 array mapping arrow code (0..255) to note kind (0..5).

    Same logic as metrics._build_arrow_note_kind_table().
    """
    table = np.zeros(constants.N_ARROW_TYPES, dtype=np.int32)
    for n in range(constants.N_ARROW_TYPES):
        d0 = (n // 64) % 4
        d1 = (n // 16) % 4
        d2 = (n // 4) % 4
        d3 = n % 4
        n1 = int(d0 == 1) + int(d1 == 1) + int(d2 == 1) + int(d3 == 1)
        n2 = int(d0 == 2) + int(d1 == 2) + int(d2 == 2) + int(d3 == 2)
        n3 = int(d0 == 3) + int(d1 == 3) + int(d2 == 3) + int(d3 == 3)
        if n == 0:
            kind = 0
        elif n2 >= 1 and n3 >= 1:
            kind = 5
        elif n2 >= 1:
            kind = 3
        elif n3 >= 1:
            kind = 4
        elif n1 == 1:
            kind = 1
        elif n1 >= 2:
            kind = 2
        else:
            kind = 0
        table[n] = kind
    return table


def _parse_bpm(chart_path: str) -> float | None:
    """Read BPM from chart header (second line). Returns None on error."""
    try:
        with open(chart_path) as f:
            f.readline()  # TITLE
            line = f.readline()
        return float(line.removeprefix("BPM").strip())
    except (ValueError, OSError):
        return None


def _column_activity_vectorized(cols: np.ndarray) -> np.ndarray:
    """Return (4,) array: count of steps with activity in each column."""
    cols = np.asarray(cols, dtype=np.int32).ravel()
    n = np.clip(cols, 0, 255)
    d0 = (n // 64) % 4
    d1 = (n // 16) % 4
    d2 = (n // 4) % 4
    d3 = n % 4
    c0 = np.sum((d0 >= 1) & (d0 <= 3))
    c1 = np.sum((d1 >= 1) & (d1 <= 3))
    c2 = np.sum((d2 >= 1) & (d2 <= 3))
    c3 = np.sum((d3 >= 1) & (d3 <= 3))
    return np.array([c0, c1, c2, c3], dtype=np.int64)


def _cramers_v(contingency: np.ndarray) -> float:
    """Cramér's V for a contingency table (rows x cols). Returns 0 if undefined."""
    C = np.asarray(contingency, dtype=np.float64)
    n = C.sum()
    if n == 0:
        return 0.0
    r, c = C.shape
    if r <= 1 or c <= 1:
        return 0.0
    row_sum = C.sum(axis=1, keepdims=True)
    col_sum = C.sum(axis=0, keepdims=True)
    E = row_sum * col_sum / n
    E[E == 0] = 1e-10
    chi2 = np.sum((C - E) ** 2 / E)
    min_dim = min(r - 1, c - 1)
    if min_dim <= 0:
        return 0.0
    return float(np.sqrt(chi2 / (n * min_dim)))


def _chord_sizes(cols: np.ndarray) -> np.ndarray:
    """Return 1d array of chord sizes (1-4) for each non-empty step."""
    cols = np.asarray(cols, dtype=np.int32).ravel()
    sizes = []
    for n in cols:
        n = int(np.clip(n, 0, 255))
        if n == 0:
            continue
        d0 = (n // 64) % 4
        d1 = (n // 16) % 4
        d2 = (n // 4) % 4
        d3 = n % 4
        count = sum(1 for d in (d0, d1, d2, d3) if d in (1, 2, 3))
        if count >= 1:
            sizes.append(min(count, 4))
    return np.array(sizes, dtype=np.int32) if sizes else np.array([], dtype=np.int32)


def collect_aggregates(data_dir: str):
    """Load all charts, aggregate arrow codes, note kinds, chart validity, etc."""
    pairs = datasets._load_and_pair_files(data_dir)  # noqa: SLF001
    if not pairs:
        return None

    note_kind_lookup = _build_note_kind_lookup()
    all_arrow_codes: list[int] = []
    all_note_kinds: list[int] = []
    steps_per_chart: list[int] = []
    bpms: list[float] = []
    violation_counts: list[int] = []
    valid_charts = 0
    invalid_charts = 0
    column_activity = np.zeros(4, dtype=np.int64)
    all_chord_sizes: list[int] = []
    chart_durations: list[float] = []
    all_inter_step_intervals: list[float] = []
    steps_per_second: list[float] = []
    per_step_time_norm: list[float] = []
    per_step_arrow_code: list[int] = []
    per_step_note_kind: list[int] = []
    per_step_interval: list[float] = []
    per_step_chord_size: list[int] = []

    for _audio_path, chart_path in pairs:
        try:
            times, cols = datasets._parse_step_chart(  # noqa: SLF001
                chart_path, binary_timings=False
            )
        except Exception:
            continue
        times = np.asarray(times, dtype=np.float64)
        cols_flat = np.asarray(cols, dtype=np.int32).ravel()
        n_steps = len(cols_flat)
        steps_per_chart.append(n_steps)

        if n_steps >= 2:
            duration = abs(float(times[-1] - times[0]))
            chart_durations.append(duration)
            all_inter_step_intervals.extend(np.diff(times).tolist())
            steps_per_second.append(n_steps / duration if duration > 0 else 0.0)
        else:
            chart_durations.append(0.0)
            steps_per_second.append(0.0)

        bpm = _parse_bpm(chart_path)
        if bpm is not None:
            bpms.append(bpm)

        violations, _hold_ends, _examples = metrics.compute_chart_validity_violations(
            cols
        )
        violation_counts.append(violations)
        if violations > 0:
            invalid_charts += 1
        else:
            valid_charts += 1

        all_arrow_codes.extend(cols_flat.tolist())
        kinds = note_kind_lookup[np.clip(cols_flat, 0, 255)]
        all_note_kinds.extend(kinds.tolist())

        column_activity += _column_activity_vectorized(cols_flat)
        chord_sz = _chord_sizes(cols_flat)
        all_chord_sizes.extend(chord_sz.tolist())

        # Per-step arrays for timing vs arrow correlation
        max_t = float(np.max(times)) + 1e-9
        time_norm = times / max_t
        per_step_time_norm.extend(time_norm.tolist())
        per_step_arrow_code.extend(cols_flat.tolist())
        per_step_note_kind.extend(kinds.tolist())
        if n_steps > 0:
            intervals = [np.nan] + (times[1:] - times[:-1]).tolist()
            per_step_interval.extend(intervals)
        for n in cols_flat:
            n = int(np.clip(n, 0, 255))
            if n == 0:
                per_step_chord_size.append(0)
            else:
                d0, d1, d2, d3 = (n // 64) % 4, (n // 16) % 4, (n // 4) % 4, n % 4
                count = sum(1 for d in (d0, d1, d2, d3) if d in (1, 2, 3))
                per_step_chord_size.append(min(count, 4))

    return {
        "n_charts": len(pairs),
        "all_arrow_codes": np.array(all_arrow_codes, dtype=np.int32),
        "all_note_kinds": np.array(all_note_kinds, dtype=np.int32),
        "steps_per_chart": np.array(steps_per_chart, dtype=np.int64),
        "bpms": np.array(bpms, dtype=np.float64) if bpms else np.array([]),
        "violation_counts": np.array(violation_counts, dtype=np.int32),
        "valid_charts": valid_charts,
        "invalid_charts": invalid_charts,
        "column_activity": column_activity,
        "all_chord_sizes": np.array(all_chord_sizes, dtype=np.int32),
        "chart_durations": np.array(chart_durations, dtype=np.float64),
        "inter_step_intervals": np.array(all_inter_step_intervals, dtype=np.float64),
        "steps_per_second": np.array(steps_per_second, dtype=np.float64),
        "per_step_time_norm": np.array(per_step_time_norm, dtype=np.float64),
        "per_step_arrow_code": np.array(per_step_arrow_code, dtype=np.int32),
        "per_step_note_kind": np.array(per_step_note_kind, dtype=np.int32),
        "per_step_interval": np.array(per_step_interval, dtype=np.float64),
        "per_step_chord_size": np.array(per_step_chord_size, dtype=np.int32),
    }


def write_summary(agg: dict, output_path: str | None) -> str:
    """Build summary text and optionally write to file. Returns summary string."""
    n_parsed = agg["valid_charts"] + agg["invalid_charts"]
    total_steps = int(np.sum(agg["steps_per_chart"]))
    spc = agg["steps_per_chart"]
    viol_pct = (
        f"{100 * agg['invalid_charts'] / n_parsed:.1f}%" if n_parsed > 0 else "N/A"
    )
    if n_parsed > 0 and spc.size > 0:
        steps_line = f"Steps per chart: min={int(spc.min())}, mean={float(spc.mean()):.1f}, max={int(spc.max())}"
    else:
        steps_line = "Steps per chart: N/A (no charts successfully parsed)"
    lines = [
        "Arrow training data summary",
        "=" * 40,
        f"Charts (parsed): {n_parsed}",
        f"Total steps: {total_steps}",
        steps_line,
        f"Charts with hold violations: {agg['invalid_charts']} ({viol_pct})",
    ]
    vc = agg["violation_counts"]
    with_viol = vc[vc > 0]
    if len(with_viol) > 0:
        lines.append(
            f"Mean violations per chart (among those with violations): {float(with_viol.mean()):.1f}"
        )

    cd = agg["chart_durations"]
    isi = agg["inter_step_intervals"]
    sps = agg["steps_per_second"]
    if cd.size > 0:
        lines.append(
            f"Chart duration (s): min={float(cd.min()):.1f}, mean={float(cd.mean()):.1f}, max={float(cd.max()):.1f}"
        )
    else:
        lines.append("Chart duration (s): N/A")
    if isi.size > 0:
        lines.append(
            f"Mean step interval (s): {float(isi.mean()):.3f}; "
            f"inter-step min={float(isi.min()):.3f}, max={float(isi.max()):.3f}"
        )
    else:
        lines.append("Mean step interval (s): N/A")
    if sps.size > 0:
        lines.append(
            f"Steps per second: min={float(sps.min()):.2f}, mean={float(sps.mean()):.2f}, max={float(sps.max()):.2f}"
        )
    else:
        lines.append("Steps per second: N/A")

    # Correlation summary: timing vs arrow type (note kind)
    lines.append("")
    lines.append("Correlation summary (timing vs note kind)")
    lines.append("-" * 40)
    t_norm = agg.get("per_step_time_norm")
    kind = agg.get("per_step_note_kind")
    interval = agg.get("per_step_interval")
    if t_norm is not None and kind is not None and t_norm.size > 0:
        time_bin_idx = np.clip(
            np.searchsorted(TIME_BIN_EDGES[1:], t_norm, side="right"),
            0,
            N_TIME_BINS - 1,
        )
        n_kinds = len(NOTE_KIND_LABELS)
        cont_time = np.zeros((n_kinds, N_TIME_BINS), dtype=np.int64)
        for i in range(len(t_norm)):
            kb = int(np.clip(kind[i], 0, n_kinds - 1))
            tb = int(time_bin_idx[i])
            cont_time[kb, tb] += 1
        v_time = _cramers_v(cont_time)
        lines.append(f"Cramér's V (time bin x note kind): {v_time:.4f}")
        lines.append("Counts (rows=note kind, cols=time bin 0=start .. 9=end):")
        label_w = max(len(lbl) for lbl in NOTE_KIND_LABELS)
        col_w = 8
        sep = " "
        header_labels = [f"{i}" for i in range(N_TIME_BINS)]
        prefix = "  " + " " * label_w + " "
        header_cells = [f"{h:>{col_w}s}" for h in header_labels] + [
            f"{'Total':>{col_w}s}"
        ]
        header = prefix + sep.join(header_cells)
        n_cols = len(header_cells)
        line_len = len(prefix) + n_cols * col_w + (n_cols - 1) * len(sep)
        lines.append(header)
        lines.append(prefix + "-" * (line_len - len(prefix)))
        for k in range(n_kinds):
            row_vals = [cont_time[k, b] for b in range(N_TIME_BINS)]
            total = sum(row_vals)
            cells = [f"{v:{col_w},d}" for v in row_vals] + [f"{total:{col_w},d}"]
            row_str = sep.join(cells)
            lines.append(f"  {NOTE_KIND_LABELS[k]:<{label_w}s} {row_str}")
        col_totals = [cont_time[:, b].sum() for b in range(N_TIME_BINS)]
        grand = sum(col_totals)
        total_cells = [f"{t:{col_w},d}" for t in col_totals] + [f"{grand:{col_w},d}"]
        total_row = sep.join(total_cells)
        lines.append(prefix + "-" * (line_len - len(prefix)))
        lines.append(f"  {'Total':<{label_w}s} {total_row}")
    else:
        lines.append("Cramér's V (time bin x note kind): N/A (no per-step data)")
    if interval is not None and kind is not None:
        valid = ~np.isnan(interval)
        if valid.sum() > 0:
            iv = interval[valid]
            kv = kind[valid]
            bin_idx = np.searchsorted(INTERVAL_BIN_EDGES[1:-1], iv, side="right")
            bin_idx = np.clip(bin_idx, 0, N_INTERVAL_BINS - 1)
            n_kinds = len(NOTE_KIND_LABELS)
            cont_interval = np.zeros((n_kinds, N_INTERVAL_BINS), dtype=np.int64)
            for i in range(len(iv)):
                kb = int(np.clip(kv[i], 0, n_kinds - 1))
                bb = int(bin_idx[i])
                cont_interval[kb, bb] += 1
            v_interval = _cramers_v(cont_interval)
            lines.append(f"Cramér's V (interval bin x note kind): {v_interval:.4f}")
        else:
            lines.append("Cramér's V (interval bin x note kind): N/A (no intervals)")
    else:
        lines.append("Cramér's V (interval bin x note kind): N/A")

    # Interpretation guide
    lines.append("")
    lines.append("How to interpret these results")
    lines.append("-" * 40)
    lines.append(
        "Cramér's V: 0 = no association between the two variables; 1 = perfect."
    )
    lines.append("Rough guide: <0.1 weak, 0.1–0.3 moderate, >0.3 strong.")
    lines.append("")
    lines.append(
        "Time bin x note kind: Does *where* in the song (start/middle/end) relate to"
    )
    lines.append(
        "note type? Low V means note types are spread similarly across the song."
    )
    lines.append(
        "High V means certain note types (e.g. holds, chords) cluster in some regions."
    )
    lines.append("")
    lines.append(
        "Interval bin x note kind: Does the *gap* since the previous step relate to"
    )
    lines.append(
        "the next note type? Low V means gap length doesn't predict single vs chord vs hold."
    )
    lines.append(
        "High V means e.g. long gaps tend to be followed by chords/holds more than singles."
    )
    lines.append("")
    lines.append("Counts table: Rows = note kind (Single, Chord, Hold start/end/both).")
    lines.append("Cols = time bin 0 (start of song) .. 9 (end). Compare row totals and")
    lines.append("column patterns to see where each note type appears most.")
    lines.append("")
    lines.append(
        "Plots: time_bin_vs_note_kind = stacked view of the table; interval_vs_*"
    )
    lines.append(
        "= heatmaps of gap length vs note type/chord size; time_vs_note_kind_strip"
    )
    lines.append("= scatter of position vs note kind.")

    summary = "\n".join(lines)
    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as f:
            f.write(summary)
            f.write("\n")
    return summary


def plot_note_kind(agg: dict, output_dir: str | None, show: bool) -> None:
    """Bar chart of note kind distribution."""
    counts = np.bincount(agg["all_note_kinds"], minlength=len(NOTE_KIND_LABELS))[
        : len(NOTE_KIND_LABELS)
    ]
    fig, ax = plt.subplots()
    ax.bar(NOTE_KIND_LABELS, counts, color="steelblue", edgecolor="black")
    ax.set_ylabel("Count")
    ax.set_title("Note kind distribution")
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    if output_dir:
        fig.savefig(os.path.join(output_dir, "note_kind_dist.png"), dpi=150)
    if show:
        plt.show()
    plt.close()


def plot_top_arrow_types(
    agg: dict, top_n: int, output_dir: str | None, show: bool
) -> None:
    """Bar chart of top N arrow types (base-4 labels) + Other."""
    codes = agg["all_arrow_codes"]
    codes = codes[codes > 0]  # exclude empty
    if codes.size == 0:
        return
    unique, cnt = np.unique(codes, return_counts=True)
    order = np.argsort(-cnt)
    top_codes = unique[order[:top_n]]
    top_counts = cnt[order[:top_n]]
    other_count = int(cnt[order[top_n:]].sum()) if len(order) > top_n else 0
    labels = [
        generator._int_to_base4_string(int(c), min_digits=4)  # noqa: SLF001
        for c in top_codes
    ]
    if other_count > 0:
        labels.append("Other")
        top_counts = np.append(top_counts, other_count)
    fig, ax = plt.subplots()
    ax.bar(range(len(labels)), top_counts, color="steelblue", edgecolor="black")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Count")
    ax.set_title(f"Top {top_n} arrow types")
    fig.tight_layout()
    if output_dir:
        fig.savefig(os.path.join(output_dir, "top_arrow_types.png"), dpi=150)
    if show:
        plt.show()
    plt.close()


def plot_per_column_activity(agg: dict, output_dir: str | None, show: bool) -> None:
    """Bar chart of per-column activity."""
    fig, ax = plt.subplots()
    ax.bar(
        COLUMN_LABELS,
        agg["column_activity"],
        color="steelblue",
        edgecolor="black",
    )
    ax.set_ylabel("Count (steps with activity)")
    ax.set_title("Per-column activity")
    fig.tight_layout()
    if output_dir:
        fig.savefig(os.path.join(output_dir, "per_column_activity.png"), dpi=150)
    if show:
        plt.show()
    plt.close()


def plot_chart_validity(agg: dict, output_dir: str | None, show: bool) -> None:
    """Bar chart: valid vs invalid charts."""
    fig, ax = plt.subplots()
    ax.bar(
        ["Valid", "With violations"],
        [agg["valid_charts"], agg["invalid_charts"]],
        color=["green", "coral"],
        edgecolor="black",
    )
    ax.set_ylabel("Number of charts")
    ax.set_title("Chart validity")
    fig.tight_layout()
    if output_dir:
        fig.savefig(os.path.join(output_dir, "chart_validity.png"), dpi=150)
    if show:
        plt.show()
    plt.close()


def plot_steps_per_chart(agg: dict, output_dir: str | None, show: bool) -> None:
    """Histogram of steps per chart."""
    fig, ax = plt.subplots()
    n_plotted = len(agg["steps_per_chart"])
    ax.hist(
        agg["steps_per_chart"],
        bins=min(50, max(1, n_plotted)),
        color="steelblue",
        edgecolor="black",
    )
    ax.set_xlabel("Steps per chart")
    ax.set_ylabel("Number of charts")
    ax.set_title("Steps per chart distribution")
    fig.tight_layout()
    if output_dir:
        fig.savefig(os.path.join(output_dir, "steps_per_chart.png"), dpi=150)
    if show:
        plt.show()
    plt.close()


def plot_chord_size(agg: dict, output_dir: str | None, show: bool) -> None:
    """Histogram of chord sizes (1-4) on non-empty steps."""
    sz = agg["all_chord_sizes"]
    if sz.size == 0:
        return
    fig, ax = plt.subplots()
    ax.hist(
        sz,
        bins=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5],
        rwidth=0.8,
        color="steelblue",
        edgecolor="black",
    )
    ax.set_xticks([1, 2, 3, 4])
    ax.set_xlabel("Chord size")
    ax.set_ylabel("Count")
    ax.set_title("Chord size (non-empty steps)")
    fig.tight_layout()
    if output_dir:
        fig.savefig(os.path.join(output_dir, "chord_size.png"), dpi=150)
    if show:
        plt.show()
    plt.close()


def plot_bpm(agg: dict, output_dir: str | None, show: bool) -> None:
    """Histogram of BPM (if parsed)."""
    bpms = agg["bpms"]
    if bpms.size == 0:
        return
    fig, ax = plt.subplots()
    ax.hist(bpms, bins=min(40, max(1, len(bpms))), color="steelblue", edgecolor="black")
    ax.set_xlabel("BPM")
    ax.set_ylabel("Number of charts")
    ax.set_title("BPM distribution")
    fig.tight_layout()
    if output_dir:
        fig.savefig(os.path.join(output_dir, "bpm.png"), dpi=150)
    if show:
        plt.show()
    plt.close()


def plot_chart_duration(agg: dict, output_dir: str | None, show: bool) -> None:
    """Histogram of chart duration (seconds)."""
    cd = agg["chart_durations"]
    if cd.size == 0:
        return
    fig, ax = plt.subplots()
    n = len(cd)
    ax.hist(
        cd,
        bins=min(40, max(1, n)),
        color="steelblue",
        edgecolor="black",
    )
    ax.set_xlabel("Chart duration (s)")
    ax.set_ylabel("Number of charts")
    ax.set_title("Chart duration distribution")
    fig.tight_layout()
    if output_dir:
        fig.savefig(os.path.join(output_dir, "chart_duration.png"), dpi=150)
    if show:
        plt.show()
    plt.close()


def plot_inter_step_interval(agg: dict, output_dir: str | None, show: bool) -> None:
    """Histogram of inter-step intervals (seconds)."""
    isi = agg["inter_step_intervals"]
    if isi.size == 0:
        return
    fig, ax = plt.subplots()
    n = len(isi)
    ax.hist(
        isi,
        bins=min(50, max(1, n)),
        color="steelblue",
        edgecolor="black",
    )
    ax.set_xlabel("Inter-step interval (s)")
    ax.set_ylabel("Count")
    ax.set_title("Inter-step interval distribution")
    fig.tight_layout()
    if output_dir:
        fig.savefig(os.path.join(output_dir, "inter_step_interval.png"), dpi=150)
    if show:
        plt.show()
    plt.close()


def plot_steps_per_second(agg: dict, output_dir: str | None, show: bool) -> None:
    """Histogram of steps per second (per chart)."""
    sps = agg["steps_per_second"]
    if sps.size == 0:
        return
    fig, ax = plt.subplots()
    n = len(sps)
    ax.hist(
        sps,
        bins=min(40, max(1, n)),
        color="steelblue",
        edgecolor="black",
    )
    ax.set_xlabel("Steps per second")
    ax.set_ylabel("Number of charts")
    ax.set_title("Steps per second distribution")
    fig.tight_layout()
    if output_dir:
        fig.savefig(os.path.join(output_dir, "steps_per_second.png"), dpi=150)
    if show:
        plt.show()
    plt.close()


def plot_time_bin_vs_note_kind(agg: dict, output_dir: str | None, show: bool) -> None:
    """Stacked bar or heatmap: normalized time bin vs note kind counts."""
    t_norm = agg["per_step_time_norm"]
    kind = agg["per_step_note_kind"]
    if t_norm.size == 0:
        return
    time_bin_idx = np.clip(
        np.searchsorted(TIME_BIN_EDGES[1:], t_norm, side="right"),
        0,
        N_TIME_BINS - 1,
    )
    n_kinds = len(NOTE_KIND_LABELS)
    counts = np.zeros((n_kinds, N_TIME_BINS), dtype=np.int64)
    for i in range(len(t_norm)):
        kb = int(np.clip(kind[i], 0, n_kinds - 1))
        tb = int(time_bin_idx[i])
        counts[kb, tb] += 1
    fig, ax = plt.subplots()
    bin_centers = (TIME_BIN_EDGES[:-1] + TIME_BIN_EDGES[1:]) / 2
    bottom = np.zeros(N_TIME_BINS)
    colors = plt.get_cmap("tab10")(np.linspace(0, 1, n_kinds))
    for k in range(n_kinds):
        ax.bar(
            bin_centers,
            counts[k],
            width=0.08,
            bottom=bottom,
            label=NOTE_KIND_LABELS[k],
            color=colors[k],
            edgecolor="black",
        )
        bottom += counts[k]
    ax.set_xlabel("Normalized time (song position)")
    ax.set_ylabel("Count")
    ax.set_title("Time bin vs note kind")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_xlim(-0.05, 1.05)
    fig.tight_layout()
    if output_dir:
        fig.savefig(os.path.join(output_dir, "time_bin_vs_note_kind.png"), dpi=150)
    if show:
        plt.show()
    plt.close()


def plot_interval_vs_note_kind(agg: dict, output_dir: str | None, show: bool) -> None:
    """Heatmap or counts: inter-step interval bin vs note kind of following step."""
    interval = agg["per_step_interval"]
    kind = agg["per_step_note_kind"]
    valid = ~np.isnan(interval)
    if valid.sum() == 0:
        return
    iv = interval[valid]
    kv = kind[valid]
    bin_idx = np.searchsorted(INTERVAL_BIN_EDGES[1:-1], iv, side="right")
    bin_idx = np.clip(bin_idx, 0, N_INTERVAL_BINS - 1)
    n_kinds = len(NOTE_KIND_LABELS)
    counts = np.zeros((n_kinds, N_INTERVAL_BINS), dtype=np.int64)
    for i in range(len(iv)):
        kb = int(np.clip(kv[i], 0, n_kinds - 1))
        bb = int(bin_idx[i])
        counts[kb, bb] += 1
    fig, ax = plt.subplots()
    im = ax.imshow(
        counts,
        aspect="auto",
        origin="lower",
        cmap="Blues",
    )
    ax.set_xticks(range(N_INTERVAL_BINS))
    interval_labels = [
        (
            f"{INTERVAL_BIN_EDGES[j]:.1f}-{INTERVAL_BIN_EDGES[j + 1]:.1f}"
            if np.isfinite(INTERVAL_BIN_EDGES[j + 1])
            else f"{INTERVAL_BIN_EDGES[j]:.1f}+"
        )
        for j in range(N_INTERVAL_BINS)
    ]
    ax.set_xticklabels(interval_labels, rotation=45, ha="right")
    ax.set_yticks(range(n_kinds))
    ax.set_yticklabels(NOTE_KIND_LABELS)
    ax.set_xlabel("Inter-step interval (s)")
    ax.set_ylabel("Note kind (following step)")
    ax.set_title("Interval vs note kind")
    plt.colorbar(im, ax=ax, label="Count")
    fig.tight_layout()
    if output_dir:
        fig.savefig(os.path.join(output_dir, "interval_vs_note_kind.png"), dpi=150)
    if show:
        plt.show()
    plt.close()


def plot_time_bin_vs_chord_size(agg: dict, output_dir: str | None, show: bool) -> None:
    """Among chord/hold steps, distribution of chord size per time bin."""
    t_norm = agg["per_step_time_norm"]
    csize = agg["per_step_chord_size"]
    mask = csize >= 1
    if mask.sum() == 0:
        return
    t_norm = t_norm[mask]
    csize = csize[mask]
    time_bin_idx = np.clip(
        np.searchsorted(TIME_BIN_EDGES[1:], t_norm, side="right"),
        0,
        N_TIME_BINS - 1,
    )
    counts = np.zeros((4, N_TIME_BINS), dtype=np.int64)
    for i in range(len(t_norm)):
        sz = int(np.clip(csize[i], 1, 4)) - 1
        tb = int(time_bin_idx[i])
        counts[sz, tb] += 1
    fig, ax = plt.subplots()
    x = np.arange(N_TIME_BINS)
    width = 0.2
    for sz in range(4):
        ax.bar(
            x + (sz - 1.5) * width,
            counts[sz],
            width=width,
            label=f"Chord size {sz + 1}",
        )
    ax.set_xticks(x)
    ax.set_xticklabels([f"{TIME_BIN_EDGES[i]:.1f}" for i in range(N_TIME_BINS)])
    ax.set_xlabel("Normalized time bin")
    ax.set_ylabel("Count")
    ax.set_title("Time bin vs chord size (non-empty steps)")
    ax.legend()
    fig.tight_layout()
    if output_dir:
        fig.savefig(os.path.join(output_dir, "time_bin_vs_chord_size.png"), dpi=150)
    if show:
        plt.show()
    plt.close()


def plot_interval_vs_chord_size(agg: dict, output_dir: str | None, show: bool) -> None:
    """Among chord/hold steps, distribution of chord size per interval bin."""
    interval = agg["per_step_interval"]
    csize = agg["per_step_chord_size"]
    valid = ~np.isnan(interval) & (csize >= 1)
    if valid.sum() == 0:
        return
    iv = interval[valid]
    cv = csize[valid]
    bin_idx = np.searchsorted(INTERVAL_BIN_EDGES[1:-1], iv, side="right")
    bin_idx = np.clip(bin_idx, 0, N_INTERVAL_BINS - 1)
    counts = np.zeros((4, N_INTERVAL_BINS), dtype=np.int64)
    for i in range(len(iv)):
        sz = int(np.clip(cv[i], 1, 4)) - 1
        bb = int(bin_idx[i])
        counts[sz, bb] += 1
    fig, ax = plt.subplots()
    im = ax.imshow(
        counts,
        aspect="auto",
        origin="lower",
        cmap="Greens",
    )
    ax.set_xticks(range(N_INTERVAL_BINS))
    interval_labels = [
        (
            f"{INTERVAL_BIN_EDGES[j]:.1f}-{INTERVAL_BIN_EDGES[j + 1]:.1f}"
            if np.isfinite(INTERVAL_BIN_EDGES[j + 1])
            else f"{INTERVAL_BIN_EDGES[j]:.1f}+"
        )
        for j in range(N_INTERVAL_BINS)
    ]
    ax.set_xticklabels(interval_labels, rotation=45, ha="right")
    ax.set_yticks(range(4))
    ax.set_yticklabels(["1", "2", "3", "4"])
    ax.set_ylabel("Chord size")
    ax.set_xlabel("Inter-step interval (s)")
    ax.set_title("Interval vs chord size (non-empty steps)")
    plt.colorbar(im, ax=ax, label="Count")
    fig.tight_layout()
    if output_dir:
        fig.savefig(os.path.join(output_dir, "interval_vs_chord_size.png"), dpi=150)
    if show:
        plt.show()
    plt.close()


def plot_time_vs_note_kind_strip(agg: dict, output_dir: str | None, show: bool) -> None:
    """Strip or violin: normalized time vs note kind."""
    t_norm = agg["per_step_time_norm"]
    kind = agg["per_step_note_kind"]
    if t_norm.size == 0:
        return
    fig, ax = plt.subplots()
    n_kinds = len(NOTE_KIND_LABELS)
    for k in range(n_kinds):
        mask = kind == k
        if mask.sum() == 0:
            continue
        x = t_norm[mask] + np.random.uniform(-0.008, 0.008, size=mask.sum())
        y = np.full(mask.sum(), k)
        ax.scatter(x, y, alpha=0.3, s=5, label=NOTE_KIND_LABELS[k])
    ax.set_xlabel("Normalized time (song position)")
    ax.set_ylabel("Note kind")
    ax.set_yticks(range(n_kinds))
    ax.set_yticklabels(NOTE_KIND_LABELS)
    ax.set_title("Normalized time vs note kind (strip)")
    ax.set_ylim(-0.5, n_kinds - 0.5)
    ax.set_xlim(-0.02, 1.02)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    if output_dir:
        fig.savefig(os.path.join(output_dir, "time_vs_note_kind_strip.png"), dpi=150)
    if show:
        plt.show()
    plt.close()


def main() -> int:
    args = PARSER.parse_args()

    agg = collect_aggregates(args.data_dir)
    if agg is None:
        print(f"No audio-chart pairs found in {args.data_dir!r}.", file=sys.stderr)
        return 2

    show = not args.no_show
    out = args.output_dir
    if out:
        os.makedirs(out, exist_ok=True)

    summary = write_summary(
        agg,
        os.path.join(out, "summary.txt") if out else None,
    )
    print(summary)

    plot_note_kind(agg, out, show)
    plot_top_arrow_types(agg, args.top_arrows, out, show)
    plot_per_column_activity(agg, out, show)
    plot_chart_validity(agg, out, show)
    plot_steps_per_chart(agg, out, show)
    plot_chord_size(agg, out, show)
    plot_bpm(agg, out, show)
    plot_chart_duration(agg, out, show)
    plot_inter_step_interval(agg, out, show)
    plot_steps_per_second(agg, out, show)
    plot_time_bin_vs_note_kind(agg, out, show)
    plot_interval_vs_note_kind(agg, out, show)
    plot_time_bin_vs_chord_size(agg, out, show)
    plot_interval_vs_chord_size(agg, out, show)
    plot_time_vs_note_kind_strip(agg, out, show)

    return 0


if __name__ == "__main__":
    sys.exit(main())
