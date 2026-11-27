"""Machine-readable experiment scoreboard for ITGPT onset placement runs."""

from __future__ import annotations

import dataclasses
import json
import pathlib
import typing

DEFAULT_SEARCH_ROOTS: tuple[str, ...] = (
    "experiments/colab",
    "models_wsl/ddc",
)
VALID_SORT_KEYS: tuple[str, ...] = (
    "f1_at_05",
    "f1_max",
    "micro_event_f1",
    "total_errors",
    "started_utc",
    "run_name",
)
VALID_WEIGHT_MODES: tuple[str, ...] = ("auto", "best", "best_f1", "last")

MIN_HEALTHY_SHUFFLE_DROP = 0.08
MIN_HEALTHY_ZEROS_DROP = 0.10
MAX_INTRO_REGRESSION = 0.03
MAX_16TH_REGRESSION = 0.015


@dataclasses.dataclass(frozen=True)
class ScoreboardRow:
    """Normalized summary of one ITGPT placement evaluation directory.

    Attributes:
        run_name: Directory basename identifying the run.
        source: Short origin tag (``colab``, ``wsl``, or custom root name).
        run_dir: Path to the run artifact directory.
        weights: Checkpoint variant scored (``best``, ``best_f1``, ``last``, etc.).
        framework: Recorded framework (``pytorch``, ``keras``, or ``""``).
        f1_at_05: Pooled ``M-slot48`` F1 at threshold 0.5.
        f1_max: Best ``M-slot48`` F1 across thresholds.
        best_threshold: Threshold achieving ``f1_max``.
        n_charts: Evaluated chart count.
        n_beats: Evaluated beat count.
        micro_event_f1: Decoded ±20 ms micro event F1 when grounding eval exists.
        skill_event_f1: Optional decoded ±20 ms event skill F1.
        slot_f1_drop_shuffle: ``none - shuffle`` slot F1 drop from grounding eval.
        slot_f1_drop_zeros: ``none - zeros`` slot F1 drop from grounding eval.
        f1_16th: Off-beat 16th-note subdivision F1 from error breakdown.
        f1_24th: Triplet / 24th subdivision F1 from error breakdown.
        f1_intro: Song intro (0–10%) F1 from error breakdown.
        total_errors: Total binary slot errors ``fp + fn`` (breakdown or slot48).
        gpu: GPU string from ``run_meta.json`` when available.
        started_utc: ISO-8601 start timestamp from ``run_meta.json`` when available.
    """

    run_name: str
    source: str
    run_dir: str
    weights: str
    framework: str
    f1_at_05: float
    f1_max: float
    best_threshold: float
    n_charts: int
    n_beats: int
    micro_event_f1: float | None = None
    skill_event_f1: float | None = None
    slot_f1_drop_shuffle: float | None = None
    slot_f1_drop_zeros: float | None = None
    f1_16th: float | None = None
    f1_24th: float | None = None
    f1_intro: float | None = None
    total_errors: int | None = None
    gpu: str = ""
    started_utc: str = ""

    def as_dict(self) -> dict[str, object]:
        """Serialize scoreboard row to a JSON-compatible dictionary.

        Returns:
            Dictionary of row fields.
        """
        return dataclasses.asdict(self)


def _read_json(path: pathlib.Path) -> dict[str, typing.Any] | None:
    """Read a JSON object from ``path``, returning ``None`` on missing or bad JSON.

    Args:
        path: Candidate JSON file path.

    Returns:
        Parsed dictionary or ``None``.
    """
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _as_float(value: object) -> float | None:
    """Convert numeric JSON value to ``float`` if possible.

    Args:
        value: Raw JSON value.

    Returns:
        Float value or ``None``.
    """
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def _as_int(value: object) -> int | None:
    """Convert integer JSON value to ``int`` if possible.

    Args:
        value: Raw JSON value.

    Returns:
        Integer value or ``None``.
    """
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return None


def _source_tag(root: pathlib.Path) -> str:
    """Derive a compact source label from a root directory path.

    Args:
        root: Search root directory.

    Returns:
        Short tag such as ``"colab"`` or ``"wsl"``.
    """
    joined = root.as_posix().lower()
    if "colab" in joined:
        return "colab"
    if "models_wsl" in joined or "wsl" in joined:
        return "wsl"
    return root.name or "local"


def _pick_slot48_file(
    run_dir: pathlib.Path,
    *,
    weights: str,
) -> tuple[pathlib.Path, str] | None:
    """Select the primary ``eval_val_slot48*.json`` file inside ``run_dir``.

    Args:
        run_dir: Candidate run directory.
        weights: Selection policy (``"auto"``, ``"best"``, ``"best_f1"``, ``"last"``).

    Returns:
        Tuple ``(path, weight_label)`` or ``None`` if no matching file exists.
    """
    candidates_by_mode: dict[str, tuple[tuple[str, str], ...]] = {
        "auto": (
            ("eval_val_slot48_best.json", "best"),
            ("eval_val_slot48_best_f1.json", "best_f1"),
            ("eval_val_slot48.json", "last"),
        ),
        "best": (("eval_val_slot48_best.json", "best"),),
        "best_f1": (("eval_val_slot48_best_f1.json", "best_f1"),),
        "last": (("eval_val_slot48.json", "last"),),
    }
    for filename, label in candidates_by_mode[weights]:
        candidate = run_dir / filename
        if candidate.is_file():
            return candidate, label
    return None


def _pick_first_existing(
    run_dir: pathlib.Path,
    filenames: tuple[str, ...],
) -> pathlib.Path | None:
    """Return the first existing file among ``filenames`` in ``run_dir``.

    Args:
        run_dir: Candidate run directory.
        filenames: Ordered candidate filenames.

    Returns:
        First existing path, or ``None``.
    """
    for filename in filenames:
        candidate = run_dir / filename
        if candidate.is_file():
            return candidate
    return None


def _grounding_candidates(weights: str) -> tuple[str, ...]:
    """Return ordered grounding JSON filenames matching ``weights``.

    Args:
        weights: Selected slot48 checkpoint label.

    Returns:
        Tuple of candidate filenames in priority order.
    """
    if weights == "best_f1":
        return (
            "eval_val_grounding_best_f1.json",
            "eval_val_grounding.json",
            "eval_val_grounding_best.json",
        )
    if weights == "last":
        return (
            "eval_val_grounding_last.json",
            "eval_val_grounding.json",
            "eval_val_grounding_best.json",
        )
    return (
        "eval_val_grounding_best.json",
        "eval_val_grounding.json",
        "eval_val_grounding_best_f1.json",
        "eval_val_grounding_last.json",
    )


def _breakdown_candidates(weights: str) -> tuple[str, ...]:
    """Return ordered error-breakdown JSON filenames matching ``weights``.

    Args:
        weights: Selected slot48 checkpoint label.

    Returns:
        Tuple of candidate filenames in priority order.
    """
    if weights == "best_f1":
        return (
            "eval_val_breakdown_best_f1.json",
            "eval_val_breakdown.json",
            "eval_val_breakdown_best.json",
        )
    return (
        "eval_val_breakdown_best.json",
        "eval_val_breakdown.json",
        "eval_val_breakdown_best_f1.json",
    )


def _load_row(
    run_dir: pathlib.Path,
    *,
    source: str,
    weights: str,
) -> ScoreboardRow | None:
    """Load and merge slot48, grounding, breakdown, and meta JSONs for one run.

    Args:
        run_dir: Run directory containing evaluation JSON outputs.
        source: Short origin tag (e.g. ``"colab"`` or ``"wsl"``).
        weights: Weight selection mode.

    Returns:
        Populated :class:`ScoreboardRow`, or ``None`` if no valid eval exists.
    """
    slot_pick = _pick_slot48_file(run_dir, weights=weights)
    if slot_pick is None:
        return None
    slot_path, picked_weights = slot_pick
    slot_data = _read_json(slot_path)
    if slot_data is None:
        return None

    f1_at_05 = _as_float(slot_data.get("f1_at_05"))
    f1_max = _as_float(slot_data.get("f1_max"))
    if f1_at_05 is None or f1_max is None:
        return None

    fp = _as_int(slot_data.get("false_positives"))
    fn = _as_int(slot_data.get("false_negatives"))
    total_errors = (fp + fn) if (fp is not None and fn is not None) else None

    grounding_path = _pick_first_existing(
        run_dir, _grounding_candidates(picked_weights)
    )
    grounding = _read_json(grounding_path) if grounding_path else None
    micro_event_f1: float | None = None
    skill_event_f1: float | None = None
    drop_shuffle: float | None = None
    drop_zeros: float | None = None
    if grounding is not None:
        drop_shuffle = _as_float(grounding.get("slot_f1_drop_shuffle"))
        drop_zeros = _as_float(grounding.get("slot_f1_drop_zeros"))
        conditions = grounding.get("conditions")
        if isinstance(conditions, dict):
            none_cond = conditions.get("none")
            if isinstance(none_cond, dict):
                micro_event_f1 = _as_float(none_cond.get("micro_event_f1"))
                skill_event_f1 = _as_float(none_cond.get("skill_event_f1"))

    breakdown_path = _pick_first_existing(
        run_dir, _breakdown_candidates(picked_weights)
    )
    breakdown = _read_json(breakdown_path) if breakdown_path else None
    f1_16th: float | None = None
    f1_24th: float | None = None
    f1_intro: float | None = None
    if breakdown is not None:
        sub_f1 = breakdown.get("subdivision_f1")
        if isinstance(sub_f1, dict):
            f1_16th = _as_float(sub_f1.get("16th_off"))
            f1_24th = _as_float(sub_f1.get("24th"))
        sec_f1 = breakdown.get("section_f1")
        if isinstance(sec_f1, dict):
            f1_intro = _as_float(sec_f1.get("intro"))
        bd_err = _as_int(breakdown.get("total_errors"))
        if bd_err is not None:
            total_errors = bd_err

    meta = _read_json(run_dir / "run_meta.json") or {}
    gpu = str(meta.get("gpu", ""))
    started_utc = str(meta.get("started_utc", ""))

    framework = str(slot_data.get("framework", ""))
    if not framework and grounding is not None:
        framework = str(grounding.get("framework", ""))
        if not framework:
            model_path = str(grounding.get("model_path", ""))
            if model_path.endswith(".pt"):
                framework = "pytorch"
            elif model_path.endswith(".keras"):
                framework = "keras"

    return ScoreboardRow(
        run_name=run_dir.name,
        source=source,
        run_dir=run_dir.as_posix(),
        weights=str(slot_data.get("weights", picked_weights)),
        framework=framework,
        f1_at_05=f1_at_05,
        f1_max=f1_max,
        best_threshold=_as_float(slot_data.get("best_threshold")) or 0.5,
        n_charts=_as_int(slot_data.get("n_charts")) or 0,
        n_beats=_as_int(slot_data.get("n_beats")) or 0,
        micro_event_f1=micro_event_f1,
        skill_event_f1=skill_event_f1,
        slot_f1_drop_shuffle=drop_shuffle,
        slot_f1_drop_zeros=drop_zeros,
        f1_16th=f1_16th,
        f1_24th=f1_24th,
        f1_intro=f1_intro,
        total_errors=total_errors,
        gpu=gpu,
        started_utc=started_utc,
    )


def scan_runs(
    roots: typing.Sequence[str | pathlib.Path],
    *,
    weights: str = "auto",
    name_filter: str = "",
    min_beats: int = 0,
) -> list[ScoreboardRow]:
    """Scan run directories under ``roots`` for ITGPT evaluation artifacts.

    Args:
        roots: Directories whose immediate children are experiment run folders.
        weights: Checkpoint weight policy from :data:`VALID_WEIGHT_MODES`.
        name_filter: Optional case-insensitive substring filter on ``run_name``.
        min_beats: Minimum evaluated beats required to include a run.

    Returns:
        List of discovered :class:`ScoreboardRow` records.

    Raises:
        ValueError: If ``weights`` is not in :data:`VALID_WEIGHT_MODES`.
    """
    if weights not in VALID_WEIGHT_MODES:
        raise ValueError(
            f"unsupported weights mode {weights!r}; expected one of "
            f"{VALID_WEIGHT_MODES}"
        )
    needle = name_filter.lower().strip()
    rows: list[ScoreboardRow] = []
    for raw_root in roots:
        root = pathlib.Path(raw_root)
        if not root.is_dir():
            continue
        source = _source_tag(root)
        for child in sorted(root.iterdir()):
            if not child.is_dir():
                continue
            if needle and needle not in child.name.lower():
                continue
            row = _load_row(child, source=source, weights=weights)
            if row is None or row.n_beats < min_beats:
                continue
            rows.append(row)
    return rows


def rank_rows(
    rows: typing.Sequence[ScoreboardRow],
    *,
    sort_by: str = "f1_at_05",
) -> list[ScoreboardRow]:
    """Return scoreboard rows sorted by ``sort_by``.

    Args:
        rows: Unordered scoreboard rows.
        sort_by: Metric or metadata key from :data:`VALID_SORT_KEYS`.

    Returns:
        Sorted list of :class:`ScoreboardRow` (best first, or ascending for
        ``total_errors`` and ``run_name``).

    Raises:
        ValueError: If ``sort_by`` is not in :data:`VALID_SORT_KEYS`.
    """
    if sort_by not in VALID_SORT_KEYS:
        raise ValueError(
            f"unsupported sort_by {sort_by!r}; expected one of {VALID_SORT_KEYS}"
        )
    if sort_by == "run_name":
        return sorted(rows, key=lambda r: (r.run_name.lower(), -r.f1_at_05))
    if sort_by == "total_errors":
        return sorted(
            rows,
            key=lambda r: (
                r.total_errors is None,
                r.total_errors if r.total_errors is not None else 10**12,
                -r.f1_at_05,
            ),
        )
    if sort_by == "started_utc":
        return sorted(
            rows,
            key=lambda r: (r.started_utc != "", r.started_utc, r.f1_at_05),
            reverse=True,
        )
    if sort_by == "micro_event_f1":
        return sorted(
            rows,
            key=lambda r: (
                r.micro_event_f1 is not None,
                r.micro_event_f1 if r.micro_event_f1 is not None else -1.0,
                r.f1_at_05,
            ),
            reverse=True,
        )
    if sort_by == "f1_max":
        return sorted(rows, key=lambda r: (r.f1_max, r.f1_at_05), reverse=True)
    return sorted(rows, key=lambda r: (r.f1_at_05, r.f1_max), reverse=True)


def detect_flags(
    row: ScoreboardRow,
    *,
    baseline: ScoreboardRow | None = None,
) -> list[str]:
    """Flag audio-grounding failures or targeted slice regressions on ``row``.

    Args:
        row: Candidate scoreboard row.
        baseline: Optional champion/baseline row for slice regression checks.

    Returns:
        List of short uppercase flag tokens (empty when clean).
    """
    flags: list[str] = []
    if (
        row.slot_f1_drop_shuffle is not None
        and row.slot_f1_drop_shuffle < MIN_HEALTHY_SHUFFLE_DROP
    ):
        flags.append("LOW_SHUFFLE_DROP")
    if (
        row.slot_f1_drop_zeros is not None
        and row.slot_f1_drop_zeros < MIN_HEALTHY_ZEROS_DROP
    ):
        flags.append("LOW_ZEROS_DROP")
    if baseline is not None and baseline.run_dir != row.run_dir:
        if (
            baseline.f1_intro is not None
            and row.f1_intro is not None
            and (baseline.f1_intro - row.f1_intro) > MAX_INTRO_REGRESSION
        ):
            flags.append("INTRO_DROP")
        if (
            baseline.f1_16th is not None
            and row.f1_16th is not None
            and (baseline.f1_16th - row.f1_16th) > MAX_16TH_REGRESSION
        ):
            flags.append("16TH_DROP")
    return flags


def detect_plateau(
    rows: typing.Sequence[ScoreboardRow],
    *,
    window: int = 5,
    tolerance: float = 0.002,
) -> bool:
    """Return True if the last ``window`` chronological runs plateaued.

    Only runs with a non-empty ``started_utc`` timestamp are considered. A
    plateau triggers when at least ``window + 1`` timestamped runs exist and no
    run inside the final ``window`` beats the best earlier run by more than
    ``tolerance`` in ``f1_at_05``.

    Args:
        rows: Candidate scoreboard rows.
        window: Number of consecutive recent runs without improvement.
        tolerance: Minimum ``f1_at_05`` gain required to reset the plateau.

    Returns:
        True when a plateau is detected, else False.
    """
    timed = sorted(
        (r for r in rows if r.started_utc),
        key=lambda r: r.started_utc,
    )
    if window <= 0 or len(timed) <= window:
        return False
    prior_best = max(r.f1_at_05 for r in timed[:-window])
    recent_best = max(r.f1_at_05 for r in timed[-window:])
    return (recent_best - prior_best) <= tolerance


def _fmt_opt(value: float | None, *, digits: int = 4) -> str:
    """Format an optional float or return an em-dash placeholder.

    Args:
        value: Optional numeric score.
        digits: Decimal places to print.

    Returns:
        Formatted string.
    """
    if value is None:
        return "—"
    return f"{value:.{digits}f}"


def _find_baseline(
    rows: typing.Sequence[ScoreboardRow],
    baseline_name: str,
) -> ScoreboardRow | None:
    """Resolve a baseline row by exact or substring match on ``run_name``.

    Args:
        rows: Available scoreboard rows.
        baseline_name: Substring identifying the baseline run (empty for none).

    Returns:
        Matching :class:`ScoreboardRow` or ``None``.
    """
    needle = baseline_name.lower().strip()
    if not needle:
        return None
    for row in rows:
        if row.run_name.lower() == needle:
            return row
    for row in rows:
        if needle in row.run_name.lower():
            return row
    return None


def format_markdown_table(
    rows: typing.Sequence[ScoreboardRow],
    *,
    baseline_name: str = "",
    top_k: int = 0,
) -> str:
    """Render ranked scoreboard rows as a GitHub-flavored markdown table.

    Args:
        rows: Pre-ranked scoreboard rows.
        baseline_name: Optional baseline run name/substring for delta column.
        top_k: Optional maximum number of rows to include (0 = all).

    Returns:
        Markdown table string (with optional plateau notice).
    """
    if not rows:
        return "_No matching ITGPT evaluation runs found._"

    baseline = _find_baseline(rows, baseline_name) if baseline_name else rows[0]
    visible = list(rows[:top_k]) if top_k > 0 else list(rows)

    header = (
        "| Run | Src | Wts | F1@0.5 | Δpp | F1_max | ±20ms | Δshuf | Δzero "
        "| 16th | 24th | Intro | Errs | Flags |"
    )
    sep = (
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: "
        "| ---: | ---: | ---: | ---: | --- |"
    )
    lines = [header, sep]
    for row in visible:
        delta_str = "—"
        if baseline is not None:
            diff_pp = 100.0 * (row.f1_at_05 - baseline.f1_at_05)
            delta_str = "base" if row.run_dir == baseline.run_dir else f"{diff_pp:+.2f}"
        flags = detect_flags(row, baseline=baseline)
        err_str = str(row.total_errors) if row.total_errors is not None else "—"
        lines.append(
            f"| `{row.run_name}` | {row.source} | {row.weights} "
            f"| {row.f1_at_05:.4f} | {delta_str} | {row.f1_max:.4f} "
            f"| {_fmt_opt(row.micro_event_f1)} "
            f"| {_fmt_opt(row.slot_f1_drop_shuffle, digits=3)} "
            f"| {_fmt_opt(row.slot_f1_drop_zeros, digits=3)} "
            f"| {_fmt_opt(row.f1_16th, digits=3)} "
            f"| {_fmt_opt(row.f1_24th, digits=3)} "
            f"| {_fmt_opt(row.f1_intro, digits=3)} "
            f"| {err_str} | {', '.join(flags) if flags else 'OK'} |"
        )

    if detect_plateau(rows):
        lines.append("")
        lines.append(
            "> **Plateau warning:** Last 5 timestamped runs did not improve "
            "`F1@0.5` by > +0.20 pp — upgrade tier (B/C) or change hypothesis class."
        )
    return "\n".join(lines)
