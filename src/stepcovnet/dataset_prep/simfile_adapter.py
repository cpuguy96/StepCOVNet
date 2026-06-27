"""Simfile open, chart selection, and timing."""

from __future__ import annotations

import dataclasses
import pathlib

import numpy as np
import simfile
from simfile import notes, timing, types
from simfile.notes import timed

from stepcovnet.dataset_prep import (
    arrow_rows,
    audio_resolve,
    config,
    constants,
    export,
    models,
    pack_results,
)


@dataclasses.dataclass
class ChartSkip:
    """One dance-single chart skipped during export.

    Attributes:
        difficulty: Lowercase difficulty label.
        meter: Raw simfile meter.
        reason: Skip reason code for this chart.
    """

    difficulty: str
    meter: int
    reason: str


@dataclasses.dataclass
class ParsePackResult:
    """Outcome of parsing one raw song pack.

    Attributes:
        reason: Skip or error code when export cannot proceed; ``None`` on success.
        pack: Parsed song object when ``reason`` is ``None``.
        warnings: Pack-level warning codes accumulated during parse.
        chart_skips: Per-chart skip records for dance-single charts.
    """

    reason: str | None
    pack: models.ParsedSongPack | None
    warnings: list[str]
    chart_skips: list[ChartSkip]


def open_simfile(sim_path: str | pathlib.Path) -> types.Simfile:
    """Open a simfile with encoding retries from ``constants.ENCODING_RETRIES``.

    Args:
        sim_path: Path to ``.ssc`` or ``.sm`` file.

    Returns:
        Parsed simfile object.

    Raises:
        UnicodeDecodeError: If every encoding retry fails.
        ValueError: If simfile syntax is invalid after a successful decode.
        FileNotFoundError: If ``sim_path`` does not exist.
    """
    path = pathlib.Path(sim_path)
    last_error: UnicodeDecodeError | None = None
    for encoding in constants.ENCODING_RETRIES:
        try:
            sim, _encoding = simfile.open_with_detected_encoding(
                str(path),
                try_encodings=[encoding],
            )
            return sim
        except UnicodeDecodeError as exc:
            last_error = exc
        except Exception as exc:
            raise ValueError(f"simfile parse failed: {path}: {exc}") from exc
    if last_error is not None:
        raise UnicodeDecodeError(
            last_error.encoding,
            last_error.object,
            last_error.start,
            last_error.end,
            last_error.reason,
        ) from last_error
    raise FileNotFoundError(f"simfile not found: {path}")


def parse_bpm_segments(sim: types.Simfile) -> list[models.BpmSegment]:
    """Parse simfile ``#BPMS`` into segment objects.

    Args:
        sim: Open simfile object.

    Returns:
        BPM segments sorted by ``start_beat``.
    """
    beat_values = timing.BeatValues.from_str(sim.bpms)
    segments = [
        models.BpmSegment(start_beat=float(item.beat), bpm=float(item.value))
        for item in beat_values
    ]
    segments.sort(key=lambda item: item.start_beat)
    return segments


def parse_metadata(sim: types.Simfile) -> models.SimfileMetadata:
    """Parse song-level simfile tags into ``SimfileMetadata``.

    Args:
        sim: Open simfile object.

    Returns:
        Parsed metadata including BPM segments.

    Raises:
        ValueError: If ``#BPMS`` is empty.
    """
    bpm_segments = parse_bpm_segments(sim)
    if not bpm_segments:
        raise ValueError("simfile has no #BPMS segments")
    selectable_raw = (sim.selectable or "").strip().upper()
    selectable = selectable_raw in {"YES", "1", "TRUE"}
    return models.SimfileMetadata(
        title=(sim.title or "").strip(),
        artist=(sim.artist or "").strip(),
        subtitle=(sim.subtitle or "").strip(),
        music_filename=(sim.music or "").strip(),
        offset_sec=float(sim.offset or 0.0),
        initial_bpm=bpm_segments[0].bpm,
        bpm_segments=bpm_segments,
        selectable=selectable,
    )


def normalize_difficulty(difficulty: str) -> tuple[str, str, list[str]]:
    """Normalize chart difficulty and classify standard vs custom.

    Args:
        difficulty: Raw ``#DIFFICULTY`` value.

    Returns:
        Tuple of lowercase difficulty, difficulty kind, and warning codes.
    """
    normalized = difficulty.strip().lower()
    if normalized in constants.STANDARD_DIFFICULTIES:
        return normalized, constants.DIFFICULTY_KIND_STANDARD, []
    return (
        normalized or "custom",
        constants.DIFFICULTY_KIND_CUSTOM,
        ["custom_difficulty"],
    )


def is_dance_single(chart: types.Chart) -> bool:
    """Return True when chart stepstype is ``dance-single``.

    Args:
        chart: Simfile chart object.

    Returns:
        True for dance-single charts (case-insensitive).
    """
    return str(chart.stepstype).strip().lower() == "dance-single"


def _chart_name(chart) -> str:
    return str(getattr(chart, "chartname", None) or "").strip()


def _chart_credit(chart) -> str:
    return str(getattr(chart, "credit", None) or "").strip()


def build_chart_summary(chart: types.Chart, *, num_steps: int) -> models.ChartSummary:
    """Build a ``ChartSummary`` from a simfile chart block.

    Args:
        chart: Simfile chart object.
        num_steps: Encoded player step count.

    Returns:
        Chart summary with normalized difficulty metadata.
    """
    difficulty, difficulty_kind, _warnings = normalize_difficulty(
        str(chart.difficulty or "")
    )
    return models.ChartSummary(
        stepstype=str(chart.stepstype or "").strip().lower(),
        difficulty=difficulty,
        difficulty_kind=difficulty_kind,
        meter=int(chart.meter or 0),
        chart_name=_chart_name(chart),
        credit=_chart_credit(chart),
        num_steps=num_steps,
    )


def build_available_charts(sim: types.Simfile) -> list[models.ChartSummary]:
    """Summarize non-dance-single charts for inventory export.

    Args:
        sim: Open simfile object.

    Returns:
        Summaries for charts whose stepstype is not ``dance-single``.
    """
    summaries: list[models.ChartSummary] = []
    for chart in sim.charts:
        if is_dance_single(chart):
            continue
        summaries.append(build_chart_summary(chart, num_steps=0))
    return summaries


def default_chart_index(charts: list[models.ParsedChart]) -> int:
    """Pick the highest ladder-rank exported chart index.

    Args:
        charts: Exported dance-single charts in simfile order.

    Returns:
        Index of the default chart; ``0`` when ``charts`` is empty.
    """
    if not charts:
        return 0
    best_index = 0
    best_rank = -1
    for index, chart in enumerate(charts):
        summary = chart.summary
        if summary.difficulty_kind == constants.DIFFICULTY_KIND_CUSTOM:
            rank = constants.DIFFICULTY_RANK["custom"]
        else:
            rank = constants.DIFFICULTY_RANK.get(summary.difficulty, 0)
        if rank > best_rank:
            best_rank = rank
            best_index = index
    return best_index


def _append_encode_warnings(
    warnings: list[str],
    stats: arrow_rows.EncodeChartStats,
) -> None:
    if stats.mine_notes_unencoded:
        warnings.append(f"mine_notes_unencoded:{stats.mine_notes_unencoded}")
    if stats.fake_notes_unencoded:
        warnings.append(f"fake_notes_unencoded:{stats.fake_notes_unencoded}")
    if stats.lift_notes_unencoded:
        warnings.append(f"lift_notes_unencoded:{stats.lift_notes_unencoded}")
    if stats.beats_dropped_empty:
        warnings.append(f"beats_dropped_empty:{stats.beats_dropped_empty}")


def encode_chart(
    sim: types.Simfile,
    chart: types.Chart,
    *,
    max_steps_per_chart: int,
    allow_over_cap: bool,
) -> tuple[models.ParsedChart | None, list[str], ChartSkip | None]:
    """Encode one dance-single chart to ``ParsedChart`` or skip it.

    Args:
        sim: Open simfile object.
        chart: Dance-single chart block.
        max_steps_per_chart: Per-chart step cap.
        allow_over_cap: When True, export charts above the cap.

    Returns:
        Parsed chart (or ``None``), pack-level warnings, and optional skip record.
    """
    warnings: list[str] = []
    difficulty, _difficulty_kind, difficulty_warnings = normalize_difficulty(
        str(chart.difficulty or "")
    )
    warnings.extend(difficulty_warnings)
    meter = int(chart.meter or 0)

    timing_data = timing.TimingData(sim, chart)
    note_data = notes.NoteData(chart)
    timed_notes = list(timed.time_notes(note_data, timing_data))
    times_sec, arrow_rows_out, column_codes, stats = arrow_rows.encode_timed_chart_rows(
        timed_notes
    )
    _append_encode_warnings(warnings, stats)

    num_steps = len(arrow_rows_out)
    if num_steps == 0:
        return (
            None,
            warnings,
            ChartSkip(
                difficulty=difficulty,
                meter=meter,
                reason=constants.CHART_SKIP_EMPTY,
            ),
        )
    if num_steps > max_steps_per_chart and not allow_over_cap:
        return (
            None,
            warnings,
            ChartSkip(
                difficulty=difficulty,
                meter=meter,
                reason=constants.CHART_SKIP_OVER_CAP,
            ),
        )

    from stepcovnet import metrics  # noqa: PLC0415

    violations, _hold_ends, _examples = metrics.compute_chart_validity_violations(
        np.asarray(column_codes, dtype=np.int32)
    )
    if violations > 0:
        return (
            None,
            warnings,
            ChartSkip(
                difficulty=difficulty,
                meter=meter,
                reason=constants.CHART_SKIP_INVALID_HOLDS,
            ),
        )

    summary = build_chart_summary(chart, num_steps=num_steps)
    return (
        models.ParsedChart(
            summary=summary,
            times_sec=times_sec,
            arrow_rows=arrow_rows_out,
            column_codes=column_codes,
        ),
        warnings,
        None,
    )


def parse_song_pack(
    pack_dir: str | pathlib.Path,
    *,
    simfile_name: str,
    normalized_bundle: str,
    normalized_id: str,
    source_pack_relpath: str,
    prep_config: config.PrepConfig | None = None,
) -> ParsePackResult:
    """Parse one raw pack directory into a ``ParsedSongPack``.

    Args:
        pack_dir: Raw song pack directory.
        simfile_name: Simfile basename inside the pack (from discovery).
        normalized_bundle: Output bundle slug from normalization.
        normalized_id: Output song slug from normalization.
        source_pack_relpath: Pack path relative to preprocess input root.
        prep_config: Optional prep settings; defaults used when omitted.

    Returns:
        Parse outcome with status, optional pack, warnings, and chart skips.
    """
    cfg = prep_config or config.default_prep_config()
    pack_path = pathlib.Path(pack_dir)
    sim_path = pack_path / simfile_name
    warnings: list[str] = []
    chart_skips: list[ChartSkip] = []

    try:
        sim = open_simfile(sim_path)
    except UnicodeDecodeError:
        return ParsePackResult(
            reason=pack_results.REASON_ENCODING_ERROR,
            pack=None,
            warnings=warnings,
            chart_skips=chart_skips,
        )
    except (OSError, ValueError) as exc:
        warnings.append(f"parse_error:{type(exc).__name__}")
        return ParsePackResult(
            reason=pack_results.REASON_PARSE_ERROR,
            pack=None,
            warnings=warnings,
            chart_skips=chart_skips,
        )

    try:
        metadata = parse_metadata(sim)
    except ValueError:
        return ParsePackResult(
            reason=pack_results.REASON_PARSE_ERROR,
            pack=None,
            warnings=warnings,
            chart_skips=chart_skips,
        )

    dance_single_charts = [chart for chart in sim.charts if is_dance_single(chart)]
    if not dance_single_charts:
        return ParsePackResult(
            reason=pack_results.REASON_NO_DANCE_SINGLE,
            pack=None,
            warnings=warnings,
            chart_skips=chart_skips,
        )

    audio = audio_resolve.resolve_audio(
        pack_path,
        music_filename=metadata.music_filename,
        simfile_name=simfile_name,
        title=metadata.title,
    )
    if audio is None:
        return ParsePackResult(
            reason=pack_results.REASON_NO_AUDIO,
            pack=None,
            warnings=warnings,
            chart_skips=chart_skips,
        )
    warnings.extend(audio.warnings)

    output_audio = export.output_audio_filename(
        normalized_id,
        audio.audio_resolved_relpath,
    )

    exported: list[models.ParsedChart] = []
    for chart in dance_single_charts:
        parsed_chart, chart_warnings, skip = encode_chart(
            sim,
            chart,
            max_steps_per_chart=cfg.max_steps_per_chart,
            allow_over_cap=cfg.allow_over_cap,
        )
        warnings.extend(chart_warnings)
        if skip is not None:
            chart_skips.append(skip)
            continue
        if parsed_chart is not None:
            exported.append(parsed_chart)

    if not exported:
        return ParsePackResult(
            reason=pack_results.REASON_NO_EXPORTABLE_CHARTS,
            pack=None,
            warnings=warnings,
            chart_skips=chart_skips,
        )

    if "custom_difficulty" in warnings:
        warnings = [item for item in warnings if item != "custom_difficulty"]
        warnings.append("custom_difficulty")

    pack = models.ParsedSongPack(
        schema_version=constants.SCHEMA_VERSION,
        normalized_bundle=normalized_bundle,
        normalized_id=normalized_id,
        source_pack_relpath=source_pack_relpath,
        source_simfile=simfile_name,
        metadata=metadata,
        charts=exported,
        default_chart_index=default_chart_index(exported),
        available_charts=build_available_charts(sim),
        audio_filename=output_audio,
        audio_source=audio.audio_source,
        audio_resolved_relpath=audio.audio_resolved_relpath,
        warnings=list(dict.fromkeys(warnings)),
    )
    return ParsePackResult(
        reason=None,
        pack=pack,
        warnings=pack.warnings,
        chart_skips=chart_skips,
    )
