"""Write ``.chart.json``, legacy ``.txt``, and pack artifacts."""

from __future__ import annotations

import dataclasses
import json
import os
import pathlib
import shutil

from stepcovnet.dataset_prep import config, models


class _DictSerializableMixin:
    """Mixin providing default as_dict and from_dict for export dataclasses."""

    def as_dict(self) -> dict:
        """Convert object to dictionary for JSON serialization."""
        return dataclasses.asdict(self)  # type: ignore[arg-type]

    @classmethod
    def from_dict(cls, data: dict):
        """Create object from dictionary."""
        return cls(**data)


@dataclasses.dataclass
class WorkerResult(_DictSerializableMixin):
    """Per-pack outcome from parallel pack processing.

    Attributes:
        normalized_bundle: Output bundle slug.
        normalized_id: Output song slug.
        output_relpath: ``{bundle}/{id}`` relative to output root.
        source_pack: Raw pack path relative to input root.
        result: Coarse outcome — ``pack_exported``, ``pack_skipped``, or ``pack_error``.
        reason: Detail code when not exported.
        warnings: Pack-level warnings.
        charts_exported: Number of charts written.
        charts_skipped: Number of per-chart skips.
        chart_skips: Serialized chart skip records.
        message: Optional error detail when ``result`` is ``pack_error``.
    """

    normalized_bundle: str
    normalized_id: str
    output_relpath: str
    source_pack: str
    result: str
    reason: str | None
    warnings: list[str]
    charts_exported: int
    charts_skipped: int
    chart_skips: list[dict]
    message: str = ""


def _difficulty_label(difficulty: str) -> str:
    if not difficulty:
        return "Custom"
    return difficulty.strip().capitalize()


def render_legacy_txt(pack: models.ParsedSongPack) -> str:
    """Render multi-block v2 ``.txt`` for a parsed pack.

    Args:
        pack: Parsed song pack with at least one chart.

    Returns:
        Legacy chart text matching ``datasets._parse_step_chart`` layout.
    """
    title = pack.metadata.title or pack.normalized_id
    lines = [
        f"TITLE {title}",
        f"BPM {pack.metadata.initial_bpm}",
        "NOTES",
    ]
    for chart in pack.charts:
        summary = chart.summary
        lines.append(f"DIFFICULTY {_difficulty_label(summary.difficulty)}")
        for arrow_row, time_sec in zip(chart.arrow_rows, chart.times_sec, strict=True):
            lines.append(f"{arrow_row} {time_sec}")
    lines.append("DIFFICULTY")
    return "\n".join(lines) + "\n"


def output_audio_filename(normalized_id: str, source_relpath: str) -> str:
    """Return output audio basename ``{normalized_id}{ext}``.

    Args:
        normalized_id: Output song slug.
        source_relpath: Resolved audio path within the raw pack directory.

    Returns:
        Basename for the copied audio file in the song output directory.
    """
    ext = pathlib.Path(source_relpath).suffix.lower()
    return f"{normalized_id}{ext}" if ext else normalized_id


def song_output_dir(
    output_dir: str | os.PathLike[str],
    normalized_bundle: str,
    normalized_id: str,
) -> pathlib.Path:
    """Return final song directory path under the preprocess output root.

    Args:
        output_dir: Preprocess output root.
        normalized_bundle: Output bundle slug.
        normalized_id: Output song slug within the bundle.

    Returns:
        Path to ``{output_dir}/{bundle}/{id}/``.
    """
    return pathlib.Path(output_dir) / normalized_bundle / normalized_id


def write_song_pack(
    pack: models.ParsedSongPack,
    *,
    raw_pack_dir: pathlib.Path,
    output_dir: str | os.PathLike[str],
    prep_config: config.PrepConfig,
) -> pathlib.Path:
    """Write one processed song directory atomically.

    Args:
        pack: Parsed and validated song pack.
        raw_pack_dir: Source pack directory containing audio.
        output_dir: Preprocess output root.
        prep_config: Prep settings (legacy txt flag).

    Returns:
        Path to the final song output directory.

    Raises:
        FileNotFoundError: If resolved audio is missing on disk.
    """
    final_dir = song_output_dir(
        output_dir,
        pack.normalized_bundle,
        pack.normalized_id,
    )
    tmp_dir = final_dir.with_name(f"{final_dir.name}.tmp")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    audio_src = raw_pack_dir / pack.audio_resolved_relpath
    if not audio_src.is_file():
        raise FileNotFoundError(f"audio not found for pack write: {audio_src}")
    pack = dataclasses.replace(
        pack,
        audio_filename=output_audio_filename(
            pack.normalized_id,
            pack.audio_resolved_relpath,
        ),
    )
    audio_dst = tmp_dir / pack.audio_filename
    shutil.copy2(audio_src, audio_dst)

    chart_path = tmp_dir / f"{pack.normalized_id}.chart.json"
    with chart_path.open("w", encoding="utf-8") as handle:
        json.dump(pack.as_dict(), handle, indent=2)
        handle.write("\n")

    if prep_config.export_legacy_txt:
        txt_path = tmp_dir / f"{pack.normalized_id}.txt"
        txt_path.write_text(render_legacy_txt(pack), encoding="utf-8")

    if final_dir.exists():
        shutil.rmtree(final_dir)
    tmp_dir.rename(final_dir)
    return final_dir


def worker_result_path(
    output_dir: str | os.PathLike[str],
    normalized_bundle: str,
    normalized_id: str,
) -> pathlib.Path:
    """Return staging path for one worker JSON result.

    Args:
        output_dir: Preprocess output root.
        normalized_bundle: Output bundle slug.
        normalized_id: Output song slug within the bundle.

    Returns:
        Path to ``_staging/worker_results/{bundle}__{id}.json``.
    """
    staging = pathlib.Path(output_dir) / "_staging" / "worker_results"
    filename = f"{normalized_bundle}__{normalized_id}.json"
    return staging / filename


def save_worker_result(
    result: WorkerResult,
    output_dir: str | os.PathLike[str],
) -> pathlib.Path:
    """Write a worker result JSON under ``_staging/worker_results/``.

    Args:
        result: Per-pack worker outcome to serialize.
        output_dir: Preprocess output root.

    Returns:
        Path to the written JSON file.
    """
    path = worker_result_path(
        output_dir,
        result.normalized_bundle,
        result.normalized_id,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(result.as_dict(), handle, indent=2)
        handle.write("\n")
    return path
