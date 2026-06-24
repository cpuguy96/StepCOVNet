"""Parsed simfile pack dataclasses and on-disk JSON load helpers."""

from __future__ import annotations

import dataclasses
import json
import os
import pathlib

from stepcovnet.dataset_prep import constants


class _DictSerializableMixin:
    """Mixin providing default as_dict and from_dict for dataclass models."""

    def as_dict(self) -> dict:
        """Convert model to dictionary for JSON serialization.

        Returns:
            Serializable mapping of dataclass fields.
        """
        return dataclasses.asdict(self)  # type: ignore[arg-type]

    @classmethod
    def from_dict(cls, data: dict):
        """Create model from dictionary.

        Args:
            data: Serialized field values for the dataclass.

        Returns:
            Instance of the model class with fields taken from data.
        """
        return cls(**data)


@dataclasses.dataclass
class BpmSegment(_DictSerializableMixin):
    """One entry from simfile ``#BPMS``.

    Attributes:
        start_beat: Beat index where this BPM begins.
        bpm: Beats per minute from that beat onward.
    """

    start_beat: float
    bpm: float


@dataclasses.dataclass
class SimfileMetadata(_DictSerializableMixin):
    """Song-level tags shared across charts in a pack.

    Attributes:
        title: ``#TITLE`` display name.
        artist: ``#ARTIST``.
        subtitle: ``#SUBTITLE``.
        music_filename: Raw ``#MUSIC`` string (may differ from resolved file).
        offset_sec: ``#OFFSET``; chart times come from TimingData only.
        initial_bpm: First ``#BPMS`` value (legacy ``.txt`` BPM line).
        bpm_segments: Full ``#BPMS`` ladder.
        selectable: ``#SELECTABLE`` flag from simfile.
    """

    title: str
    artist: str
    subtitle: str
    music_filename: str
    offset_sec: float
    initial_bpm: float
    bpm_segments: list[BpmSegment]
    selectable: bool

    @classmethod
    def from_dict(cls, data: dict):
        """Parse metadata including nested BPM segments.

        Args:
            data: Serialized metadata block from ``.chart.json``.

        Returns:
            Parsed song-level metadata with nested BPM segments.
        """
        segments = [BpmSegment.from_dict(item) for item in data.get("bpm_segments", [])]
        payload = dict(data)
        payload["bpm_segments"] = segments
        return cls(**payload)


@dataclasses.dataclass
class ChartSummary(_DictSerializableMixin):
    """Per-chart metadata without step rows.

    Attributes:
        stepstype: Simfile stepstype (exported charts are ``dance-single``).
        difficulty: Lowercase ``#DIFFICULTY`` label.
        difficulty_kind: ``standard`` or ``custom`` (non-enum difficulties).
        meter: Raw ``#METER`` integer.
        chart_name: Optional ``#CHARTNAME``.
        credit: Optional chart credit string.
        num_steps: Player step count after encoding rules.
    """

    stepstype: str
    difficulty: str
    difficulty_kind: str
    meter: int
    chart_name: str
    credit: str
    num_steps: int


@dataclasses.dataclass
class ParsedChart(_DictSerializableMixin):
    """One exported dance-single chart.

    Attributes:
        summary: Chart metadata block.
        times_sec: Seconds per encoded beat row.
        arrow_rows: Quaternary strings (four columns, digits ``0``–``3``).
        column_codes: ``int(arrow_row, 4)`` per row for arrow-model compat.
    """

    summary: ChartSummary
    times_sec: list[float]
    arrow_rows: list[str]
    column_codes: list[int]

    @classmethod
    def from_dict(cls, data: dict):
        """Parse chart including nested summary.

        Args:
            data: Serialized chart block from ``.chart.json``.

        Returns:
            Parsed chart with nested summary and step rows.
        """
        payload = dict(data)
        payload["summary"] = ChartSummary.from_dict(data["summary"])
        return cls(**payload)


@dataclasses.dataclass
class ParsedSongPack(_DictSerializableMixin):
    """Canonical on-disk object written to ``{id}.chart.json``.

    Attributes:
        schema_version: JSON layout version for chart objects.
        normalized_bundle: Slug of source bundle folder.
        normalized_id: Slug of song within bundle.
        source_pack_relpath: Path to raw pack from input root.
        source_simfile: Basename of parsed simfile (``.ssc`` preferred).
        metadata: Song-level simfile tags.
        charts: Exported ``dance-single`` charts.
        default_chart_index: Index of highest ladder-rank chart in ``charts``.
        available_charts: Summaries of non-``dance-single`` charts only.
        audio_filename: Output audio basename ``{normalized_id}{ext}``.
        audio_source: ``music_tag`` or ``inferred``.
        audio_resolved_relpath: Resolved audio path within the raw pack dir.
        warnings: Machine-readable warning codes for this pack.
    """

    schema_version: int
    normalized_bundle: str
    normalized_id: str
    source_pack_relpath: str
    source_simfile: str
    metadata: SimfileMetadata
    charts: list[ParsedChart]
    default_chart_index: int
    available_charts: list[ChartSummary]
    audio_filename: str
    audio_source: str
    audio_resolved_relpath: str
    warnings: list[str]

    @classmethod
    def from_dict(cls, data: dict):
        """Parse a song pack including nested metadata and charts.

        Args:
            data: Serialized ``.chart.json`` root object.

        Returns:
            Parsed song pack with nested metadata and chart blocks.
        """
        payload = dict(data)
        payload["metadata"] = SimfileMetadata.from_dict(data["metadata"])
        payload["charts"] = [ParsedChart.from_dict(c) for c in data.get("charts", [])]
        payload["available_charts"] = [
            ChartSummary.from_dict(c) for c in data.get("available_charts", [])
        ]
        return cls(**payload)


def chart_json_path(
    output_dir: str | os.PathLike[str],
    normalized_bundle: str,
    normalized_id: str,
) -> pathlib.Path:
    """Return the canonical ``.chart.json`` path for a processed song.

    Args:
        output_dir: Preprocess output root.
        normalized_bundle: Normalized bundle slug.
        normalized_id: Normalized song slug within the bundle.

    Returns:
        Path to ``{output_dir}/{bundle}/{id}/{id}.chart.json``.
    """
    root = pathlib.Path(output_dir)
    return root / normalized_bundle / normalized_id / f"{normalized_id}.chart.json"


def load_parsed_song(
    output_dir: str | os.PathLike[str],
    normalized_bundle: str,
    normalized_id: str,
) -> ParsedSongPack:
    """Load ``{id}.chart.json`` from the nested final_data layout.

    Args:
        output_dir: Preprocess output root.
        normalized_bundle: Normalized bundle slug.
        normalized_id: Normalized song slug within the bundle.

    Returns:
        Parsed song pack for the given output location.

    Raises:
        ValueError: If ``schema_version`` is missing or unsupported.
    """
    path = chart_json_path(output_dir, normalized_bundle, normalized_id)
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    version = data.get("schema_version")
    if version is None:
        raise ValueError(f"missing schema_version in {path}")
    if version != constants.SCHEMA_VERSION:
        raise ValueError(
            f"unsupported schema_version {version} in {path}; "
            f"expected {constants.SCHEMA_VERSION}"
        )
    return ParsedSongPack.from_dict(data)
