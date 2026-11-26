"""Pure-Python StepMania ``.sm`` / ``.ssc`` parser and timing engine.

Implements MSD tokenization, ``.sm`` / ``.ssc`` chart & metadata extraction
(including SSC ``>= 0.7`` split timing), note-grid parsing, and StepMania
beat-to-seconds timing (``#OFFSET``, ``#BPMS``, ``#STOPS``/``#FREEZES``,
``#DELAYS``, and ``#WARPS``) with zero third-party dependencies.
"""

from __future__ import annotations

import bisect
import collections.abc
import dataclasses
import decimal
import enum
import fractions
import heapq
import io
import pathlib
import re

MEASURE_SUBDIVISION = 192
BEAT_SUBDIVISION = MEASURE_SUBDIVISION // 4
SSC_VERSION_SPLIT_TIMING = 0.7

_MULTI_VALUE_PROPERTIES = frozenset({"ATTACKS", "DISPLAYBPM"})
_CHART_TIMING_KEYS = (
    "BPMS",
    "STOPS",
    "DELAYS",
    "TIMESIGNATURES",
    "TICKCOUNTS",
    "COMBOS",
    "WARPS",
    "SPEEDS",
    "SCROLLS",
    "FAKES",
    "LABELS",
)
_SM_CHART_PROPERTIES = (
    "STEPSTYPE",
    "DESCRIPTION",
    "DIFFICULTY",
    "METER",
    "RADARVALUES",
    "NOTES",
)


class _MSDToken(enum.Enum):
    TEXT = enum.auto()
    START_PARAMETER = enum.auto()
    NEXT_COMPONENT = enum.auto()
    END_PARAMETER = enum.auto()
    ESCAPE = enum.auto()
    COMMENT = enum.auto()


_LEXER_RULES: tuple[tuple[re.Pattern[str], _MSDToken, _MSDToken], ...] = (
    (re.compile(r"[^\\/:;#]+"), _MSDToken.TEXT, _MSDToken.TEXT),
    (re.compile(r"#"), _MSDToken.START_PARAMETER, _MSDToken.TEXT),
    (re.compile(r":"), _MSDToken.TEXT, _MSDToken.NEXT_COMPONENT),
    (re.compile(r";"), _MSDToken.TEXT, _MSDToken.END_PARAMETER),
    (re.compile(r"(?s)\\."), _MSDToken.TEXT, _MSDToken.ESCAPE),
    (re.compile(r"//[^\r\n]*"), _MSDToken.COMMENT, _MSDToken.COMMENT),
    (re.compile(r"/"), _MSDToken.TEXT, _MSDToken.TEXT),
)


@dataclasses.dataclass(frozen=True)
class MSDParameter:
    """One parsed ``#KEY:VAL1:VAL2;`` parameter from an MSD document.

    Attributes:
        components: Raw tuple ``(key, val1, ...)``.
    """

    components: tuple[str, ...]

    @property
    def key(self) -> str:
        """Parameter key name (first component)."""
        return self.components[0] if self.components else ""

    @property
    def value(self) -> str:
        """Primary parameter value (second component, or empty string)."""
        return self.components[1] if len(self.components) > 1 else ""


def _lex_msd(text: str) -> collections.abc.Iterator[tuple[_MSDToken, str]]:
    pos = 0
    length = len(text)
    inside_parameter = False
    last_text: str | None = None

    while pos < length:
        for regex, outside_tok, inside_tok in _LEXER_RULES:
            match = regex.match(text, pos)
            if match is None:
                continue
            matched = match.group(0)
            pos = match.end()
            token = inside_tok if inside_parameter else outside_tok

            # Recover from a missing ';' when a new '#' begins on the next line.
            if (
                matched == "#"
                and token is _MSDToken.TEXT
                and last_text is not None
                and last_text[-1] in "\r\n"
            ):
                token = _MSDToken.START_PARAMETER

            if token is _MSDToken.START_PARAMETER:
                inside_parameter = True
            elif token is _MSDToken.END_PARAMETER:
                inside_parameter = False

            if token is _MSDToken.TEXT:
                last_text = matched

            yield token, matched
            break


def parse_msd(
    text: str,
    *,
    strict: bool = True,
) -> collections.abc.Iterator[MSDParameter]:
    """Parse MSD text into a stream of :class:`MSDParameter` instances.

    Args:
        text: Full MSD document text.
        strict: When True, raise ``ValueError`` on stray non-whitespace text
            between parameters.

    Yields:
        MSDParameter: Parsed parameter objects in document order.
    """
    components: list[io.StringIO] = []
    inside_parameter = False
    last_key: str | None = None

    def push_text(chunk: str) -> None:
        if inside_parameter:
            components[-1].write(chunk)
        elif strict and chunk and not chunk.isspace() and chunk != "\ufeff":
            char = chunk.lstrip()[0]
            where = (
                "at start of document"
                if last_key is None
                else f"after {last_key!r} parameter"
            )
            raise ValueError(f"stray {char!r} encountered {where}")

    def complete_parameter() -> MSDParameter:
        nonlocal last_key, components, inside_parameter
        param = MSDParameter(tuple(buf.getvalue() for buf in components))
        last_key = param.key
        components = []
        inside_parameter = False
        return param

    for token, value in _lex_msd(text):
        if token is _MSDToken.TEXT:
            push_text(value)
        elif token is _MSDToken.START_PARAMETER:
            if inside_parameter:
                yield complete_parameter()
            inside_parameter = True
            components.append(io.StringIO())
        elif token is _MSDToken.END_PARAMETER:
            if inside_parameter:
                yield complete_parameter()
        elif token is _MSDToken.NEXT_COMPONENT:
            if inside_parameter:
                components.append(io.StringIO())
        elif token is _MSDToken.ESCAPE:
            push_text(value[1])
        elif token is _MSDToken.COMMENT:
            pass

    if inside_parameter:
        yield complete_parameter()


def round_beat_to_tick(beat: fractions.Fraction) -> fractions.Fraction:
    """Round a rational beat to the nearest 1/48th-beat StepMania tick.

    Args:
        beat: Rational beat value.

    Returns:
        Beat rounded to the nearest ``1 / BEAT_SUBDIVISION``.
    """
    return fractions.Fraction(round(beat * BEAT_SUBDIVISION), BEAT_SUBDIVISION)


def parse_beat(raw: str | float | decimal.Decimal) -> fractions.Fraction:
    """Parse a beat literal and round to the nearest StepMania tick.

    Args:
        raw: Numeric string, float, or Decimal beat value.

    Returns:
        Exact ``Fraction`` rounded to the nearest 1/48th beat.
    """
    return round_beat_to_tick(fractions.Fraction(raw))


@dataclasses.dataclass(frozen=True)
class BeatValue:
    """A timing event occurring at a specific beat.

    Attributes:
        beat: Beat position (exact Fraction).
        value: Decimal payload (BPM, stop/delay seconds, or warp beats).
    """

    beat: fractions.Fraction
    value: decimal.Decimal


def parse_beat_values(raw: str | None) -> list[BeatValue]:
    """Parse a comma-separated ``beat=value`` timing list.

    Args:
        raw: MSD timing property string (``BPMS``, ``STOPS``, ``DELAYS``, ``WARPS``).

    Returns:
        List of ``BeatValue`` entries in source order.
    """
    items: list[BeatValue] = []
    if raw and raw.strip():
        for row in raw.split(","):
            stripped = row.strip()
            if not stripped:
                continue
            beat_str, val_str = stripped.split("=")
            items.append(
                BeatValue(
                    beat=parse_beat(beat_str.strip()),
                    value=decimal.Decimal(val_str.strip()),
                )
            )
    return items


class NoteType(enum.Enum):
    """StepMania note characters supported in chart note data.

    Attributes:
        TAP: Standard tap note (``1``).
        HOLD_HEAD: Hold start (``2``).
        TAIL: Hold or roll tail (``3``).
        ROLL_HEAD: Roll start (``4``).
        ATTACK: Attack note (``A``).
        FAKE: Unhittable fake note (``F``).
        KEYSOUND: Keysound marker (``K``).
        LIFT: Lift note (``L``).
        MINE: Mine note (``M``).
    """

    TAP = "1"
    HOLD_HEAD = "2"
    TAIL = "3"
    ROLL_HEAD = "4"
    ATTACK = "A"
    FAKE = "F"
    KEYSOUND = "K"
    LIFT = "L"
    MINE = "M"


@dataclasses.dataclass(frozen=True)
class Note:
    """One non-zero note cell in a chart's note data.

    Attributes:
        beat: Rational beat position of the note.
        column: Zero-based column index.
        note_type: StepMania note type enum.
        player: Player index (non-zero only for routine charts).
        keysound_index: Optional SSC keysound index.
    """

    beat: fractions.Fraction
    column: int
    note_type: NoteType
    player: int = 0
    keysound_index: int | None = None


@dataclasses.dataclass(frozen=True)
class TimedNote:
    """A note paired with its computed song time in seconds.

    Attributes:
        time: Song timestamp in seconds.
        note: Underlying parsed note.
    """

    time: float
    note: Note


@dataclasses.dataclass
class ParsedChartBlock:
    """One chart block parsed from an ``.sm`` or ``.ssc`` simfile.

    Attributes:
        properties: Uppercase MSD key-value mapping for the chart.
        is_ssc: True when parsed from an ``.ssc`` ``#NOTEDATA`` block.
    """

    properties: dict[str, str]
    is_ssc: bool = False

    @property
    def stepstype(self) -> str | None:
        """Chart ``#STEPSTYPE`` (e.g. ``dance-single``)."""
        return self.properties.get("STEPSTYPE")

    @property
    def description(self) -> str | None:
        """Chart ``#DESCRIPTION``."""
        return self.properties.get("DESCRIPTION")

    @property
    def difficulty(self) -> str | None:
        """Chart ``#DIFFICULTY``."""
        return self.properties.get("DIFFICULTY")

    @property
    def meter(self) -> str | None:
        """Chart ``#METER``."""
        return self.properties.get("METER")

    @property
    def chartname(self) -> str | None:
        """SSC ``#CHARTNAME`` (``None`` for ``.sm`` charts)."""
        return self.properties.get("CHARTNAME")

    @property
    def credit(self) -> str | None:
        """SSC ``#CREDIT`` (``None`` for ``.sm`` charts)."""
        return self.properties.get("CREDIT")

    @property
    def notes(self) -> str | None:
        """Chart note data (``#NOTES``, or ``#NOTES2`` alias in SSC)."""
        if "NOTES" in self.properties:
            return self.properties["NOTES"]
        if self.is_ssc:
            return self.properties.get("NOTES2")
        return None

    @property
    def bpms(self) -> str | None:
        """Chart-level ``#BPMS`` override (SSC)."""
        return self.properties.get("BPMS")

    @property
    def stops(self) -> str | None:
        """Chart-level ``#STOPS`` override (SSC)."""
        return self.properties.get("STOPS")

    @property
    def delays(self) -> str | None:
        """Chart-level ``#DELAYS`` override (SSC)."""
        return self.properties.get("DELAYS")

    @property
    def warps(self) -> str | None:
        """Chart-level ``#WARPS`` override (SSC)."""
        return self.properties.get("WARPS")

    @property
    def offset(self) -> str | None:
        """Chart-level ``#OFFSET`` override (SSC)."""
        return self.properties.get("OFFSET")


@dataclasses.dataclass
class ParsedSimfile:
    """Song-level properties and charts parsed from an ``.sm`` or ``.ssc`` file.

    Attributes:
        properties: Uppercase song-level MSD key-value mapping.
        charts: Ordered list of parsed chart blocks.
        is_ssc: True when parsed as ``.ssc`` format, False for ``.sm``.
    """

    properties: dict[str, str]
    charts: list[ParsedChartBlock]
    is_ssc: bool = False

    @property
    def version(self) -> str | None:
        """SSC ``#VERSION`` tag."""
        return self.properties.get("VERSION")

    @property
    def title(self) -> str | None:
        """Song ``#TITLE``."""
        return self.properties.get("TITLE")

    @property
    def titletranslit(self) -> str | None:
        """Song ``#TITLETRANSLIT``."""
        return self.properties.get("TITLETRANSLIT")

    @property
    def subtitle(self) -> str | None:
        """Song ``#SUBTITLE``."""
        return self.properties.get("SUBTITLE")

    @property
    def artist(self) -> str | None:
        """Song ``#ARTIST``."""
        return self.properties.get("ARTIST")

    @property
    def music(self) -> str | None:
        """Song ``#MUSIC``."""
        return self.properties.get("MUSIC")

    @property
    def offset(self) -> str | None:
        """Song ``#OFFSET``."""
        return self.properties.get("OFFSET")

    @property
    def bpms(self) -> str | None:
        """Song ``#BPMS``."""
        return self.properties.get("BPMS")

    @property
    def stops(self) -> str | None:
        """Song ``#STOPS`` (with ``#FREEZES`` alias in ``.sm``)."""
        if "STOPS" in self.properties:
            return self.properties["STOPS"]
        if not self.is_ssc:
            return self.properties.get("FREEZES")
        return None

    @property
    def delays(self) -> str | None:
        """Song ``#DELAYS``."""
        return self.properties.get("DELAYS")

    @property
    def warps(self) -> str | None:
        """Song ``#WARPS``."""
        return self.properties.get("WARPS")

    @property
    def selectable(self) -> str | None:
        """Song ``#SELECTABLE``."""
        return self.properties.get("SELECTABLE")


def _parse_sm_chart(components: tuple[str, ...]) -> ParsedChartBlock:
    if len(components) < len(_SM_CHART_PROPERTIES):
        raise ValueError(
            f"expected at least {len(_SM_CHART_PROPERTIES)} "
            f"chart components, got {len(components)}"
        )
    props = {
        key: val.strip()
        for key, val in zip(_SM_CHART_PROPERTIES, components, strict=False)
    }
    return ParsedChartBlock(properties=props, is_ssc=False)


def parse_simfile_text(
    text: str,
    *,
    is_ssc: bool | None = None,
    strict: bool = True,
) -> ParsedSimfile:
    """Parse ``.sm`` or ``.ssc`` text into a :class:`ParsedSimfile`.

    Args:
        text: Decoded simfile text.
        is_ssc: Explicitly select SSC (True) or SM (False); when ``None``,
            auto-detect from whether the first parameter is ``#VERSION``.
        strict: When True, reject stray non-whitespace text between parameters.

    Returns:
        Parsed simfile object.
    """
    params = list(parse_msd(text, strict=strict))
    if is_ssc is None:
        is_ssc = bool(params) and params[0].key.upper() == "VERSION"

    song_props: dict[str, str] = {}
    charts: list[ParsedChartBlock] = []

    if is_ssc:
        current_chart: ParsedChartBlock | None = None
        for param in params:
            key = param.key.upper()
            val = (
                ":".join(param.components[1:])
                if key in _MULTI_VALUE_PROPERTIES
                else param.value
            )
            if key == "NOTEDATA":
                if current_chart is not None:
                    charts.append(current_chart)
                current_chart = ParsedChartBlock(properties={}, is_ssc=True)
            elif current_chart is not None:
                current_chart.properties[key] = val
            else:
                song_props[key] = val
        if current_chart is not None:
            charts.append(current_chart)
    else:
        for param in params:
            key = param.key.upper()
            if key == "NOTES":
                charts.append(_parse_sm_chart(param.components[1:]))
            elif key in _MULTI_VALUE_PROPERTIES:
                song_props[key] = ":".join(param.components[1:])
            else:
                song_props[key] = param.value

    return ParsedSimfile(properties=song_props, charts=charts, is_ssc=is_ssc)


def load_simfile_path(
    sim_path: str | pathlib.Path,
    *,
    encodings: collections.abc.Sequence[str],
    strict: bool = True,
) -> ParsedSimfile:
    """Read and parse a simfile from disk trying encodings in order.

    Args:
        sim_path: Path to ``.ssc`` or ``.sm`` file.
        encodings: Ordered sequence of text encodings to attempt.
        strict: When True, reject stray non-whitespace text between parameters.

    Returns:
        Parsed simfile object.

    Raises:
        UnicodeDecodeError: If every encoding fails to decode the file.
        ValueError: If no encodings are provided.
    """
    path = pathlib.Path(sim_path)
    raw_bytes = path.read_bytes()
    suffix = path.suffix.lower()
    is_ssc: bool | None = (
        True if suffix == ".ssc" else False if suffix == ".sm" else None
    )

    last_error: UnicodeDecodeError | None = None
    decoded: str | None = None
    for encoding in encodings:
        try:
            decoded = raw_bytes.decode(encoding)
            break
        except UnicodeDecodeError as exc:
            last_error = exc

    if decoded is None:
        if last_error is not None:
            raise UnicodeDecodeError(
                last_error.encoding,
                last_error.object,
                last_error.start,
                last_error.end,
                last_error.reason,
            ) from last_error
        raise ValueError(f"no encodings provided to read {path}")

    return parse_simfile_text(decoded, is_ssc=is_ssc, strict=strict)


def _select_timing_source(
    sim: ParsedSimfile,
    chart: ParsedChartBlock | None,
) -> ParsedSimfile | ParsedChartBlock:
    if (
        sim.is_ssc
        and chart is not None
        and chart.is_ssc
        and float(sim.version or "0") >= SSC_VERSION_SPLIT_TIMING
        and any(chart.properties.get(key) for key in _CHART_TIMING_KEYS)
    ):
        return chart
    return sim


@dataclasses.dataclass(frozen=True)
class TimingData:
    """Timing events for a simfile, respecting SSC ``>= 0.7`` split timing.

    Attributes:
        bpms: Parsed ``#BPMS`` events.
        stops: Parsed ``#STOPS`` (or ``#FREEZES``) events.
        delays: Parsed ``#DELAYS`` events.
        warps: Parsed ``#WARPS`` events.
        offset: Song or chart ``#OFFSET`` in seconds.
    """

    bpms: list[BeatValue]
    stops: list[BeatValue]
    delays: list[BeatValue]
    warps: list[BeatValue]
    offset: decimal.Decimal

    @classmethod
    def from_simfile(
        cls,
        sim: ParsedSimfile,
        chart: ParsedChartBlock | None = None,
    ) -> TimingData:
        """Build timing data from a parsed simfile and optional chart.

        Args:
            sim: Parsed simfile.
            chart: Optional chart for SSC split-timing override resolution.

        Returns:
            Resolved ``TimingData`` instance.
        """
        source = _select_timing_source(sim, chart)
        return cls(
            bpms=parse_beat_values(source.bpms),
            stops=parse_beat_values(source.stops),
            delays=parse_beat_values(source.delays),
            warps=parse_beat_values(source.warps),
            offset=decimal.Decimal(source.offset or 0),
        )


class _EventTag(enum.IntEnum):
    WARP = 0
    WARP_END = 1
    BPM = 2
    DELAY = 3
    DELAY_END = 4
    STOP = 5
    STOP_END = 6


@dataclasses.dataclass(frozen=True, order=True)
class _TaggedEvent:
    beat: fractions.Fraction
    tag: _EventTag
    value: decimal.Decimal = dataclasses.field(compare=False)


@dataclasses.dataclass(frozen=True)
class _TimingState:
    beat: fractions.Fraction
    value: decimal.Decimal
    tag: _EventTag
    time: float
    bpm: decimal.Decimal
    warp: bool

    def time_until(self, beat: fractions.Fraction, tag: _EventTag) -> float:
        delta = 0.0 if self.warp else float(beat - self.beat) * 60.0 / float(self.bpm)
        if self.tag in (_EventTag.STOP, _EventTag.DELAY) and tag in (
            _EventTag.STOP_END,
            _EventTag.DELAY_END,
        ):
            delta += float(self.value)
        return delta


class TimingEngine:
    """StepMania beat-to-seconds timing state machine.

    Args:
        timing_data: Song or split-timing chart timing data.

    Raises:
        ValueError: If ``bpms`` is empty or the first BPM is not on beat 0.
    """

    def __init__(self, timing_data: TimingData) -> None:
        self.timing_data = timing_data
        if not timing_data.bpms:
            raise ValueError("timing data has no BPM segments")
        first_bpm = timing_data.bpms[0]
        if first_bpm.beat != 0:
            raise ValueError("first BPM change should be on beat 0")

        states: list[_TimingState] = [
            _TimingState(
                beat=fractions.Fraction(0),
                value=first_bpm.value,
                tag=_EventTag.BPM,
                time=float(-timing_data.offset),
                bpm=first_bpm.value,
                warp=False,
            )
        ]

        warp_starts, warp_ends = self._coalesce_warps(timing_data.warps)
        event_streams = (
            (warp_starts, _EventTag.WARP),
            (warp_ends, _EventTag.WARP_END),
            (timing_data.bpms[1:], _EventTag.BPM),
            (timing_data.delays, _EventTag.DELAY),
            (timing_data.delays, _EventTag.DELAY_END),
            (timing_data.stops, _EventTag.STOP),
            (timing_data.stops, _EventTag.STOP_END),
        )
        chronological = heapq.merge(
            *(
                [_TaggedEvent(beat=ev.beat, tag=tag, value=ev.value) for ev in stream]
                for stream, tag in event_streams
            )
        )

        for event in chronological:
            prev = states[-1]
            time_now = prev.time + prev.time_until(event.beat, event.tag)
            bpm = event.value if event.tag is _EventTag.BPM else prev.bpm
            warp = prev.warp
            if event.tag is _EventTag.WARP:
                warp = True
            elif event.tag is _EventTag.WARP_END:
                warp = False
            states.append(
                _TimingState(
                    beat=event.beat,
                    value=event.value,
                    tag=event.tag,
                    time=time_now,
                    bpm=bpm,
                    warp=warp,
                )
            )

        self._states = states
        self._tagged_beats = [(state.beat, state.tag) for state in states]

    @staticmethod
    def _coalesce_warps(
        warps: list[BeatValue],
    ) -> tuple[list[BeatValue], list[BeatValue]]:
        zero = decimal.Decimal(0)
        starts: list[BeatValue] = []
        ends: list[BeatValue] = []
        for warp in warps:
            warp_end = warp.beat + parse_beat(warp.value)
            if starts and warp.beat <= ends[-1].beat:
                if warp_end > ends[-1].beat:
                    ends[-1] = BeatValue(beat=warp_end, value=zero)
            else:
                starts.append(BeatValue(beat=warp.beat, value=zero))
                ends.append(BeatValue(beat=warp_end, value=zero))
        return starts, ends

    def hittable(self, beat: fractions.Fraction) -> bool:
        """Return True if a note on ``beat`` is hittable (not inside a warp).

        Args:
            beat: Rational beat position.

        Returns:
            True unless the beat falls inside a warp without a stop/delay.
        """
        idx = max(0, bisect.bisect(self._tagged_beats, (beat, _EventTag.STOP_END)) - 1)
        prior = self._states[idx]
        if not prior.warp:
            return True
        return (
            prior.tag in (_EventTag.STOP_END, _EventTag.DELAY_END)
            and beat == prior.beat
        )

    def time_at(self, beat: fractions.Fraction) -> float:
        """Compute song time in seconds for a hittable note on ``beat``.

        Args:
            beat: Rational beat position.

        Returns:
            Timestamp in seconds.
        """
        idx = max(0, bisect.bisect(self._tagged_beats, (beat, _EventTag.STOP)) - 1)
        prior = self._states[idx]
        return prior.time + prior.time_until(beat, _EventTag.STOP)


def _strip_keysound_indices(line: str) -> tuple[str, list[int | None]]:
    cleaned_chars: list[str] = []
    keysounds: list[int | None] = []
    pos = 0
    length = len(line)
    while pos < length:
        ch = line[pos]
        if ch == "[":
            close = line.index("]", pos)
            if keysounds:
                keysounds[-1] = int(line[pos + 1 : close])
            pos = close + 1
            continue
        cleaned_chars.append(ch)
        keysounds.append(None)
        pos += 1
    return "".join(cleaned_chars), keysounds


def iter_chart_notes(chart: ParsedChartBlock) -> collections.abc.Iterator[Note]:
    """Iterate over all non-zero notes in a chart's note data.

    Args:
        chart: Parsed chart block with non-empty ``notes``.

    Yields:
        Note: Notes in measure/row/column order.

    Raises:
        ValueError: If ``chart.notes`` is ``None`` or contains an unknown note
            character.
    """
    raw_notes = chart.notes
    if raw_notes is None:
        raise ValueError("chart has no notes")

    for player, player_block in enumerate(raw_notes.split("&")):
        for measure_idx, measure in enumerate(player_block.split(",")):
            lines = measure.strip().splitlines()
            subdivision = len(lines)
            if subdivision == 0:
                continue
            for line_idx, raw_line in enumerate(lines):
                stripped = raw_line.strip()
                if not stripped:
                    continue
                cleaned, keysounds = _strip_keysound_indices(stripped)
                beat = fractions.Fraction(
                    measure_idx * 4 * subdivision + line_idx * 4,
                    subdivision,
                )
                for col, char in enumerate(cleaned):
                    if char != "0":
                        yield Note(
                            beat=beat,
                            column=col,
                            note_type=NoteType(char),
                            player=player,
                            keysound_index=keysounds[col],
                        )


def time_chart_notes(
    sim: ParsedSimfile,
    chart: ParsedChartBlock,
) -> list[TimedNote]:
    """Convert a chart's notes into timed notes using StepMania timing rules.

    Unhittable notes inside warp segments match StepMania's ``TAP_TO_FAKE``
    behavior: tap notes are converted to ``NoteType.FAKE``, while other note
    types are dropped.

    Args:
        sim: Parsed simfile.
        chart: Parsed chart block.

    Returns:
        List of ``TimedNote`` instances in chart order.
    """
    timing_data = TimingData.from_simfile(sim, chart)
    engine = TimingEngine(timing_data)
    timed_notes: list[TimedNote] = []

    for note in iter_chart_notes(chart):
        if engine.hittable(note.beat):
            timed_notes.append(TimedNote(time=engine.time_at(note.beat), note=note))
        elif note.note_type is NoteType.TAP:
            timed_notes.append(
                TimedNote(
                    time=engine.time_at(note.beat),
                    note=Note(
                        beat=note.beat,
                        column=note.column,
                        note_type=NoteType.FAKE,
                        player=note.player,
                        keysound_index=note.keysound_index,
                    ),
                )
            )
    return timed_notes
