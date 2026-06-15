"""Shared constants for the dataset preprocessing pipeline (PRE stage)."""

SCHEMA_VERSION = 1

MAX_STEPS_PER_CHART = 2048
MAX_SLUG_LENGTH = 64

DEFAULT_INPUT_DIR = "data/raw_data"
DEFAULT_OUTPUT_DIR = "data/final_data"

EXPORT_MODE_ALL_SINGLES = "export_all_singles"

ENCODING_RETRIES = ("utf-8-sig", "cp932", "shift_jis", "euc-jp")

AUDIO_EXTENSIONS = (".ogg", ".mp3", ".wav", ".flac")

SIMFILE_EXTENSIONS = (".ssc", ".sm")

STANDARD_DIFFICULTIES = frozenset({"beginner", "easy", "medium", "hard", "challenge"})

DIFFICULTY_KIND_STANDARD = "standard"
DIFFICULTY_KIND_CUSTOM = "custom"

DIFFICULTY_RANK = {
    "challenge": 5,
    "hard": 4,
    "medium": 3,
    "easy": 2,
    "beginner": 1,
    "custom": 0,
}

AUDIO_SOURCE_MUSIC_TAG = "music_tag"
AUDIO_SOURCE_INFERRED = "inferred"

PACK_STATUS_OK = "ok"
PACK_STATUS_NO_DANCE_SINGLE = "no_dance_single"
PACK_STATUS_NO_EXPORTABLE_CHARTS = "no_exportable_charts"
PACK_STATUS_NO_AUDIO = "no_audio"
PACK_STATUS_ENCODING_ERROR = "encoding_error"
PACK_STATUS_PARSE_ERROR = "parse_error"
PACK_STATUS_PENDING = "pending"
PACK_STATUS_FAILED = "failed"

CHART_SKIP_OVER_CAP = "chart_skipped_over_cap"
CHART_SKIP_EMPTY = "chart_skipped_empty"
CHART_SKIP_INVALID_HOLDS = "chart_skipped_invalid_holds"

WINDOWS_RESERVED_SLUGS = frozenset(
    {
        "con",
        "prn",
        "aux",
        "nul",
        *{f"com{i}" for i in range(1, 10)},
        *{f"lpt{i}" for i in range(1, 10)},
    }
)
