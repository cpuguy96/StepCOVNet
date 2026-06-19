"""Pack outcome and reason codes for preprocess manifests."""

from __future__ import annotations

PACK_RESULT_PENDING = "pack_pending"
PACK_RESULT_EXPORTED = "pack_exported"
PACK_RESULT_SKIPPED = "pack_skipped"
PACK_RESULT_ERROR = "pack_error"

REASON_NO_DANCE_SINGLE = "no_dance_single"
REASON_NO_EXPORTABLE_CHARTS = "no_exportable_charts"
REASON_NO_AUDIO = "no_audio"
REASON_ENCODING_ERROR = "encoding_error"
REASON_PARSE_ERROR = "parse_error"
REASON_OUTPUT_EXISTS = "output_exists"
REASON_VALIDATION_FAILED = "validation_failed"
REASON_IO_ERROR = "io_error"

SKIP_REASONS = frozenset(
    {
        REASON_NO_DANCE_SINGLE,
        REASON_NO_EXPORTABLE_CHARTS,
        REASON_NO_AUDIO,
        REASON_ENCODING_ERROR,
        REASON_PARSE_ERROR,
    }
)

ERROR_REASONS = frozenset(
    {
        REASON_OUTPUT_EXISTS,
        REASON_VALIDATION_FAILED,
        REASON_IO_ERROR,
    }
)


def pack_result(reason: str | None) -> str:
    """Map a pack reason to a coarse outcome.

    Args:
        reason: ``None`` for a successful export; otherwise a skip or error code.

    Returns:
        One of ``pack_pending``, ``pack_exported``, ``pack_skipped``, or ``pack_error``.
    """
    if reason is None:
        return PACK_RESULT_EXPORTED
    if reason in SKIP_REASONS:
        return PACK_RESULT_SKIPPED
    return PACK_RESULT_ERROR


def pack_entry_row(
    *,
    source_pack: str,
    normalized_bundle: str,
    normalized_id: str,
    reason: str,
    warnings: list[str],
    message: str = "",
) -> dict:
    """Build a report row for a non-exported pack.

    Args:
        source_pack: Raw pack path relative to the preprocess input root.
        normalized_bundle: Output bundle slug.
        normalized_id: Output song slug within the bundle.
        reason: Skip or error reason code.
        warnings: Pack-level warning codes accumulated during processing.
        message: Optional error detail for ``pack_error`` outcomes.

    Returns:
        Report row dict suitable for ``skipped_packs`` or ``pack_errors``.
    """
    row = {
        "source_pack": source_pack,
        "normalized_bundle": normalized_bundle,
        "normalized_id": normalized_id,
        "result": pack_result(reason),
        "reason": reason,
        "warnings": warnings,
    }
    if message:
        row["message"] = message
    return row
