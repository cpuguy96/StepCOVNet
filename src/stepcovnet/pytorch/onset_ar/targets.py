"""Pure-NumPy tokenization and alignment targets for PyTorch AR onset models."""

from __future__ import annotations

import dataclasses

import numpy as np

from stepcovnet import constants
from stepcovnet.onset_ar.targets import (
    DeltaBucketVocab as DeltaBucketVocab,
)
from stepcovnet.onset_ar.targets import (
    PatchGapVocab as PatchGapVocab,
)

PAD_ID = 0
BOS_ID = 1
EOS_ID = 2

DEFAULT_DELTA_MAX_DENSE = 256
DEFAULT_N_LOG_BUCKETS = 16
DEFAULT_N_FIRST_ABS_BINS = 64
DEFAULT_PATCH_DELTA_MAX_DENSE = 256
DEFAULT_PATCH_N_LOG_BUCKETS = 16

SPECIAL_TOKEN_IDS = frozenset({PAD_ID, BOS_ID, EOS_ID})


def clip_times_to_duration(times_sec: np.ndarray, duration_sec: float) -> np.ndarray:
    """Keep onset times inside ``[0, duration_sec)``, sorted ascending.

    Args:
        times_sec: Raw onset times in seconds.
        duration_sec: Audio duration upper bound in seconds.

    Returns:
        Sorted float64 array of valid onset times.
    """
    arr = np.asarray(times_sec, dtype=np.float64).reshape(-1)
    if arr.size == 0 or duration_sec <= 0.0:
        return np.zeros(0, dtype=np.float64)
    kept = arr[(arr >= 0.0) & (arr < float(duration_sec))]
    return np.sort(kept)


@dataclasses.dataclass(frozen=True)
class OnsetTokenSequence:
    """Teacher-forcing token and alignment targets for one chart.

    Attributes:
        token_ids: Token-LM ids, length ``n_steps``.
        frame_indices: Hop-frame indices per onset.
        patch_indices: Absolute patch indices per onset.
        residual_sec: Within-patch residual seconds.
        delta_patches: Raw patch deltas (first onset = absolute patch).
        gap_token_ids: Patch-gap vocabulary ids.
        decoder_input_ids: Teacher-forcing decoder inputs (BOS + tokens).
        decoder_target_ids: Teacher-forcing decoder targets (tokens + EOS).
    """

    token_ids: np.ndarray
    frame_indices: np.ndarray
    patch_indices: np.ndarray
    residual_sec: np.ndarray
    delta_patches: np.ndarray
    gap_token_ids: np.ndarray
    decoder_input_ids: np.ndarray
    decoder_target_ids: np.ndarray

    @property
    def n_steps(self) -> int:
        """Number of onset tokens."""
        return int(self.token_ids.size)


def times_to_frame_indices(times_sec: np.ndarray, hop_sec: float) -> np.ndarray:
    """Convert sorted onset times to strictly increasing hop-frame indices.

    Args:
        times_sec: Sorted onset times in seconds.
        hop_sec: Seconds per hop frame.

    Returns:
        Strictly increasing int32 hop-frame indices.
    """
    if times_sec.size == 0:
        return np.zeros(0, dtype=np.int32)
    frames = np.floor(np.asarray(times_sec, dtype=np.float64) / hop_sec + 1e-9).astype(
        np.int32
    )
    frames = np.maximum(frames, 0)
    deduped: list[int] = []
    last = -1
    for frame in frames:
        frame_int = int(frame)
        if frame_int <= last:
            continue
        deduped.append(frame_int)
        last = frame_int
    return np.asarray(deduped, dtype=np.int32)


def frame_to_pointer(
    frame_idx: int,
    *,
    patch_frames: int,
    hop_sec: float,
) -> tuple[int, float]:
    """Map a hop frame index to ``(patch_idx, residual_sec)``.

    Args:
        frame_idx: Hop frame index.
        patch_frames: Number of frames per encoder patch.
        hop_sec: Seconds per hop frame.

    Returns:
        Tuple ``(patch_idx, residual_sec)``.
    """
    patch_frames = max(1, int(patch_frames))
    patch_idx = frame_idx // patch_frames
    frame_in_patch = frame_idx % patch_frames
    return int(patch_idx), float(frame_in_patch) * hop_sec


def gap_delta_lookup_table(gap_vocab: PatchGapVocab) -> np.ndarray:
    """Return ``decode_delta(id)`` for every gap-vocab id.

    Args:
        gap_vocab: Patch-gap vocabulary.

    Returns:
        Int32 lookup array of shape ``(vocab_size,)``.
    """
    return np.asarray(
        [gap_vocab.decode_delta(gap_id) for gap_id in range(gap_vocab.vocab_size)],
        dtype=np.int32,
    )


def patch_delta_targets(patch_indices: np.ndarray) -> np.ndarray:
    """Return per-step patch deltas with first onset equal to its absolute patch.

    Args:
        patch_indices: Non-decreasing absolute patch indices ``(N,)``.

    Returns:
        Int32 patch deltas ``(N,)``.
    """
    patches = np.asarray(patch_indices, dtype=np.int32)
    if patches.size == 0:
        return np.zeros(0, dtype=np.int32)
    deltas = np.empty(patches.shape, dtype=np.int32)
    deltas[0] = patches[0]
    if patches.size > 1:
        deltas[1:] = patches[1:] - patches[:-1]
    return deltas


def decode_gap_ids_to_patch_indices(
    gap_token_ids: np.ndarray,
    *,
    gap_vocab: PatchGapVocab,
    max_patch: int,
) -> np.ndarray:
    """Resolve gap-vocab ids to clamped absolute patch indices.

    Args:
        gap_token_ids: Sequence of gap vocabulary ids.
        gap_vocab: Vocabulary used to decode each gap.
        max_patch: Inclusive upper clamp on patch index.

    Returns:
        Int32 absolute patch indices.
    """
    ids = np.asarray(gap_token_ids, dtype=np.int32)
    if ids.size == 0:
        return np.zeros(0, dtype=np.int32)
    max_patch = max(0, int(max_patch))
    patches = np.empty(ids.shape, dtype=np.int32)
    prev = 0
    for i, gap_id in enumerate(ids.tolist()):
        patch = min(max(prev + gap_vocab.decode_delta(int(gap_id)), 0), max_patch)
        patches[i] = patch
        prev = patch
    return patches


def encode_onset_times(
    times_sec: np.ndarray,
    *,
    duration_sec: float,
    hop_sec: float,
    patch_frames: int,
    vocab: DeltaBucketVocab,
    gap_vocab: PatchGapVocab | None = None,
    max_steps: int = constants.MAX_STEPS,
) -> OnsetTokenSequence:
    """Encode sorted onset times into AR token and alignment targets.

    Args:
        times_sec: Onset times in seconds.
        duration_sec: Clip horizon for chart times.
        hop_sec: Seconds per hop frame.
        patch_frames: Frames per encoder patch.
        vocab: Token-LM vocabulary.
        gap_vocab: Optional patch-gap vocabulary.
        max_steps: Maximum allowed onset count.

    Returns:
        Populated :class:`OnsetTokenSequence`.

    Raises:
        ValueError: If onset count exceeds ``max_steps``.
    """
    gap_vocab = gap_vocab or PatchGapVocab()
    clipped = clip_times_to_duration(times_sec, duration_sec)
    frames = times_to_frame_indices(clipped, hop_sec)
    if frames.size > max_steps:
        raise ValueError(
            f"onset count {frames.size} exceeds max_steps {max_steps}; clip or skip"
        )
    if frames.size == 0:
        empty = np.zeros(0, dtype=np.int32)
        return OnsetTokenSequence(
            token_ids=empty,
            frame_indices=empty,
            patch_indices=empty,
            residual_sec=np.zeros(0, dtype=np.float32),
            delta_patches=empty,
            gap_token_ids=empty,
            decoder_input_ids=np.asarray([BOS_ID], dtype=np.int32),
            decoder_target_ids=np.asarray([EOS_ID], dtype=np.int32),
        )

    max_frame = max(int(frames[-1]), 1)
    token_ids: list[int] = [
        vocab.encode_first_frame(int(frames[0]), max_frame=max_frame)
    ]
    patch0, res0 = frame_to_pointer(
        int(frames[0]), patch_frames=patch_frames, hop_sec=hop_sec
    )
    patch_indices: list[int] = [patch0]
    residual_sec: list[float] = [res0]
    for prev_frame, frame in zip(frames[:-1], frames[1:], strict=True):
        delta = max(1, int(frame) - int(prev_frame))
        token_ids.append(vocab.encode_delta_frames(delta))
        patch_idx, residual = frame_to_pointer(
            int(frame), patch_frames=patch_frames, hop_sec=hop_sec
        )
        patch_indices.append(patch_idx)
        residual_sec.append(residual)

    token_arr = np.asarray(token_ids, dtype=np.int32)
    frame_arr = frames.astype(np.int32)
    patch_arr = np.asarray(patch_indices, dtype=np.int32)
    residual_arr = np.asarray(residual_sec, dtype=np.float32)
    delta_arr = patch_delta_targets(patch_arr)
    gap_ids = np.asarray(
        [gap_vocab.encode_delta(int(delta)) for delta in delta_arr.tolist()],
        dtype=np.int32,
    )
    return OnsetTokenSequence(
        token_ids=token_arr,
        frame_indices=frame_arr,
        patch_indices=patch_arr,
        residual_sec=residual_arr,
        delta_patches=delta_arr,
        gap_token_ids=gap_ids,
        decoder_input_ids=np.concatenate(
            [np.asarray([BOS_ID], dtype=np.int32), token_arr]
        ),
        decoder_target_ids=np.concatenate(
            [token_arr, np.asarray([EOS_ID], dtype=np.int32)]
        ),
    )


def decode_token_sequence_to_times(
    token_ids: np.ndarray,
    *,
    hop_sec: float,
    vocab: DeltaBucketVocab,
    max_frame: int,
) -> np.ndarray:
    """Detokenize AR token ids back to onset times in seconds.

    Args:
        token_ids: Token id sequence (special tokens ignored).
        hop_sec: Seconds per hop frame.
        vocab: Token vocabulary.
        max_frame: Song length in hop frames.

    Returns:
        Float32 array of decoded onset times in seconds.
    """
    ids = [
        int(token)
        for token in np.asarray(token_ids).tolist()
        if int(token) not in SPECIAL_TOKEN_IDS
    ]
    if not ids:
        return np.zeros(0, dtype=np.float32)
    first_idx = next(
        (idx for idx, token in enumerate(ids) if vocab.is_first_abs_token(token)),
        None,
    )
    if first_idx is None:
        return np.zeros(0, dtype=np.float32)
    cur_frame = vocab.decode_first_frame(ids[first_idx], max_frame=max_frame)
    frames = [cur_frame]
    for token in ids[first_idx + 1 :]:
        if vocab.is_delta_token(token):
            cur_frame += vocab.decode_delta_frames(token)
            frames.append(cur_frame)
    return (np.asarray(frames, dtype=np.float32)) * float(hop_sec)
