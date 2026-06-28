"""Tokenization and alignment targets for AR onset (delta_bucketed + pointer/residual)."""

from __future__ import annotations

import dataclasses
import math

import numpy as np

from stepcovnet import constants
from stepcovnet.onset_events import targets as event_targets

PAD_ID = 0
BOS_ID = 1
EOS_ID = 2

DEFAULT_DELTA_MAX_DENSE = 256
DEFAULT_N_LOG_BUCKETS = 16
DEFAULT_N_FIRST_ABS_BINS = 64

SPECIAL_TOKEN_IDS = frozenset({PAD_ID, BOS_ID, EOS_ID})


@dataclasses.dataclass(frozen=True)
class DeltaBucketVocab:
    """Finite vocabulary for ``delta_bucketed`` AR targets."""

    delta_max_dense: int = DEFAULT_DELTA_MAX_DENSE
    n_log_buckets: int = DEFAULT_N_LOG_BUCKETS
    n_first_abs_bins: int = DEFAULT_N_FIRST_ABS_BINS
    hop_sec: float = constants.HOP_COEFF

    @property
    def first_abs_start(self) -> int:
        return 3

    @property
    def dense_delta_start(self) -> int:
        return self.first_abs_start + self.n_first_abs_bins

    @property
    def log_delta_start(self) -> int:
        return self.dense_delta_start + self.delta_max_dense

    @property
    def vocab_size(self) -> int:
        return self.log_delta_start + self.n_log_buckets

    def _first_abs_edges(self, max_frame: int) -> np.ndarray:
        max_frame = max(1, int(max_frame))
        edges = np.logspace(
            0,
            math.log10(max(max_frame, 2)),
            num=self.n_first_abs_bins + 1,
            base=10.0,
        )
        edges[0] = 0.0
        edges[-1] = float(max_frame)
        return edges.astype(np.float64)

    def encode_first_frame(self, frame_idx: int, *, max_frame: int) -> int:
        frame_idx = max(0, int(frame_idx))
        edges = self._first_abs_edges(max_frame)
        bin_idx = int(np.searchsorted(edges, frame_idx, side="right") - 1)
        bin_idx = min(max(bin_idx, 0), self.n_first_abs_bins - 1)
        return self.first_abs_start + bin_idx

    def decode_first_frame(self, token_id: int, *, max_frame: int) -> int:
        bin_idx = token_id - self.first_abs_start
        bin_idx = min(max(bin_idx, 0), self.n_first_abs_bins - 1)
        edges = self._first_abs_edges(max_frame)
        lo = int(edges[bin_idx])
        hi = int(edges[bin_idx + 1])
        return (lo + hi) // 2

    def encode_delta_frames(self, delta_frames: int) -> int:
        delta_frames = max(1, int(delta_frames))
        if delta_frames <= self.delta_max_dense:
            return self.dense_delta_start + delta_frames - 1
        log_idx = self._log_bucket_index(delta_frames)
        return self.log_delta_start + log_idx

    def decode_delta_frames(self, token_id: int) -> int:
        if self.dense_delta_start <= token_id < self.log_delta_start:
            return token_id - self.dense_delta_start + 1
        if self.log_delta_start <= token_id < self.vocab_size:
            log_idx = token_id - self.log_delta_start
            return self._log_bucket_value(log_idx)
        raise ValueError(f"token_id {token_id} is not a delta token")

    def is_first_abs_token(self, token_id: int) -> bool:
        return self.first_abs_start <= int(token_id) < self.dense_delta_start

    def is_delta_token(self, token_id: int) -> bool:
        token_id = int(token_id)
        return self.dense_delta_start <= token_id < self.vocab_size

    def _log_bucket_index(self, delta_frames: int) -> int:
        lo = self.delta_max_dense + 1
        hi = max(lo + 1, lo * (2**self.n_log_buckets))
        edges = np.logspace(
            math.log10(lo),
            math.log10(hi),
            num=self.n_log_buckets + 1,
            base=10.0,
        )
        idx = int(np.searchsorted(edges, delta_frames, side="right") - 1)
        return min(max(idx, 0), self.n_log_buckets - 1)

    def _log_bucket_value(self, log_idx: int) -> int:
        lo = self.delta_max_dense + 1
        hi = max(lo + 1, lo * (2**self.n_log_buckets))
        edges = np.logspace(
            math.log10(lo),
            math.log10(hi),
            num=self.n_log_buckets + 1,
            base=10.0,
        )
        log_idx = min(max(int(log_idx), 0), self.n_log_buckets - 1)
        left = int(edges[log_idx])
        right = int(edges[log_idx + 1])
        return max(1, (left + right) // 2)


@dataclasses.dataclass(frozen=True)
class OnsetTokenSequence:
    """Teacher-forcing token targets for one chart."""

    token_ids: np.ndarray
    frame_indices: np.ndarray
    patch_indices: np.ndarray
    residual_sec: np.ndarray
    decoder_input_ids: np.ndarray
    decoder_target_ids: np.ndarray

    @property
    def n_steps(self) -> int:
        return int(self.token_ids.size)


def times_to_frame_indices(times_sec: np.ndarray, hop_sec: float) -> np.ndarray:
    """Convert sorted onset times to monotonic frame indices on the hop grid."""
    if times_sec.size == 0:
        return np.zeros(0, dtype=np.int32)
    frames = np.floor(np.asarray(times_sec, dtype=np.float64) / hop_sec + 1e-9).astype(
        np.int32,
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
    """Map a hop frame index to monotonic patch pointer and within-patch residual."""
    patch_frames = max(1, int(patch_frames))
    patch_idx = frame_idx // patch_frames
    frame_in_patch = frame_idx % patch_frames
    residual_sec = float(frame_in_patch) * hop_sec
    return int(patch_idx), residual_sec


def encode_onset_times(
    times_sec: np.ndarray,
    *,
    duration_sec: float,
    hop_sec: float,
    patch_frames: int,
    vocab: DeltaBucketVocab,
    max_steps: int = constants.MAX_STEPS,
) -> OnsetTokenSequence:
    """Encode clipped sorted onset times into AR token and pointer targets."""
    clipped = event_targets.clip_times_to_duration(times_sec, duration_sec)
    frames = times_to_frame_indices(clipped, hop_sec)
    if frames.size > max_steps:
        raise ValueError(
            f"onset count {frames.size} exceeds max_steps {max_steps}; skip chart or clip first",
        )
    if frames.size == 0:
        empty = np.zeros(0, dtype=np.int32)
        decoder_target = np.asarray([EOS_ID], dtype=np.int32)
        decoder_input = np.asarray([BOS_ID], dtype=np.int32)
        return OnsetTokenSequence(
            token_ids=empty,
            frame_indices=empty,
            patch_indices=empty,
            residual_sec=np.zeros(0, dtype=np.float32),
            decoder_input_ids=decoder_input,
            decoder_target_ids=decoder_target,
        )

    max_frame = max(int(frames[-1]), 1)
    token_ids: list[int] = []
    patch_indices: list[int] = []
    residual_sec: list[float] = []
    token_ids.append(vocab.encode_first_frame(int(frames[0]), max_frame=max_frame))
    patch_idx, residual = frame_to_pointer(
        int(frames[0]),
        patch_frames=patch_frames,
        hop_sec=hop_sec,
    )
    patch_indices.append(patch_idx)
    residual_sec.append(residual)
    for prev_frame, frame in zip(frames[:-1], frames[1:], strict=True):
        delta = max(1, int(frame) - int(prev_frame))
        token_ids.append(vocab.encode_delta_frames(delta))
        patch_idx, residual = frame_to_pointer(
            int(frame),
            patch_frames=patch_frames,
            hop_sec=hop_sec,
        )
        patch_indices.append(patch_idx)
        residual_sec.append(residual)

    token_arr = np.asarray(token_ids, dtype=np.int32)
    frame_arr = frames.astype(np.int32)
    patch_arr = np.asarray(patch_indices, dtype=np.int32)
    residual_arr = np.asarray(residual_sec, dtype=np.float32)
    decoder_target = np.concatenate([token_arr, np.asarray([EOS_ID], dtype=np.int32)])
    decoder_input = np.concatenate([np.asarray([BOS_ID], dtype=np.int32), token_arr])
    return OnsetTokenSequence(
        token_ids=token_arr,
        frame_indices=frame_arr,
        patch_indices=patch_arr,
        residual_sec=residual_arr,
        decoder_input_ids=decoder_input,
        decoder_target_ids=decoder_target,
    )


def decode_token_sequence_to_times(
    token_ids: np.ndarray,
    *,
    hop_sec: float,
    vocab: DeltaBucketVocab,
    max_frame: int,
) -> np.ndarray:
    """Detokenize AR ids back to seconds (diagnostics / round-trip tests)."""
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
    frame = vocab.decode_first_frame(ids[first_idx], max_frame=max_frame)
    frames = [frame]
    for token in ids[first_idx + 1 :]:
        if not vocab.is_delta_token(token):
            continue
        frame += vocab.decode_delta_frames(token)
        frames.append(frame)
    return (np.asarray(frames, dtype=np.float64) * hop_sec).astype(np.float32)


def decode_pointer_residual_to_times(
    patch_indices: np.ndarray,
    residual_sec: np.ndarray,
    *,
    patch_frames: int,
    hop_sec: float,
) -> np.ndarray:
    """Convert pointer+residual targets to seconds."""
    patch_frames = max(1, int(patch_frames))
    times = (
        np.asarray(patch_indices, dtype=np.float64) * patch_frames * hop_sec
    ) + np.asarray(
        residual_sec,
        dtype=np.float64,
    )
    return times.astype(np.float32)
