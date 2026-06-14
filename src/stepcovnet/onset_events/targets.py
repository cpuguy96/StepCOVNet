"""Ground-truth onset time padding and duration clipping for event-based onset detection."""

import numpy as np

N_MAX_ONSETS = 1024


def clip_times_to_duration(times: np.ndarray, duration_sec: float) -> np.ndarray:
    """Drop ground-truth onset times beyond an audio duration cap.

    Args:
        times: Sorted onset times in seconds.
        duration_sec: Maximum valid time in seconds (inclusive).

    Returns:
        Sorted array of times with values ``<= duration_sec``, same dtype as
        ``times`` when possible.
    """
    times_arr = np.asarray(times)
    if times_arr.size == 0:
        return np.zeros(0, dtype=times_arr.dtype)
    kept = times_arr[times_arr <= duration_sec]
    return np.asarray(kept, dtype=times_arr.dtype)


def pad_onset_times(
    times: np.ndarray,
    n_max: int = N_MAX_ONSETS,
) -> tuple[np.ndarray, np.ndarray]:
    """Pad sorted onset times to a fixed length with a validity mask.

    Args:
        times: Sorted onset times in seconds; length must be at most ``n_max``.
        n_max: Target length for padded arrays.

    Returns:
        Tuple of ``(times_padded, mask)`` each with shape ``(n_max,)`` and
        dtype float32. ``mask`` is ``1`` for real steps and ``0`` for padding.

    Raises:
        ValueError: If ``len(times)`` exceeds ``n_max``.
    """
    times_arr = np.asarray(times)
    n = int(times_arr.size)
    if n > n_max:
        raise ValueError(
            f"onset count {n} exceeds n_max {n_max}; skip chart or clip first"
        )
    times_padded = np.zeros(n_max, dtype=np.float32)
    mask = np.zeros(n_max, dtype=np.float32)
    if n > 0:
        times_padded[:n] = np.asarray(times_arr, dtype=np.float32)
        mask[:n] = 1.0
    return times_padded, mask
