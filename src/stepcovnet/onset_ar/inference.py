"""Decode AR onset model outputs into event times for evaluation."""

from __future__ import annotations

import dataclasses
from typing import Literal

import numpy as np
import tensorflow as tf

from stepcovnet.onset_ar import config, kv_decode, losses, targets
from stepcovnet.onset_ar import models as ar_models

ArTimeSource = Literal["pointer_residual", "tokens"]


def _density_array_for_decoder(
    density_scalar: np.ndarray | float | None,
    *,
    experiment_config: config.ArExperimentConfig,
    batch_size: int = 1,
) -> np.ndarray | None:
    """Return a batched ``(N, 1)`` density feature when conditioning is enabled."""
    if not config.density_conditioning_active(experiment_config.model):
        return None
    if density_scalar is None:
        density_scalar = 0.0
    arr = np.asarray(density_scalar, dtype=np.float32).reshape(-1)
    if arr.size == 1 and batch_size > 1:
        arr = np.repeat(arr, batch_size)
    return arr.reshape(-1, 1)


def _with_density_decoder_inputs(
    inputs: dict[str, np.ndarray],
    *,
    experiment_config: config.ArExperimentConfig,
    density_scalar: np.ndarray | float | None = None,
    batch_size: int = 1,
) -> dict[str, np.ndarray]:
    """Attach ``density_scalar`` to decoder inputs when the model expects it."""
    density = _density_array_for_decoder(
        density_scalar,
        experiment_config=experiment_config,
        batch_size=batch_size,
    )
    if density is None:
        return inputs
    merged = dict(inputs)
    merged["density_scalar"] = density
    return merged


@dataclasses.dataclass(frozen=True)
class ArDecodeStats:
    """Free-running autoregressive decode summary.

    Attributes:
        times: Predicted onset times in seconds.
        n_forward_steps: Decoder forward passes run.
        n_onset_tokens: Onset tokens emitted before stopping.
        stopped_on_eos: Whether decode ended on ``<EOS>`` rather than the cap.
        onset_token_ids: Emitted onset token ids, when available.
        eos_prob_trace: Per-step ``<EOS>`` probability before length control.
    """

    times: np.ndarray
    n_forward_steps: int
    n_onset_tokens: int
    stopped_on_eos: bool
    onset_token_ids: np.ndarray | None = None
    eos_prob_trace: np.ndarray | None = None


@dataclasses.dataclass(frozen=True)
class ArLengthControl:
    """Decode-time constraints on free-run sequence length.

    Attributes:
        eos_logit_bias: Added to the ``<EOS>`` logit before argmax; negative
            values discourage stopping.
        min_onset_tokens: Suppress ``<EOS>`` until this many onset tokens have
            been emitted.
    """

    eos_logit_bias: float = 0.0
    min_onset_tokens: int = 0

    def is_active(self) -> bool:
        """Return whether this control changes greedy decoding at all."""
        return self.eos_logit_bias != 0.0 or self.min_onset_tokens > 0


def eos_probability(token_logits: np.ndarray, *, eos_id: int = targets.EOS_ID) -> float:
    """Return the softmax probability of ``<EOS>`` for one step's token logits.

    Args:
        token_logits: Unnormalized token scores for a single decoder position.
        eos_id: Vocabulary id of the end-of-sequence token.
    """
    logits = np.asarray(token_logits, dtype=np.float64).reshape(-1)
    shifted = np.exp(logits - float(np.max(logits)))
    return float(shifted[eos_id] / max(float(np.sum(shifted)), 1e-12))


def select_next_token(
    token_logits: np.ndarray,
    *,
    eos_id: int = targets.EOS_ID,
    n_emitted: int = 0,
    length_control: ArLengthControl | None = None,
) -> int:
    """Return the greedy next token id after applying ``length_control``.

    Args:
        token_logits: Unnormalized token scores for a single decoder position.
        eos_id: Vocabulary id of the end-of-sequence token.
        n_emitted: Onset tokens emitted so far in this sequence.
        length_control: Optional length constraints; ``None`` is plain argmax.
    """
    logits = np.asarray(token_logits, dtype=np.float32)
    if length_control is None or not length_control.is_active():
        return int(np.argmax(logits))
    logits = logits.copy()
    if n_emitted < length_control.min_onset_tokens:
        logits[eos_id] = -np.inf
    else:
        logits[eos_id] += float(length_control.eos_logit_bias)
    return int(np.argmax(logits))


@dataclasses.dataclass
class _EncoderMemoryCache:
    """Cached encoder output for one MERT patch sequence."""

    fingerprint: tuple[int | float, ...]
    memory: np.ndarray
    patch_mask: np.ndarray


_ENCODER_MEMORY_CACHE_ATTR = "_ar_onset_encoder_memory_cache"


def _mert_input_fingerprint(
    mert_patches: np.ndarray,
    patch_mask: np.ndarray,
) -> tuple[int | float, ...]:
    mp = np.asarray(mert_patches, dtype=np.float32)
    pm = np.asarray(patch_mask, dtype=np.float32)
    if mp.ndim == 3:
        mp = mp[0]
    if pm.ndim == 2:
        pm = pm[0]
    valid = int(np.sum(pm > 0.5))
    if valid == 0:
        return (tuple(int(dim) for dim in mp.shape), 0)
    last = valid - 1
    return (
        tuple(int(dim) for dim in mp.shape),
        valid,
        float(mp[0, 0]),
        float(mp[last, 0]),
        float(pm[last]),
    )


def clear_encoder_memory_cache(model: tf.keras.Model) -> None:
    """Drop cached encoder memory on ``model`` (for tests or a new song)."""
    if hasattr(model, _ENCODER_MEMORY_CACHE_ATTR):
        delattr(model, _ENCODER_MEMORY_CACHE_ATTR)


def get_encoder_memory_numpy(
    model: tf.keras.Model,
    mert_patches: np.ndarray,
    patch_mask: np.ndarray,
    experiment_config: config.ArExperimentConfig,
    *,
    use_cache: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Run encoder once per unique MERT input; reuse cached memory when possible."""
    mert_patches, patch_mask = _batch_mert_patch_mask(mert_patches, patch_mask)
    fingerprint = _mert_input_fingerprint(mert_patches, patch_mask)
    if use_cache:
        cached = getattr(model, _ENCODER_MEMORY_CACHE_ATTR, None)
        if (
            isinstance(cached, _EncoderMemoryCache)
            and cached.fingerprint == fingerprint
        ):
            return cached.memory, cached.patch_mask
    encoder, _ = _infer_encoder_decoder(model, experiment_config)
    memory = encoder(
        {"mert_patches": mert_patches, "patch_mask": patch_mask},
        training=False,
    ).numpy()
    if use_cache:
        setattr(
            model,
            _ENCODER_MEMORY_CACHE_ATTR,
            _EncoderMemoryCache(fingerprint, memory, patch_mask),
        )
    return memory, patch_mask


def get_inference_encoder_decoder(
    model: tf.keras.Model,
    experiment_config: config.ArExperimentConfig,
) -> tuple[tf.keras.Model, tf.keras.Model]:
    """Return cached encoder/decoder inference submodels."""
    return _infer_encoder_decoder(model, experiment_config)


def _infer_encoder_decoder(
    model: tf.keras.Model,
    experiment_config: config.ArExperimentConfig,
) -> tuple[tf.keras.Model, tf.keras.Model]:
    cache_key = "_ar_onset_infer_models"
    infer_models = getattr(model, cache_key, None)
    if infer_models is None:
        infer_models = ar_models.build_ar_onset_inference_models(
            model,
            experiment_config,
        )
        setattr(model, cache_key, infer_models)
    return infer_models


def _batch_mert_patch_mask(
    mert_patches: np.ndarray,
    patch_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    mert_patches = np.asarray(mert_patches, dtype=np.float32)
    patch_mask = np.asarray(patch_mask, dtype=np.float32)
    if mert_patches.ndim == 2:
        mert_patches = mert_patches[np.newaxis, ...]
        patch_mask = patch_mask[np.newaxis, ...]
    return mert_patches, patch_mask


def build_decoder_inputs_for_onset_tokens(
    onset_token_ids: np.ndarray,
    *,
    max_decoder_len: int,
    bos_id: int = targets.BOS_ID,
) -> tuple[np.ndarray, np.ndarray]:
    """Build padded teacher-forcing inputs for ``[BOS, t0, …, t_{n-1}]``."""
    tokens = np.asarray(onset_token_ids, dtype=np.int32).reshape(-1)
    decoder_input_ids = np.zeros((1, max_decoder_len), dtype=np.int32)
    decoder_mask = np.zeros((1, max_decoder_len), dtype=np.float32)
    decoder_input_ids[0, 0] = bos_id
    decoder_mask[0, 0] = 1.0
    if tokens.size > 0:
        decoder_input_ids[0, 1 : tokens.size + 1] = tokens
        decoder_mask[0, : tokens.size + 1] = 1.0
    return decoder_input_ids, decoder_mask


def decode_parallel_pointer_times_numpy(
    model: tf.keras.Model,
    mert_patches: np.ndarray,
    patch_mask: np.ndarray,
    onset_token_ids: np.ndarray,
    *,
    experiment_config: config.ArExperimentConfig,
    encoder_memory: np.ndarray | None = None,
    patch_mask_batched: np.ndarray | None = None,
    density_scalar: np.ndarray | float | None = None,
) -> np.ndarray:
    """One parallel decoder forward; pointer+residual times at each onset step."""
    tokens = np.asarray(onset_token_ids, dtype=np.int32).reshape(-1)
    max_decoder_len = experiment_config.max_decoder_len()
    patch_frames = experiment_config.model.patch_frames
    hop_sec = experiment_config.dataset.hop_sec
    patch_duration = float(patch_frames) * float(hop_sec)

    if encoder_memory is None or patch_mask_batched is None:
        memory, patch_mask = get_encoder_memory_numpy(
            model,
            mert_patches,
            patch_mask,
            experiment_config,
        )
    else:
        memory = encoder_memory
        patch_mask = patch_mask_batched

    _, decoder = _infer_encoder_decoder(model, experiment_config)
    decoder_input_ids, decoder_mask = build_decoder_inputs_for_onset_tokens(
        tokens,
        max_decoder_len=max_decoder_len,
    )
    outputs = decoder(
        _with_density_decoder_inputs(
            {
                "encoder_memory": memory,
                "patch_mask": patch_mask,
                "decoder_input_ids": decoder_input_ids,
                "decoder_mask": decoder_mask,
            },
            experiment_config=experiment_config,
            density_scalar=density_scalar,
            batch_size=int(np.asarray(patch_mask).shape[0]),
        ),
        training=False,
    )
    pointer_logits = np.asarray(outputs["pointer_logits"][0], dtype=np.float32)
    residual_sec = np.asarray(outputs["residual_sec"][0], dtype=np.float32)
    times: list[float] = []
    for pos in range(int(tokens.size)):
        patch_idx = int(np.argmax(pointer_logits[pos]))
        times.append(float(patch_idx) * patch_duration + float(residual_sec[pos]))
    return np.asarray(times, dtype=np.float32)


def decode_gt_incremental_pointer_times_numpy(
    model: tf.keras.Model,
    mert_patches: np.ndarray,
    patch_mask: np.ndarray,
    decoder_input_ids: np.ndarray,
    onset_step_mask: np.ndarray,
    *,
    experiment_config: config.ArExperimentConfig,
    bos_id: int = targets.BOS_ID,
    eos_id: int = targets.EOS_ID,
    density_scalar: np.ndarray | float | None = None,
) -> np.ndarray:
    """GT tokens fed incrementally; pointer+residual at each onset step."""
    mert_patches, patch_mask = _batch_mert_patch_mask(mert_patches, patch_mask)
    decoder_input_ids = np.asarray(decoder_input_ids, dtype=np.int32).reshape(-1)
    onset_step_mask = np.asarray(onset_step_mask, dtype=np.float32).reshape(-1)
    max_decoder_len = experiment_config.max_decoder_len()
    patch_frames = experiment_config.model.patch_frames
    hop_sec = experiment_config.dataset.hop_sec
    patch_duration = float(patch_frames) * float(hop_sec)

    memory, patch_mask = get_encoder_memory_numpy(
        model,
        mert_patches,
        patch_mask,
        experiment_config,
    )
    _, decoder = _infer_encoder_decoder(model, experiment_config)

    dec_in = np.zeros((1, max_decoder_len), dtype=np.int32)
    dec_mask = np.zeros((1, max_decoder_len), dtype=np.float32)
    dec_in[0, 0] = bos_id
    dec_mask[0, 0] = 1.0

    times: list[float] = []
    cur_len = 1
    while cur_len < max_decoder_len:
        outputs = decoder(
            _with_density_decoder_inputs(
                {
                    "encoder_memory": memory,
                    "patch_mask": patch_mask,
                    "decoder_input_ids": dec_in,
                    "decoder_mask": dec_mask,
                },
                experiment_config=experiment_config,
                density_scalar=density_scalar,
            ),
            training=False,
        )
        pos = cur_len - 1
        if onset_step_mask[pos] > 0.5:
            pointer_logits = np.asarray(
                outputs["pointer_logits"][0, pos], dtype=np.float32
            )
            residual_sec = float(outputs["residual_sec"][0, pos].numpy())
            patch_idx = int(np.argmax(pointer_logits))
            times.append(float(patch_idx) * patch_duration + residual_sec)
        next_token = int(decoder_input_ids[cur_len])
        if next_token == eos_id:
            break
        if cur_len >= max_decoder_len - 1:
            break
        dec_in[0, cur_len] = next_token
        dec_mask[0, cur_len] = 1.0
        cur_len += 1
    return np.asarray(times, dtype=np.float32)


def decode_autoregressive_two_pass_with_stats_numpy(
    model: tf.keras.Model,
    mert_patches: np.ndarray,
    patch_mask: np.ndarray,
    *,
    max_decoder_len: int,
    patch_frames: int,
    hop_sec: float,
    experiment_config: config.ArExperimentConfig,
    bos_id: int = targets.BOS_ID,
    eos_id: int = targets.EOS_ID,
    use_kv_cache: bool = True,
    token_pass: ArDecodeStats | None = None,
    length_control: ArLengthControl | None = None,
    density_scalar: np.ndarray | float | None = None,
) -> ArDecodeStats:
    """Parallel pointer+residual re-forward for free-run (or supplied) onset tokens."""
    if token_pass is None:
        token_pass = decode_autoregressive_with_stats_numpy(
            model,
            mert_patches,
            patch_mask,
            max_decoder_len=max_decoder_len,
            patch_frames=patch_frames,
            hop_sec=hop_sec,
            experiment_config=experiment_config,
            bos_id=bos_id,
            eos_id=eos_id,
            use_kv_cache=use_kv_cache,
            time_source="pointer_residual",
            length_control=length_control,
            density_scalar=density_scalar,
        )
    if token_pass.onset_token_ids is None or token_pass.onset_token_ids.size == 0:
        return ArDecodeStats(
            times=np.zeros(0, dtype=np.float32),
            n_forward_steps=token_pass.n_forward_steps + 1,
            n_onset_tokens=0,
            stopped_on_eos=token_pass.stopped_on_eos,
            onset_token_ids=token_pass.onset_token_ids,
            eos_prob_trace=token_pass.eos_prob_trace,
        )
    parallel_times = decode_parallel_pointer_times_numpy(
        model,
        mert_patches,
        patch_mask,
        token_pass.onset_token_ids,
        experiment_config=experiment_config,
        density_scalar=density_scalar,
    )
    return ArDecodeStats(
        times=parallel_times,
        n_forward_steps=token_pass.n_forward_steps + 1,
        n_onset_tokens=int(parallel_times.size),
        stopped_on_eos=token_pass.stopped_on_eos,
        onset_token_ids=token_pass.onset_token_ids,
        eos_prob_trace=token_pass.eos_prob_trace,
    )


def decode_autoregressive_gate_with_stats_numpy(
    model: tf.keras.Model,
    mert_patches: np.ndarray,
    patch_mask: np.ndarray,
    *,
    max_decoder_len: int,
    patch_frames: int,
    hop_sec: float,
    experiment_config: config.ArExperimentConfig,
    bos_id: int = targets.BOS_ID,
    eos_id: int = targets.EOS_ID,
    use_kv_cache: bool = True,
    length_control: ArLengthControl | None = None,
    density_scalar: np.ndarray | float | None = None,
) -> ArDecodeStats:
    """Offline gate decode: incremental tokens, parallel pointer+residual times."""
    return decode_autoregressive_two_pass_with_stats_numpy(
        model,
        mert_patches,
        patch_mask,
        max_decoder_len=max_decoder_len,
        patch_frames=patch_frames,
        hop_sec=hop_sec,
        experiment_config=experiment_config,
        bos_id=bos_id,
        eos_id=eos_id,
        use_kv_cache=use_kv_cache,
        length_control=length_control,
        density_scalar=density_scalar,
    )


def max_hop_frames_for_config(experiment_config: config.ArExperimentConfig) -> int:
    """Upper hop-frame index used for first-token detokenization."""
    return max(
        1,
        int(
            round(
                experiment_config.dataset.max_audio_seconds
                / experiment_config.dataset.hop_sec
            )
        ),
    )


def max_hop_frames_from_patch_mask(
    patch_mask: np.ndarray,
    *,
    patch_frames: int,
) -> int:
    """Hop-frame cap implied by valid encoder patches (matches training encode scale)."""
    mask = np.asarray(patch_mask)
    if mask.ndim == 2:
        mask = mask[0]
    valid_patches = int(np.sum(mask > 0.5))
    return max(1, valid_patches * int(patch_frames) - 1)


def decode_onset_tokens_to_times(
    onset_token_ids: np.ndarray,
    *,
    experiment_config: config.ArExperimentConfig,
    patch_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Map predicted onset token ids to seconds via ``delta_bucketed`` detokenize."""
    vocab = experiment_config.build_vocab()
    if patch_mask is not None:
        max_frame = max_hop_frames_from_patch_mask(
            patch_mask,
            patch_frames=experiment_config.model.patch_frames,
        )
    else:
        max_frame = max_hop_frames_for_config(experiment_config)
    return targets.decode_token_sequence_to_times(
        onset_token_ids,
        hop_sec=experiment_config.dataset.hop_sec,
        vocab=vocab,
        max_frame=max_frame,
    )


def _finalize_decode_times(
    *,
    time_source: ArTimeSource,
    onset_token_ids: list[int],
    pointer_times: list[float],
    experiment_config: config.ArExperimentConfig | None,
    patch_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    token_arr = np.asarray(onset_token_ids, dtype=np.int32)
    if time_source == "tokens":
        if experiment_config is None:
            msg = "experiment_config is required when time_source='tokens'."
            raise ValueError(msg)
        times = decode_onset_tokens_to_times(
            token_arr,
            experiment_config=experiment_config,
            patch_mask=patch_mask,
        )
        return times, token_arr
    return np.asarray(pointer_times, dtype=np.float32), token_arr


def build_scheduled_decoder_inputs(
    decoder_input_ids: tf.Tensor,
    token_logits: tf.Tensor,
    decoder_mask: tf.Tensor,
    sample_prob: tf.Tensor | float,
) -> tf.Tensor:
    """Mix teacher tokens with model argmax predictions for scheduled sampling."""
    sample_prob_f = tf.cast(sample_prob, tf.float32)
    teacher_input = tf.cast(decoder_input_ids, tf.int32)
    predictions = tf.argmax(token_logits, axis=-1, output_type=tf.int32)
    prev_predictions = tf.concat([predictions[:, :1], predictions[:, :-1]], axis=1)
    positions = tf.range(tf.shape(teacher_input)[1])[tf.newaxis, :]
    rand = tf.random.uniform(tf.shape(teacher_input), dtype=tf.float32)
    use_prediction = tf.logical_and(
        decoder_mask > 0.5,
        tf.logical_and(positions > 0, rand < sample_prob_f),
    )
    return tf.where(use_prediction, prev_predictions, teacher_input)


def _decode_autoregressive_prefix_numpy(
    decoder,
    decoder_inputs: dict[str, np.ndarray],
    *,
    max_decoder_len: int,
    patch_duration: float,
    eos_id: int,
    time_source: ArTimeSource = "pointer_residual",
    experiment_config: config.ArExperimentConfig | None = None,
    length_control: ArLengthControl | None = None,
) -> ArDecodeStats:
    """Legacy full-prefix decode loop (one forward per token)."""
    decoder_input = decoder_inputs["decoder_input_ids"]
    decoder_mask_arr = decoder_inputs["decoder_mask"]
    pointer_times: list[float] = []
    onset_token_ids: list[int] = []
    eos_probs: list[float] = []
    cur_len = 1
    n_forward_steps = 0
    stopped_on_eos = False
    while cur_len < max_decoder_len:
        n_forward_steps += 1
        outputs = decoder(decoder_inputs, training=False)
        pos = cur_len - 1
        token_logits = np.asarray(outputs["token_logits"][0, pos], dtype=np.float32)
        pointer_logits = np.asarray(outputs["pointer_logits"][0, pos], dtype=np.float32)
        residual_sec = float(outputs["residual_sec"][0, pos].numpy())
        eos_probs.append(eos_probability(token_logits, eos_id=eos_id))
        next_token = select_next_token(
            token_logits,
            eos_id=eos_id,
            n_emitted=len(onset_token_ids),
            length_control=length_control,
        )
        if next_token == eos_id:
            stopped_on_eos = True
            break
        onset_token_ids.append(next_token)
        patch_idx = int(np.argmax(pointer_logits))
        pointer_times.append(float(patch_idx) * patch_duration + residual_sec)
        if cur_len >= max_decoder_len - 1:
            break
        decoder_input[0, cur_len] = next_token
        decoder_mask_arr[0, cur_len] = 1.0
        decoder_inputs["decoder_input_ids"] = decoder_input
        decoder_inputs["decoder_mask"] = decoder_mask_arr
        cur_len += 1
    times, token_arr = _finalize_decode_times(
        time_source=time_source,
        onset_token_ids=onset_token_ids,
        pointer_times=pointer_times,
        experiment_config=experiment_config,
        patch_mask=decoder_inputs.get("patch_mask"),
    )
    return ArDecodeStats(
        times=times,
        n_forward_steps=n_forward_steps,
        n_onset_tokens=len(onset_token_ids),
        stopped_on_eos=stopped_on_eos,
        onset_token_ids=token_arr,
        eos_prob_trace=np.asarray(eos_probs, dtype=np.float32),
    )


def decode_autoregressive_with_stats_numpy(
    model: tf.keras.Model,
    mert_patches: np.ndarray,
    patch_mask: np.ndarray,
    *,
    max_decoder_len: int,
    patch_frames: int,
    hop_sec: float,
    experiment_config: config.ArExperimentConfig | None = None,
    bos_id: int = targets.BOS_ID,
    eos_id: int = targets.EOS_ID,
    use_kv_cache: bool = True,
    time_source: ArTimeSource = "pointer_residual",
    length_control: ArLengthControl | None = None,
    density_scalar: np.ndarray | float | None = None,
) -> ArDecodeStats:
    """Free-running decode until ``<EOS>`` or ``max_decoder_len``."""
    if experiment_config is not None and use_kv_cache:
        return kv_decode.decode_autoregressive_with_kv_cache_numpy(
            model,
            mert_patches,
            patch_mask,
            experiment_config=experiment_config,
            max_decoder_len=max_decoder_len,
            patch_frames=patch_frames,
            hop_sec=hop_sec,
            bos_id=bos_id,
            eos_id=eos_id,
            time_source=time_source,
            length_control=length_control,
            density_scalar=density_scalar,
        )

    mert_patches = np.asarray(mert_patches, dtype=np.float32)
    patch_mask = np.asarray(patch_mask, dtype=np.float32)
    if mert_patches.ndim == 2:
        mert_patches = mert_patches[np.newaxis, ...]
        patch_mask = patch_mask[np.newaxis, ...]

    decoder_input = np.zeros((1, max_decoder_len), dtype=np.int32)
    decoder_mask_arr = np.zeros((1, max_decoder_len), dtype=np.float32)
    decoder_input[0, 0] = bos_id
    decoder_mask_arr[0, 0] = 1.0

    if experiment_config is not None:
        memory_np, patch_mask = get_encoder_memory_numpy(
            model,
            mert_patches,
            patch_mask,
            experiment_config,
        )
        _, decoder = _infer_encoder_decoder(model, experiment_config)
        decoder_inputs = {
            "encoder_memory": memory_np,
            "patch_mask": patch_mask,
            "decoder_input_ids": decoder_input,
            "decoder_mask": decoder_mask_arr,
        }
    else:
        decoder = model
        decoder_inputs = {
            "mert_patches": mert_patches,
            "patch_mask": patch_mask,
            "decoder_input_ids": decoder_input,
            "decoder_mask": decoder_mask_arr,
        }
    if experiment_config is not None:
        decoder_inputs = _with_density_decoder_inputs(
            decoder_inputs,
            experiment_config=experiment_config,
            density_scalar=density_scalar,
        )

    patch_duration = float(patch_frames) * float(hop_sec)
    return _decode_autoregressive_prefix_numpy(
        decoder,
        decoder_inputs,
        max_decoder_len=max_decoder_len,
        patch_duration=patch_duration,
        eos_id=eos_id,
        time_source=time_source,
        experiment_config=experiment_config,
        length_control=length_control,
    )


def decode_autoregressive_times_numpy(
    model: tf.keras.Model,
    mert_patches: np.ndarray,
    patch_mask: np.ndarray,
    *,
    max_decoder_len: int,
    patch_frames: int,
    hop_sec: float,
    experiment_config: config.ArExperimentConfig | None = None,
    bos_id: int = targets.BOS_ID,
    eos_id: int = targets.EOS_ID,
    time_source: ArTimeSource = "pointer_residual",
    length_control: ArLengthControl | None = None,
) -> np.ndarray:
    """Free-running token decode until ``<EOS>``; return onset times in seconds."""
    return decode_autoregressive_with_stats_numpy(
        model,
        mert_patches,
        patch_mask,
        max_decoder_len=max_decoder_len,
        patch_frames=patch_frames,
        hop_sec=hop_sec,
        experiment_config=experiment_config,
        bos_id=bos_id,
        eos_id=eos_id,
        time_source=time_source,
        length_control=length_control,
    ).times


def decode_teacher_fed_times_numpy(
    pointer_logits: np.ndarray,
    residual_sec: np.ndarray,
    onset_step_mask: np.ndarray,
    *,
    patch_frames: int,
    hop_sec: float,
) -> np.ndarray:
    """Extract sorted onset times from teacher-fed decoder outputs."""
    pointer_logits = np.asarray(pointer_logits, dtype=np.float32)
    residual_sec = np.asarray(residual_sec, dtype=np.float32)
    onset_step_mask = np.asarray(onset_step_mask, dtype=np.float32)
    if pointer_logits.ndim == 3:
        pointer_logits = pointer_logits[0]
        residual_sec = residual_sec[0]
    if onset_step_mask.ndim > 1:
        onset_step_mask = onset_step_mask[0]

    patch_duration = float(patch_frames) * float(hop_sec)
    times: list[float] = []
    for step_idx, active in enumerate(onset_step_mask):
        if active <= 0.5:
            continue
        patch_idx = int(np.argmax(pointer_logits[step_idx]))
        times.append(float(patch_idx) * patch_duration + float(residual_sec[step_idx]))
    return np.asarray(times, dtype=np.float32)


def decode_teacher_fed_times_tf(
    outputs: dict[str, tf.Tensor],
    batch: dict[str, tf.Tensor],
    *,
    patch_frames: int,
    hop_sec: float,
    use_soft_expected: bool = False,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Tensor wrapper returning padded predicted times and a validity mask."""
    pred_times = losses.predicted_times_from_outputs(
        outputs["pointer_logits"],
        outputs["residual_sec"],
        patch_frames=patch_frames,
        hop_sec=hop_sec,
        use_soft_expected=use_soft_expected,
    )
    pred_mask = tf.cast(batch["onset_step_mask"], tf.float32)
    return pred_times, pred_mask
