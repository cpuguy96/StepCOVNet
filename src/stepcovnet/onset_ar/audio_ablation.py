"""Teacher-forced audio-grounding ablation for AR onset checkpoints.

Corrupts only ``mert_patches`` while keeping decoder prefix and targets fixed.
Scores pointer timing and token accuracy per variant. A checkpoint fails the
standing gate when shuffle or zeros reproduces matched performance — the model
is not reading the encoder.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import tensorflow as tf

from stepcovnet import timing_match
from stepcovnet.onset_ar import config, inference, pointer_mask, trainers

VARIANTS = ("matched", "cross_song", "reverse", "shuffle", "zeros")
CORRUPTED_VARIANTS = ("shuffle", "zeros")
GATE_TIMING_MATCH_EPS = 0.02
GATE_SAME_PRED_MAX = 0.85
GATE_TOKEN_ACC_EPS = 0.02
# Keys-only grounding leaves query cosine ≈ 1.0; require a real drop.
GATE_QUERY_COSINE_MAX = 0.95
_QUERY_EXTRACTOR_ATTR = "_ar_onset_pointer_query_extractor"


def corrupt_patches(
    patches: np.ndarray,
    n_valid: int,
    variant: str,
    donor: np.ndarray | None,
    donor_valid: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Replace the valid patch region; padding rows stay zero."""
    out = patches.copy()
    region = out[0, :n_valid]
    if variant == "matched":
        return out
    if variant == "zeros":
        out[0, :n_valid] = 0.0
        return out
    if variant == "reverse":
        out[0, :n_valid] = region[::-1]
        return out
    if variant == "shuffle":
        out[0, :n_valid] = region[rng.permutation(n_valid)]
        return out
    if variant == "cross_song":
        if donor is None or donor_valid <= 0:
            return out
        src = donor[0, :donor_valid]
        if donor_valid < n_valid:
            reps = int(np.ceil(n_valid / donor_valid))
            src = np.tile(src, (reps, 1))
        out[0, :n_valid] = src[:n_valid]
        return out
    msg = f"unknown variant: {variant}"
    raise ValueError(msg)


def model_inputs(
    batch: dict[str, np.ndarray],
    experiment_config: config.ArExperimentConfig,
) -> dict[str, tf.Tensor]:
    keys = ["mert_patches", "patch_mask", "decoder_input_ids", "decoder_mask"]
    if config.density_conditioning_active(experiment_config.model):
        keys.append("density_scalar")
    return {key: tf.constant(batch[key]) for key in keys}


def pointer_nll(pointer_logits: np.ndarray, targets_idx: np.ndarray) -> float:
    """Mean cross-entropy in nats over onset steps."""
    logits = pointer_logits.astype(np.float64)
    shifted = logits - logits.max(axis=-1, keepdims=True)
    log_norm = shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))
    return float(-log_norm[np.arange(targets_idx.size), targets_idx].sum())


def token_accuracy_numpy(
    token_logits: np.ndarray,
    decoder_target_ids: np.ndarray,
    decoder_mask: np.ndarray,
) -> tuple[int, int]:
    """Return (correct, total) over non-padded decoder positions."""
    predictions = np.argmax(token_logits, axis=-1)
    valid = decoder_mask > 0.5
    correct = int(np.sum((predictions == decoder_target_ids) & valid))
    total = int(np.sum(valid))
    return correct, total


def f1_from_counts(tp: int, fp: int, fn: int) -> float:
    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    return float(2.0 * precision * recall / (precision + recall + 1e-9))


def _pointer_query_numpy(
    model: tf.keras.Model,
    batch: dict[str, np.ndarray],
    experiment_config: config.ArExperimentConfig,
) -> np.ndarray | None:
    """Return ``pointer_query`` activations when the layer exists."""
    try:
        model.get_layer("pointer_query")
    except ValueError:
        return None
    extractor = getattr(model, _QUERY_EXTRACTOR_ATTR, None)
    if extractor is None:
        extractor = tf.keras.Model(
            inputs=model.input,
            outputs=model.get_layer("pointer_query").output,
            name="pointer_query_extractor",
        )
        setattr(model, _QUERY_EXTRACTOR_ATTR, extractor)
    query = extractor(model_inputs(batch, experiment_config), training=False)
    return np.asarray(query.numpy()[0], dtype=np.float32)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a.reshape(-1).astype(np.float64)
    b = b.reshape(-1).astype(np.float64)
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return float(np.dot(a, b) / denom)


def score_batch(
    model: tf.keras.Model,
    batch: dict[str, np.ndarray],
    *,
    experiment_config: config.ArExperimentConfig,
) -> dict[str, object]:
    """Teacher-forced alignment + token metrics for one corrupted batch."""
    outputs = model(model_inputs(batch, experiment_config), training=False)
    token_logits = outputs["token_logits"].numpy()[0]
    residual_sec = outputs["residual_sec"].numpy()[0]
    query = _pointer_query_numpy(model, batch, experiment_config)

    onset_step_mask = batch["onset_step_mask"][0]
    step_indices = np.flatnonzero(onset_step_mask > 0.5)
    target_patches = batch["target_patch_indices"][0][step_indices]
    target_times = batch["target_times"][0][step_indices]
    gt_times = batch["gt_times"][0][batch["gt_mask"][0] > 0.5]
    decoder_target_ids = batch["decoder_target_ids"][0]
    decoder_mask = batch["decoder_mask"][0]
    gap_alignment = config.gap_alignment_active(experiment_config.model)
    n_valid_patches = int(batch["patch_mask"][0].sum())
    max_patch = max(n_valid_patches - 1, 0)

    if gap_alignment:
        gap_logits = outputs["gap_logits"].numpy()[0]
        gap_vocab = experiment_config.build_gap_vocab()
        pred_times = inference.decode_teacher_fed_gap_times_numpy(
            gap_logits,
            residual_sec,
            onset_step_mask,
            batch["target_patch_indices"][0],
            gap_vocab=gap_vocab,
            patch_frames=experiment_config.model.patch_frames,
            hop_sec=experiment_config.dataset.hop_sec,
            max_patch=max_patch,
        )
        prev = pointer_mask.teacher_forced_prev_patch_indices_numpy(
            batch["target_patch_indices"][0],
        )
        pred_patches_list: list[int] = []
        for i in step_indices:
            gap_id = int(np.argmax(gap_logits[i]))
            delta = gap_vocab.decode_delta(gap_id)
            pred_patches_list.append(
                min(max(int(prev[i]) + delta, 0), max_patch),
            )
        pred_patches = np.asarray(pred_patches_list, dtype=np.int32)
        target_gap_ids = batch["target_gap_ids"][0][step_indices]
        logits_for_nll = gap_logits
        nll_targets = target_gap_ids
        uniform_classes = gap_vocab.vocab_size
    else:
        pointer_logits = outputs["pointer_logits"].numpy()[0]
        monotonic = bool(experiment_config.model.monotonic_pointer)
        max_ahead = config.pointer_decode_max_ahead(experiment_config.run)
        soft_alpha = config.pointer_soft_distance_alpha(experiment_config.run)
        pred_times = inference.decode_teacher_fed_times_numpy(
            pointer_logits,
            residual_sec,
            onset_step_mask,
            patch_frames=experiment_config.model.patch_frames,
            hop_sec=experiment_config.dataset.hop_sec,
            target_patch_indices=batch["target_patch_indices"][0],
            monotonic=monotonic,
            max_ahead=max_ahead,
            soft_distance_alpha=soft_alpha,
        )
        if monotonic:
            prev = pointer_mask.teacher_forced_prev_patch_indices_numpy(
                batch["target_patch_indices"][0],
            )
            # Per-step mono (+ soft prior / optional prev+R) so NLL matches train CE.
            logits_for_nll = pointer_logits.astype(np.float32).copy()
            n_patches = logits_for_nll.shape[-1]
            for i in range(logits_for_nll.shape[0]):
                p = int(prev[i])
                if p > 0:
                    logits_for_nll[i, : min(p, n_patches)] = -1e9
                if soft_alpha > 0.0 and p > 0:
                    ahead = np.arange(n_patches, dtype=np.float32) - float(p)
                    logits_for_nll[i] -= soft_alpha * np.maximum(ahead, 0.0)
                if max_ahead > 0:
                    hi = p + max_ahead
                    if hi + 1 < n_patches:
                        logits_for_nll[i, hi + 1 :] = -1e9
            pred_patches = np.asarray(
                [
                    inference._argmax_pointer_patch(  # noqa: SLF001
                        pointer_logits[i],
                        prev_patch=int(prev[i]),
                        monotonic=True,
                        max_ahead=max_ahead,
                        soft_distance_alpha=soft_alpha,
                    )
                    for i in step_indices
                ],
                dtype=np.int32,
            )
        else:
            logits_for_nll = pointer_logits
            pred_patches = np.argmax(pointer_logits, axis=-1)[step_indices]
        nll_targets = target_patches
        uniform_classes = max(n_valid_patches, 1)

    tolerance_sec = experiment_config.run.tolerance_sec

    tp, fp, fn = trainers._ar_event_onset_counts_numpy(  # noqa: SLF001
        pred_times,
        np.ones(pred_times.shape, dtype=np.float32),
        gt_times,
        np.ones(gt_times.shape, dtype=np.float32),
        tolerance_sec=tolerance_sec,
    )
    report = timing_match.timing_match_report(
        pred_times,
        target_times,
        tolerance_sec=tolerance_sec,
    )
    token_correct, token_total = token_accuracy_numpy(
        token_logits,
        decoder_target_ids,
        decoder_mask,
    )
    token_preds = np.argmax(token_logits, axis=-1)
    valid = decoder_mask > 0.5
    return {
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "n_matched": int(report["n_matched"]),
        "n_denom": int(report["n_denom"]),
        "n_steps": int(step_indices.size),
        "patch_wrong": int(np.sum(pred_patches != target_patches)),
        "nll_sum": pointer_nll(logits_for_nll[step_indices], nll_targets),
        "uniform_nll_sum": float(np.log(max(uniform_classes, 1)) * step_indices.size),
        "n_valid_patches": n_valid_patches,
        "token_correct": token_correct,
        "token_total": token_total,
        "token_wrong": token_total - token_correct,
        "_pred_patches": pred_patches,
        "_token_preds": token_preds[valid],
        "_token_targets": decoder_target_ids[valid],
        "_pointer_query": None if query is None else query[step_indices],
    }


def empty_variant_totals() -> dict[str, dict[str, float]]:
    return {
        variant: {
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "n_matched": 0,
            "n_denom": 0,
            "n_steps": 0,
            "patch_wrong": 0,
            "nll_sum": 0.0,
            "uniform_nll_sum": 0.0,
            "agree": 0,
            "token_correct": 0,
            "token_total": 0,
            "token_agree": 0,
            "query_cosine_sum": 0.0,
            "query_cosine_count": 0.0,
        }
        for variant in VARIANTS
    }


def accumulate_variant_row(
    totals: dict[str, dict[str, float]],
    variant: str,
    row: dict[str, object],
    *,
    matched_pred_patches: np.ndarray | None,
    matched_token_preds: np.ndarray | None,
    matched_token_targets: np.ndarray | None,
    matched_pointer_query: np.ndarray | None = None,
) -> None:
    pred_patches = row.pop("_pred_patches")
    token_preds = row.pop("_token_preds")
    row.pop("_token_targets")
    query = row.pop("_pointer_query", None)
    acc = totals[variant]
    for key in ("tp", "fp", "fn", "n_matched", "n_denom", "n_steps"):
        acc[key] += row[key]
    acc["patch_wrong"] += row["patch_wrong"]
    acc["nll_sum"] += row["nll_sum"]
    acc["uniform_nll_sum"] += row["uniform_nll_sum"]
    acc["token_correct"] += row["token_correct"]
    acc["token_total"] += row["token_total"]
    if matched_pred_patches is not None:
        acc["agree"] += int(np.sum(pred_patches == matched_pred_patches))
    if (
        matched_token_preds is not None
        and matched_token_targets is not None
        and token_preds.shape == matched_token_preds.shape
    ):
        acc["token_agree"] += int(np.sum(token_preds == matched_token_preds))
    if (
        matched_pointer_query is not None
        and isinstance(query, np.ndarray)
        and query.shape == matched_pointer_query.shape
    ):
        acc["query_cosine_sum"] = acc.get("query_cosine_sum", 0.0) + _cosine(
            matched_pointer_query,
            query,
        )
        acc["query_cosine_count"] = acc.get("query_cosine_count", 0.0) + 1.0


def summarize_variants(
    totals: Mapping[str, Mapping[str, float]],
    *,
    skip_cross_song: bool = False,
) -> dict[str, dict[str, float]]:
    rows: dict[str, dict[str, float]] = {}
    for variant in VARIANTS:
        if variant == "cross_song" and skip_cross_song:
            continue
        acc = totals[variant]
        steps = max(acc["n_steps"], 1)
        token_total = max(acc["token_total"], 1)
        query_count = float(acc.get("query_cosine_count", 0.0))
        query_cosine = (
            float(acc.get("query_cosine_sum", 0.0)) / query_count
            if query_count > 0
            else 1.0
        )
        rows[variant] = {
            "f1_hungarian": f1_from_counts(
                int(acc["tp"]),
                int(acc["fp"]),
                int(acc["fn"]),
            ),
            "timing_match": acc["n_matched"] / max(acc["n_denom"], 1),
            "patch_wrong_rate": acc["patch_wrong"] / steps,
            "pointer_nll": acc["nll_sum"] / steps,
            "uniform_nll": acc["uniform_nll_sum"] / steps,
            "same_pred_as_matched": acc["agree"] / steps,
            "token_accuracy": acc["token_correct"] / token_total,
            "token_wrong_rate": (acc["token_total"] - acc["token_correct"])
            / token_total,
            "same_token_as_matched": acc["token_agree"] / token_total,
            "query_cosine_vs_matched": query_cosine,
            "n_steps": int(acc["n_steps"]),
            "n_token_positions": int(acc["token_total"]),
        }
    return rows


@dataclass(frozen=True)
class AudioGroundingGateResult:
    passed: bool
    failures: tuple[str, ...]
    pointer_passed: bool
    token_passed: bool
    query_passed: bool


def audio_grounding_gate(
    variant_rows: Mapping[str, Mapping[str, float]],
    *,
    timing_match_eps: float = GATE_TIMING_MATCH_EPS,
    same_pred_max: float = GATE_SAME_PRED_MAX,
    token_acc_eps: float = GATE_TOKEN_ACC_EPS,
    query_cosine_max: float = GATE_QUERY_COSINE_MAX,
) -> AudioGroundingGateResult:
    """Fail when corruption reproduces matched scores without decoder grounding.

    Pointer timing collapse alone is not enough: keys-only content pointers can
    flip argmax when keys move while ``pointer_query`` stays fixed. The decisive
    decoder check is **zeros** — silence must move tokens and/or
    ``pointer_query``. Shuffle query/token cosine is a weak probe on its own:
    attention pooling can be nearly permutation-invariant while zeros still
    prove the query path reads audio (and pe-free keys still collapse the
    pointer under shuffle).
    """
    matched = variant_rows.get("matched")
    if matched is None:
        return AudioGroundingGateResult(
            passed=False,
            failures=("missing matched variant",),
            pointer_passed=False,
            token_passed=False,
            query_passed=False,
        )

    failures: list[str] = []
    matched_timing = float(matched["timing_match"])
    matched_token = float(matched.get("token_accuracy", 0.0))

    pointer_ok = True
    zeros_token_ok = False
    zeros_query_ok = False
    for variant in CORRUPTED_VARIANTS:
        row = variant_rows.get(variant)
        if row is None:
            failures.append(f"{variant}: missing")
            pointer_ok = False
            continue

        timing = float(row["timing_match"])
        same_pred = float(row["same_pred_as_matched"])
        # When matched timing is already below eps, ``matched - eps`` is ≤ 0 so
        # every non-negative corrupted score "matches" — a floor lie that always
        # fails the pointer clause on R2. Rely on same_pred collapse instead.
        if matched_timing >= timing_match_eps and (
            timing >= matched_timing - timing_match_eps
        ):
            failures.append(
                f"pointer/{variant}: timing_match {timing:.4f} "
                f"≈ matched {matched_timing:.4f}",
            )
            pointer_ok = False
        if same_pred >= same_pred_max:
            failures.append(
                f"pointer/{variant}: same_pred_as_matched {same_pred:.4f} "
                f">= {same_pred_max:.2f}",
            )
            pointer_ok = False

        token_acc = float(row.get("token_accuracy", 0.0))
        token_collapsed = token_acc < matched_token - token_acc_eps
        query_cos = float(row.get("query_cosine_vs_matched", 1.0))
        query_collapsed = query_cos < query_cosine_max

        if variant == "zeros":
            zeros_token_ok = token_collapsed
            zeros_query_ok = query_collapsed
            if not token_collapsed:
                failures.append(
                    f"token/zeros: token_accuracy {token_acc:.4f} "
                    f"≈ matched {matched_token:.4f}",
                )
            if not query_collapsed:
                failures.append(
                    f"query/zeros: query_cosine_vs_matched {query_cos:.4f} "
                    f">= {query_cosine_max:.2f}",
                )

    decoder_ok = zeros_token_ok or zeros_query_ok
    if not decoder_ok:
        failures.append(
            "decoder: zeros left token accuracy and pointer_query unchanged "
            "(keys-only false positive)",
        )

    return AudioGroundingGateResult(
        passed=pointer_ok and decoder_ok,
        failures=tuple(failures),
        pointer_passed=pointer_ok,
        token_passed=zeros_token_ok,
        query_passed=zeros_query_ok,
    )
