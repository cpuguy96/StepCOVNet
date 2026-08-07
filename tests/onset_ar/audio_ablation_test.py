"""Unit tests for AR onset audio-grounding ablation helpers."""

from __future__ import annotations

import numpy as np

from stepcovnet.onset_ar import audio_ablation


def test_corrupt_patches_variants_change_valid_region() -> None:
    rng = np.random.default_rng(0)
    patches = np.arange(24, dtype=np.float32).reshape(1, 8, 3)
    donor = np.full((1, 8, 3), 99.0, dtype=np.float32)

    zeros = audio_ablation.corrupt_patches(patches, 8, "zeros", None, 0, rng)
    assert np.allclose(zeros[0, :8], 0.0)
    assert np.allclose(zeros[0, 8:], patches[0, 8:])

    reverse = audio_ablation.corrupt_patches(patches, 8, "reverse", None, 0, rng)
    assert np.allclose(reverse[0, :8], patches[0, :8][::-1])

    shuffle = audio_ablation.corrupt_patches(patches, 8, "shuffle", None, 0, rng)
    assert np.array_equal(
        np.sort(shuffle[0, :8], axis=0), np.sort(patches[0, :8], axis=0)
    )
    assert not np.array_equal(shuffle[0, :8], patches[0, :8])

    cross = audio_ablation.corrupt_patches(patches, 8, "cross_song", donor, 8, rng)
    assert np.allclose(cross[0, :8], 99.0)


def test_token_accuracy_numpy_counts_masked_positions() -> None:
    token_logits = np.array(
        [
            [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        ],
        dtype=np.float32,
    )
    targets = np.array([[1, 0, 2]], dtype=np.int32)
    mask = np.array([[1.0, 1.0, 0.0]], dtype=np.float32)
    correct, total = audio_ablation.token_accuracy_numpy(token_logits, targets, mask)
    assert correct == 2
    assert total == 2


def test_summarize_variants_includes_token_metrics() -> None:
    totals = audio_ablation.empty_variant_totals()
    totals["matched"]["n_steps"] = 10
    totals["matched"]["n_denom"] = 10
    totals["matched"]["n_matched"] = 8
    totals["matched"]["token_correct"] = 90
    totals["matched"]["token_total"] = 100
    totals["matched"]["token_agree"] = 100

    rows = audio_ablation.summarize_variants(totals)
    assert rows["matched"]["timing_match"] == 0.8
    assert rows["matched"]["token_accuracy"] == 0.9
    assert rows["matched"]["same_token_as_matched"] == 1.0


def test_audio_grounding_gate_passes_when_corruption_collapses() -> None:
    rows = {
        "matched": {
            "timing_match": 0.95,
            "same_pred_as_matched": 1.0,
            "token_accuracy": 0.90,
            "query_cosine_vs_matched": 1.0,
        },
        "shuffle": {
            "timing_match": 0.05,
            "same_pred_as_matched": 0.10,
            "token_accuracy": 0.20,
            "query_cosine_vs_matched": 0.40,
        },
        "zeros": {
            "timing_match": 0.02,
            "same_pred_as_matched": 0.05,
            "token_accuracy": 0.15,
            "query_cosine_vs_matched": 0.30,
        },
    }
    gate = audio_ablation.audio_grounding_gate(rows)
    assert gate.passed
    assert gate.pointer_passed
    assert gate.token_passed
    assert gate.query_passed
    assert gate.failures == ()


def test_audio_grounding_gate_fails_keys_only_query_blind() -> None:
    """Pointer collapses but query cosine stays ~1 — classic false positive."""
    rows = {
        "matched": {
            "timing_match": 0.95,
            "same_pred_as_matched": 1.0,
            "token_accuracy": 0.90,
            "query_cosine_vs_matched": 1.0,
        },
        "shuffle": {
            "timing_match": 0.05,
            "same_pred_as_matched": 0.10,
            "token_accuracy": 0.89,
            "query_cosine_vs_matched": 0.999,
        },
        "zeros": {
            "timing_match": 0.02,
            "same_pred_as_matched": 0.05,
            "token_accuracy": 0.88,
            "query_cosine_vs_matched": 0.998,
        },
    }
    gate = audio_ablation.audio_grounding_gate(rows)
    assert not gate.passed
    assert gate.pointer_passed
    assert not gate.token_passed
    assert not gate.query_passed
    assert any("keys-only" in failure for failure in gate.failures)


def test_audio_grounding_gate_passes_on_query_even_if_tokens_blind() -> None:
    rows = {
        "matched": {
            "timing_match": 0.95,
            "same_pred_as_matched": 1.0,
            "token_accuracy": 0.90,
            "query_cosine_vs_matched": 1.0,
        },
        "shuffle": {
            "timing_match": 0.05,
            "same_pred_as_matched": 0.10,
            "token_accuracy": 0.89,
            "query_cosine_vs_matched": 0.40,
        },
        "zeros": {
            "timing_match": 0.02,
            "same_pred_as_matched": 0.05,
            "token_accuracy": 0.88,
            "query_cosine_vs_matched": 0.35,
        },
    }
    gate = audio_ablation.audio_grounding_gate(rows)
    assert gate.passed
    assert gate.pointer_passed
    assert not gate.token_passed
    assert gate.query_passed


def test_audio_grounding_gate_passes_when_zeros_grounds_shuffle_invariant_query() -> (
    None
):
    """Shuffle-invariant attention pooling is OK if zeros moves the query."""
    rows = {
        "matched": {
            "timing_match": 0.95,
            "same_pred_as_matched": 1.0,
            "token_accuracy": 1.0,
            "query_cosine_vs_matched": 1.0,
        },
        "shuffle": {
            "timing_match": 0.0,
            "same_pred_as_matched": 0.0,
            "token_accuracy": 1.0,
            "query_cosine_vs_matched": 1.0,
        },
        "zeros": {
            "timing_match": 0.0,
            "same_pred_as_matched": 0.0,
            "token_accuracy": 0.12,
            "query_cosine_vs_matched": 0.42,
        },
    }
    gate = audio_ablation.audio_grounding_gate(rows)
    assert gate.passed
    assert gate.pointer_passed
    assert gate.token_passed
    assert gate.query_passed


def test_audio_grounding_gate_skips_timing_clause_at_floor() -> None:
    """Matched timing below eps must not auto-fail every non-negative score."""
    rows = {
        "matched": {
            "timing_match": 0.009,
            "same_pred_as_matched": 1.0,
            "token_accuracy": 0.20,
            "query_cosine_vs_matched": 1.0,
        },
        "shuffle": {
            "timing_match": 0.008,
            "same_pred_as_matched": 0.10,
            "token_accuracy": 0.18,
            "query_cosine_vs_matched": 0.99,
        },
        "zeros": {
            "timing_match": 0.007,
            "same_pred_as_matched": 0.05,
            "token_accuracy": 0.10,
            "query_cosine_vs_matched": 0.40,
        },
    }
    gate = audio_ablation.audio_grounding_gate(rows)
    assert gate.pointer_passed
    assert gate.query_passed
    assert gate.passed
    assert not any("timing_match" in failure for failure in gate.failures)


def test_audio_grounding_gate_fails_when_shuffle_reproduces_matched() -> None:
    rows = {
        "matched": {
            "timing_match": 0.95,
            "same_pred_as_matched": 1.0,
            "token_accuracy": 0.90,
            "query_cosine_vs_matched": 1.0,
        },
        "shuffle": {
            "timing_match": 0.94,
            "same_pred_as_matched": 0.99,
            "token_accuracy": 0.89,
            "query_cosine_vs_matched": 0.99,
        },
        "zeros": {
            "timing_match": 0.02,
            "same_pred_as_matched": 0.05,
            "token_accuracy": 0.15,
            "query_cosine_vs_matched": 0.30,
        },
    }
    gate = audio_ablation.audio_grounding_gate(rows)
    assert not gate.passed
    assert not gate.pointer_passed
    assert any("pointer/shuffle" in failure for failure in gate.failures)
