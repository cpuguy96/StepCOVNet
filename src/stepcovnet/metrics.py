"""PyTorch and NumPy metrics for onset detection and StepMania arrow quality."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.nn import functional as nn_func

from stepcovnet import constants
from stepcovnet.pytorch.core import losses as core_losses

_N_NOTE_KINDS = 6
_EPS = 1e-7


def _to_tensor(
    value: torch.Tensor | np.ndarray | Any, dtype: torch.dtype
) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(dtype=dtype)
    return torch.as_tensor(np.asarray(value), dtype=dtype)


def _build_arrow_note_kind_table() -> torch.Tensor:
    """Build lookup table mapping arrow code (0..255) to note kind (0..5)."""
    table: list[int] = []
    for n in range(constants.N_ARROW_TYPES):
        d0 = (n // 64) % 4
        d1 = (n // 16) % 4
        d2 = (n // 4) % 4
        d3 = n % 4
        n1 = int(d0 == 1) + int(d1 == 1) + int(d2 == 1) + int(d3 == 1)
        n2 = int(d0 == 2) + int(d1 == 2) + int(d2 == 2) + int(d3 == 2)
        n3 = int(d0 == 3) + int(d1 == 3) + int(d2 == 3) + int(d3 == 3)
        if n == 0:
            kind = 0
        elif n2 >= 1 and n3 >= 1:
            kind = 5
        elif n2 >= 1:
            kind = 3
        elif n3 >= 1:
            kind = 4
        elif n1 == 1:
            kind = 1
        elif n1 >= 2:
            kind = 2
        else:
            kind = 0
        table.append(kind)
    return torch.tensor(table, dtype=torch.long)


_ARROW_NOTE_KIND_TABLE = _build_arrow_note_kind_table()


class OnsetF1Metric:
    """F1-score for onset detection with a frame tolerance window.

    Attributes:
        tolerance: Frame radius around each true onset counted as a match.
        threshold: Probability threshold for predicted onsets.
        target_threshold: Minimum target value counted as a true onset peak.
        name: Metric display name.
        window_size: Convolution window width ``2 * tolerance + 1``.
        true_positives: Accumulated true positives.
        false_positives: Accumulated false positives.
        false_negatives: Accumulated false negatives.
    """

    def __init__(
        self,
        tolerance: int = 1,
        threshold: float = 0.5,
        name: str = "onset_f1_score",
        target_threshold: float = 1.0,
    ) -> None:
        self.tolerance = int(tolerance)
        self.threshold = float(threshold)
        self.target_threshold = float(target_threshold)
        self.name = name
        self.window_size = 2 * self.tolerance + 1
        self.true_positives = 0.0
        self.false_positives = 0.0
        self.false_negatives = 0.0

    @staticmethod
    def _to_b1t(arr: torch.Tensor) -> torch.Tensor:
        if arr.ndim == 2:
            return arr.unsqueeze(1)
        if arr.ndim == 3:
            return arr.squeeze(-1).unsqueeze(1)
        if arr.ndim == 4:
            return arr.reshape(arr.shape[0], arr.shape[1]).unsqueeze(1)
        raise ValueError(f"Expected 2D/3D/4D tensor, got shape {tuple(arr.shape)}")

    def update_state(
        self,
        y_true: torch.Tensor | np.ndarray,
        y_pred: torch.Tensor | np.ndarray,
        sample_weight: Any = None,
    ) -> None:
        """Accumulate TP/FP/FN for a batch of dense onset predictions.

        Args:
            y_true: Ground truth frame labels ``(B, T)`` or ``(B, T, 1)``.
            y_pred: Predicted onset probabilities matching ``y_true``.
            sample_weight: Unused optional sample weight.
        """
        del sample_weight
        yt = self._to_b1t(_to_tensor(y_true, torch.float32))
        yp = self._to_b1t(_to_tensor(y_pred, torch.float32)).to(device=yt.device)

        yt_bin = (yt >= self.target_threshold - _EPS).float()
        yp_bin = (yp >= self.threshold).float()

        kernel = torch.ones(
            (1, 1, self.window_size), dtype=torch.float32, device=yt.device
        )
        true_win = nn_func.conv1d(yt_bin, kernel, padding=self.tolerance) > 0
        pred_win = nn_func.conv1d(yp_bin, kernel, padding=self.tolerance) > 0

        yt_bool = yt_bin > 0.5
        yp_bool = yp_bin > 0.5

        tp = float((yp_bool & true_win).sum().detach().cpu().item())
        fp = float((yp_bool & (~true_win)).sum().detach().cpu().item())
        fn = float((yt_bool & (~pred_win)).sum().detach().cpu().item())
        self.true_positives += tp
        self.false_positives += fp
        self.false_negatives += fn

    def result(self) -> float:
        """Compute and return the accumulated F1 score.

        Returns:
            Scalar F1 score in ``[0, 1]``.
        """
        p_denom = self.true_positives + self.false_positives
        r_denom = self.true_positives + self.false_negatives
        precision = self.true_positives / p_denom if p_denom > 0.0 else 0.0
        recall = self.true_positives / r_denom if r_denom > 0.0 else 0.0
        f1_denom = precision + recall
        return float(2.0 * precision * recall / f1_denom) if f1_denom > 0.0 else 0.0

    def reset_state(self) -> None:
        """Reset accumulated TP/FP/FN counters to zero."""
        self.true_positives = 0.0
        self.false_positives = 0.0
        self.false_negatives = 0.0

    def get_config(self) -> dict[str, Any]:
        """Return serializable metric configuration.

        Returns:
            Dictionary of metric hyperparameters.
        """
        return {
            "name": self.name,
            "tolerance": self.tolerance,
            "threshold": self.threshold,
            "target_threshold": self.target_threshold,
        }


def _jsd_match_from_counts(
    pred_counts: np.ndarray, true_counts: np.ndarray, eps: float = _EPS
) -> float:
    p = pred_counts / (float(np.sum(pred_counts)) + eps)
    q = true_counts / (float(np.sum(true_counts)) + eps)
    m = 0.5 * (p + q)
    kl_p = float(np.sum(p * (np.log(p + eps) - np.log(m + eps))))
    kl_q = float(np.sum(q * (np.log(q + eps) - np.log(m + eps))))
    return float(1.0 - 0.5 * (kl_p + kl_q))


class ArrowDistributionMatchMetric:
    """Measures ``1 - JSD`` between predicted and ground-truth arrow distributions.

    Attributes:
        num_classes: Number of arrow classes.
        ignore_class: Padding class excluded from counts.
        name: Metric name.
        pred_counts: Accumulated predicted class histogram.
        true_counts: Accumulated ground-truth class histogram.
    """

    def __init__(
        self,
        num_classes: int = constants.N_ARROW_TYPES,
        ignore_class: int = constants.ARROW_PADDING_CLASS,
        name: str = "arrow_dist_match",
    ) -> None:
        self.num_classes = int(num_classes)
        self.ignore_class = int(ignore_class)
        self.name = name
        self.eps = _EPS
        self.pred_counts = np.zeros(self.num_classes, dtype=np.float64)
        self.true_counts = np.zeros(self.num_classes, dtype=np.float64)

    def update_state(
        self,
        y_true: torch.Tensor | np.ndarray,
        y_pred: torch.Tensor | np.ndarray,
        sample_weight: Any = None,
    ) -> None:
        """Accumulate class histograms over valid (non-padding) positions.

        Args:
            y_true: Ground truth arrow indices ``(B, S)``.
            y_pred: Predicted probabilities/logits ``(B, S, num_classes)``.
            sample_weight: Unused optional sample weight.
        """
        del sample_weight
        yt = _to_tensor(y_true, torch.long).detach().cpu().numpy()
        yp = _to_tensor(y_pred, torch.float32).detach().cpu().numpy()
        mask = yt != self.ignore_class
        if not np.any(mask):
            return
        true_vals = np.clip(yt[mask], 0, self.num_classes - 1)
        pred_vals = np.clip(np.argmax(yp, axis=-1)[mask], 0, self.num_classes - 1)
        self.true_counts += np.bincount(true_vals, minlength=self.num_classes)
        self.pred_counts += np.bincount(pred_vals, minlength=self.num_classes)

    def result(self) -> float:
        """Return ``1 - JSD`` over accumulated arrow histograms."""
        return _jsd_match_from_counts(self.pred_counts, self.true_counts, self.eps)

    def reset_state(self) -> None:
        """Reset accumulated histograms."""
        self.pred_counts.fill(0.0)
        self.true_counts.fill(0.0)

    def get_config(self) -> dict[str, Any]:
        """Return serializable metric configuration."""
        return {
            "name": self.name,
            "num_classes": self.num_classes,
            "ignore_class": self.ignore_class,
        }


class ArrowNoteKindDistributionMetric:
    """Measures ``1 - JSD`` over the 6 StepMania note-kind categories.

    Attributes:
        num_note_kinds: Number of note-kind categories (default 6).
        ignore_class: Padding class excluded from counts.
        name: Metric name.
        pred_counts: Accumulated predicted note-kind counts.
        true_counts: Accumulated true note-kind counts.
    """

    def __init__(
        self,
        num_note_kinds: int = _N_NOTE_KINDS,
        ignore_class: int = constants.ARROW_PADDING_CLASS,
        name: str = "arrow_note_kind_dist_match",
    ) -> None:
        self.num_note_kinds = int(num_note_kinds)
        self.ignore_class = int(ignore_class)
        self.name = name
        self.eps = _EPS
        self.pred_counts = np.zeros(self.num_note_kinds, dtype=np.float64)
        self.true_counts = np.zeros(self.num_note_kinds, dtype=np.float64)

    def update_state(
        self,
        y_true: torch.Tensor | np.ndarray,
        y_pred: torch.Tensor | np.ndarray,
        sample_weight: Any = None,
    ) -> None:
        """Accumulate note-kind histograms over valid positions.

        Args:
            y_true: Ground truth arrow indices ``(B, S)``.
            y_pred: Predicted probabilities/logits ``(B, S, N_ARROW_TYPES)``.
            sample_weight: Unused optional sample weight.
        """
        del sample_weight
        table = _ARROW_NOTE_KIND_TABLE.numpy()
        yt = _to_tensor(y_true, torch.long).detach().cpu().numpy()
        yp = _to_tensor(y_pred, torch.float32).detach().cpu().numpy()
        mask = yt != self.ignore_class
        if not np.any(mask):
            return
        true_codes = np.clip(yt[mask], 0, constants.N_ARROW_TYPES - 1)
        pred_codes = np.clip(
            np.argmax(yp, axis=-1)[mask], 0, constants.N_ARROW_TYPES - 1
        )
        self.true_counts += np.bincount(
            table[true_codes], minlength=self.num_note_kinds
        )
        self.pred_counts += np.bincount(
            table[pred_codes], minlength=self.num_note_kinds
        )

    def result(self) -> float:
        """Return ``1 - JSD`` over accumulated note-kind counts."""
        return _jsd_match_from_counts(self.pred_counts, self.true_counts, self.eps)

    def reset_state(self) -> None:
        """Reset accumulated note-kind histograms."""
        self.pred_counts.fill(0.0)
        self.true_counts.fill(0.0)

    def get_config(self) -> dict[str, Any]:
        """Return serializable metric configuration."""
        return {
            "name": self.name,
            "num_note_kinds": self.num_note_kinds,
            "ignore_class": self.ignore_class,
        }


def compute_chart_validity_violations(
    arrow_codes: np.ndarray,
    *,
    max_examples: int = 5,
) -> tuple[int, int, list[tuple[int, int, str]]]:
    """Check chart validity on a 1D sequence of arrow codes (0..255).

    Args:
        arrow_codes: 1D array of integer arrow codes, shape ``(seq_len,)``.
        max_examples: Maximum number of example violations to return.

    Returns:
        ``(num_violations, num_hold_ends, examples)``.
    """
    codes = np.asarray(arrow_codes, dtype=np.int32).ravel()
    seq_len = len(codes)
    digits = np.zeros((seq_len, 4), dtype=np.int32)
    for i in range(seq_len):
        n = int(np.clip(codes[i], 0, 255))
        digits[i, 0] = (n // 64) % 4
        digits[i, 1] = (n // 16) % 4
        digits[i, 2] = (n // 4) % 4
        digits[i, 3] = n % 4
    violations = 0
    hold_ends = 0
    examples: list[tuple[int, int, str]] = []
    for c in range(4):
        col = digits[:, c]
        run = 0
        for i in range(seq_len):
            run_prev = run
            d = col[i]
            if d == 2:
                if run_prev >= 1:
                    violations += 1
                    if len(examples) < max_examples:
                        examples.append((i, c, "nested_hold"))
                run += 1
            elif d == 3:
                hold_ends += 1
                if run_prev < 1:
                    violations += 1
                    if len(examples) < max_examples:
                        examples.append((i, c, "unmatched_3"))
                else:
                    run -= 1
            elif d == 1 and run_prev >= 1:
                violations += 1
                if len(examples) < max_examples:
                    examples.append((i, c, "tap_during_hold"))
        if run > 0:
            violations += 1
            if len(examples) < max_examples:
                examples.append((seq_len - 1, c, "unterminated_hold"))
    return int(violations), int(hold_ends), examples


def chart_validity_auxiliary_loss(
    y_true: torch.Tensor | np.ndarray,
    y_pred: torch.Tensor | np.ndarray,
    ignore_class: int = constants.ARROW_PADDING_CLASS,
) -> torch.Tensor:
    """Differentiable auxiliary loss encouraging chart-valid predictions.

    Args:
        y_true: ``(batch, seq)`` int arrow codes.
        y_pred: ``(batch, seq, N_ARROW_TYPES)`` float probabilities.
        ignore_class: Label value treated as padding.

    Returns:
        Scalar tensor: mean penalty over valid step-columns.
    """
    yt = _to_tensor(y_true, torch.long)
    yp = _to_tensor(y_pred, torch.float32)
    return core_losses.chart_validity_auxiliary_loss(
        yt.to(device=yp.device), yp, ignore_class=ignore_class
    )


def note_kind_balance_auxiliary_loss(
    y_true: torch.Tensor | np.ndarray,
    y_pred: torch.Tensor | np.ndarray,
    ignore_class: int = constants.ARROW_PADDING_CLASS,
) -> torch.Tensor:
    """Differentiable auxiliary loss matching predicted hold rate to labels.

    Args:
        y_true: ``(batch, seq)`` int arrow codes.
        y_pred: ``(batch, seq, N_ARROW_TYPES)`` float probabilities.
        ignore_class: Label value treated as padding.

    Returns:
        Scalar tensor: MSE of predicted vs true hold rate over valid steps.
    """
    yt = _to_tensor(y_true, torch.long)
    yp = _to_tensor(y_pred, torch.float32)
    return core_losses.note_kind_balance_auxiliary_loss(
        yt.to(device=yp.device), yp, ignore_class=ignore_class
    )


def _hard_chart_validity_counts(
    y_true: torch.Tensor | np.ndarray,
    y_pred: torch.Tensor | np.ndarray,
    ignore_class: int = constants.ARROW_PADDING_CLASS,
) -> tuple[float, float]:
    yt = _to_tensor(y_true, torch.long)
    yp = _to_tensor(y_pred, torch.float32)
    device = yp.device
    yt = yt.to(device=device)

    pred_classes = torch.argmax(yp, dim=-1).clamp(0, constants.N_ARROW_TYPES - 1)
    mask_bool = yt != ignore_class
    mask = mask_bool.float()

    table = core_losses._ARROW_DIGIT_ONEHOT_TABLE.to(device=device, dtype=torch.float32)
    pred_onehot = nn_func.one_hot(
        pred_classes, num_classes=constants.N_ARROW_TYPES
    ).float()
    prob_digit = torch.einsum("bsn,ncd->bscd", pred_onehot, table)
    prob_digit = prob_digit * mask.unsqueeze(-1).unsqueeze(-1)

    b, seq_len = int(yt.shape[0]), int(yt.shape[1])
    seq_idx = torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0)
    last_valid_idx = torch.argmax(mask_bool.to(torch.long) * seq_idx, dim=1)
    last_valid_weights = torch.zeros(
        (b, seq_len, 4), device=device, dtype=torch.float32
    )
    valid_batch = mask.sum(dim=1) > 0.0
    batch_ids = torch.arange(b, device=device, dtype=torch.long)
    last_valid_weights[batch_ids, last_valid_idx, :] = valid_batch.float().unsqueeze(-1)

    weights = core_losses._chart_validity_violation_weights(
        prob_digit, last_valid_weights, soft=False
    )
    total_violations = float((weights * mask.unsqueeze(-1)).sum().detach().cpu().item())
    total_slots = float((mask.sum() * 4.0).detach().cpu().item())
    return total_violations, total_slots


def chart_validity_per_batch(
    y_true: torch.Tensor | np.ndarray,
    y_pred: torch.Tensor | np.ndarray,
    ignore_class: int = constants.ARROW_PADDING_CLASS,
) -> torch.Tensor:
    """Compute hard chart validity (argmax) for a single batch in ``[0, 1]``.

    Args:
        y_true: ``(batch, seq)`` int arrow codes.
        y_pred: ``(batch, seq, N_ARROW_TYPES)`` float probabilities.
        ignore_class: Label value treated as padding.

    Returns:
        Scalar float32 tensor in ``[0, 1]``.
    """
    violations, slots = _hard_chart_validity_counts(
        y_true, y_pred, ignore_class=ignore_class
    )
    val = float(np.clip(1.0 - violations / max(1.0, slots), 0.0, 1.0))
    return torch.tensor(val, dtype=torch.float32)


class ChartValidityAuxiliaryLossMetric:
    """Tracks mean ``chart_validity_auxiliary_loss`` across batches.

    Attributes:
        ignore_class: Padding class excluded from loss.
        name: Metric name.
    """

    def __init__(
        self,
        ignore_class: int = constants.ARROW_PADDING_CLASS,
        name: str = "chart_validity_aux_loss",
    ) -> None:
        self.ignore_class = int(ignore_class)
        self.name = name
        self._sum = 0.0
        self._count = 0

    def update_state(
        self,
        y_true: torch.Tensor | np.ndarray,
        y_pred: torch.Tensor | np.ndarray,
        sample_weight: Any = None,
    ) -> None:
        """Accumulate batch chart validity auxiliary loss."""
        del sample_weight
        val = chart_validity_auxiliary_loss(
            y_true, y_pred, ignore_class=self.ignore_class
        )
        self._sum += float(val.detach().cpu().item())
        self._count += 1

    def result(self) -> float:
        """Return mean loss over accumulated batches."""
        return self._sum / self._count if self._count > 0 else 0.0

    def reset_state(self) -> None:
        """Reset running sum and count."""
        self._sum = 0.0
        self._count = 0

    def get_config(self) -> dict[str, Any]:
        """Return serializable metric configuration."""
        return {"name": self.name, "ignore_class": self.ignore_class}


class NoteKindBalanceAuxiliaryLossMetric:
    """Tracks mean ``note_kind_balance_auxiliary_loss`` across batches.

    Attributes:
        ignore_class: Padding class excluded from loss.
        name: Metric name.
    """

    def __init__(
        self,
        ignore_class: int = constants.ARROW_PADDING_CLASS,
        name: str = "note_kind_balance_aux_loss",
    ) -> None:
        self.ignore_class = int(ignore_class)
        self.name = name
        self._sum = 0.0
        self._count = 0

    def update_state(
        self,
        y_true: torch.Tensor | np.ndarray,
        y_pred: torch.Tensor | np.ndarray,
        sample_weight: Any = None,
    ) -> None:
        """Accumulate batch note-kind balance auxiliary loss."""
        del sample_weight
        val = note_kind_balance_auxiliary_loss(
            y_true, y_pred, ignore_class=self.ignore_class
        )
        self._sum += float(val.detach().cpu().item())
        self._count += 1

    def result(self) -> float:
        """Return mean loss over accumulated batches."""
        return self._sum / self._count if self._count > 0 else 0.0

    def reset_state(self) -> None:
        """Reset running sum and count."""
        self._sum = 0.0
        self._count = 0

    def get_config(self) -> dict[str, Any]:
        """Return serializable metric configuration."""
        return {"name": self.name, "ignore_class": self.ignore_class}


class ChartValidityMetric:
    """Measures full-sequence hard chart validity across batches in ``[0, 1]``.

    Attributes:
        ignore_class: Padding class excluded from validity counts.
        name: Metric name.
        total_violations: Accumulated violation count.
        total_valid_step_columns: Accumulated valid step-column slots.
    """

    def __init__(
        self,
        ignore_class: int = constants.ARROW_PADDING_CLASS,
        name: str = "chart_validity",
    ) -> None:
        self.ignore_class = int(ignore_class)
        self.name = name
        self.total_violations = 0.0
        self.total_valid_step_columns = 0.0

    def update_state(
        self,
        y_true: torch.Tensor | np.ndarray,
        y_pred: torch.Tensor | np.ndarray,
        sample_weight: Any = None,
    ) -> None:
        """Accumulate hard chart violations and valid slot counts."""
        del sample_weight
        violations, slots = _hard_chart_validity_counts(
            y_true, y_pred, ignore_class=self.ignore_class
        )
        self.total_violations += violations
        self.total_valid_step_columns += slots

    def result(self) -> float:
        """Return ``clip(1 - violations / max(1, slots), 0, 1)``."""
        denom = max(1.0, self.total_valid_step_columns)
        return float(np.clip(1.0 - self.total_violations / denom, 0.0, 1.0))

    def reset_state(self) -> None:
        """Reset accumulated counts."""
        self.total_violations = 0.0
        self.total_valid_step_columns = 0.0

    def get_config(self) -> dict[str, Any]:
        """Return serializable metric configuration."""
        return {"name": self.name, "ignore_class": self.ignore_class}


class ChartValidityPassRateMetric:
    """Fraction of batches whose hard chart validity meets ``threshold``.

    Attributes:
        threshold: Minimum validity in ``[0, 1]`` to count a batch as passing.
        ignore_class: Padding class excluded from validity counts.
        name: Metric name.
        passed: Number of passing batches.
        total: Total batches observed.
    """

    def __init__(
        self,
        threshold: float,
        ignore_class: int = constants.ARROW_PADDING_CLASS,
        name: str = "chart_validity_pass_rate",
    ) -> None:
        self.threshold = float(threshold)
        self.ignore_class = int(ignore_class)
        self.name = name
        self.passed = 0.0
        self.total = 0.0

    def update_state(
        self,
        y_true: torch.Tensor | np.ndarray,
        y_pred: torch.Tensor | np.ndarray,
        sample_weight: Any = None,
    ) -> None:
        """Update passing-batch counters from one batch."""
        del sample_weight
        batch_val = float(
            chart_validity_per_batch(y_true, y_pred, ignore_class=self.ignore_class)
        )
        if batch_val >= self.threshold:
            self.passed += 1.0
        self.total += 1.0

    def result(self) -> float:
        """Return fraction of batches meeting ``threshold``."""
        return self.passed / self.total if self.total > 0.0 else 0.0

    def reset_state(self) -> None:
        """Reset batch counters."""
        self.passed = 0.0
        self.total = 0.0

    def get_config(self) -> dict[str, Any]:
        """Return serializable metric configuration."""
        return {
            "name": self.name,
            "threshold": self.threshold,
            "ignore_class": self.ignore_class,
        }


__all__ = [
    "ArrowDistributionMatchMetric",
    "ArrowNoteKindDistributionMetric",
    "ChartValidityAuxiliaryLossMetric",
    "ChartValidityMetric",
    "ChartValidityPassRateMetric",
    "NoteKindBalanceAuxiliaryLossMetric",
    "OnsetF1Metric",
    "chart_validity_auxiliary_loss",
    "chart_validity_per_batch",
    "compute_chart_validity_violations",
    "note_kind_balance_auxiliary_loss",
]
