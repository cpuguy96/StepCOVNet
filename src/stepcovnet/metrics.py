"""Custom metrics and divergence functions for evaluating step detection models."""

import numpy as np
import keras
import tensorflow as tf
from keras import backend as K

from stepcovnet import constants

# Note kinds for arrow quality: empty, single, chord, hold_start, hold_end, hold_both
_N_NOTE_KINDS = 6


@keras.saving.register_keras_serializable()
class OnsetF1Metric(keras.metrics.Metric):
    """
    Custom Keras metric to calculate the F1-score for onset detection
    with a tolerance window.

    Onsets are considered correctly predicted (True Positive) if a predicted
    onset falls within a specified tolerance window around a true onset.

    Handles inputs `y_true` and `y_pred` with shapes:
    - (batch_size, time_steps)
    - (batch_size, time_steps, 1)
    - (batch_size, time_steps, 1, 1) # Attempts to handle potential extra dim

    Args:
        tolerance (int): The number of time steps allowed on either side of a
                         true onset for a prediction to be considered correct.
                         Defaults to 1 (meaning +/- 1 time step).
        threshold (float): The probability threshold to convert model outputs
                           (probabilities) into binary predictions (0 or 1).
                           Defaults to 0.5.
        name (str): Name of the metric instance. Defaults to 'onset_f1_score'.
        **kwargs: Additional keyword arguments.
    """

    def __init__(self, tolerance=1, threshold=0.5, name="onset_f1_score", **kwargs):
        super().__init__(name=name, **kwargs)
        self.tolerance = tolerance
        self.threshold = threshold
        # Calculate the full window size for convolution
        self.window_size = 2 * self.tolerance + 1

        # State variables to accumulate counts across batches
        self.true_positives = self.add_weight(name="tp", initializer="zeros")
        self.false_positives = self.add_weight(name="fp", initializer="zeros")
        self.false_negatives = self.add_weight(name="fn", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        """
        Updates the state variables (TP, FP, FN) for a batch of data.

        Args:
            y_true: Ground truth labels (binary tensor: 1 for onset, 0 otherwise).
                    Shape: (batch_size, time_steps) or (batch_size, time_steps, 1).
            y_pred: Predicted probabilities from the model.
                    Shape: (batch_size, time_steps) or (batch_size, time_steps, 1).
            sample_weight: Optional weights for samples. Not used in this metric.
        """
        # Ensure inputs are float32 for calculations
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        # --- Reshape inputs to ensure they are 3D: (batch, time_steps, 1) using tf.case ---
        # tf.case handles conditional logic robustly in graph mode.

        # Reshape y_true
        rank_true = tf.rank(y_true)
        shape_true = tf.shape(y_true)
        # Define reshaping logic based on rank
        y_true_conv = tf.case(
            [
                (
                    tf.equal(rank_true, 2),
                    lambda: tf.reshape(y_true, [shape_true[0], shape_true[1], 1]),  # type: ignore
                ),
                (
                    tf.equal(rank_true, 4),
                    lambda: tf.reshape(y_true, [shape_true[0], shape_true[1], 1]),  # type: ignore
                ),
                # Squeeze rank 4 -> 3
            ],
            default=lambda: y_true,  # Assumes rank 3 is already correct
            exclusive=True,
            name="reshape_y_true",
        )
        # Set a general shape hint after reshaping
        y_true_conv.set_shape([None, None, 1])

        # Reshape y_pred
        rank_pred = tf.rank(y_pred)
        shape_pred = tf.shape(y_pred)
        # Define reshaping logic based on rank
        y_pred_proc = tf.case(
            [
                (
                    tf.equal(rank_pred, 2),
                    lambda: tf.reshape(y_pred, [shape_pred[0], shape_pred[1], 1]),  # type: ignore
                ),
                (
                    tf.equal(rank_pred, 4),
                    lambda: tf.reshape(y_pred, [shape_pred[0], shape_pred[1], 1]),  # type: ignore
                ),
                # Squeeze rank 4 -> 3
            ],
            default=lambda: y_pred,  # Assumes rank 3 is already correct
            exclusive=True,
            name="reshape_y_pred",
        )
        # Set a general shape hint after reshaping
        y_pred_proc.set_shape([None, None, 1])

        # Apply threshold to get binary predictions (shape will match y_pred_proc)
        y_pred_binary_conv = tf.cast(y_pred_proc >= self.threshold, tf.float32)

        # --- Use convolution to find matches within the tolerance window ---

        # Create a convolution kernel (filter) of all ones
        # Shape: (filter_width, in_channels, out_channels)
        kernel = tf.ones((self.window_size, 1, 1), dtype=tf.float32)

        # Pad inputs to handle edges correctly during convolution
        # Padding amount: 'tolerance' on each side. Input tensors are now guaranteed rank 3.
        padding = [
            [0, 0],
            [self.tolerance, self.tolerance],
            [0, 0],
        ]  # Pad only the time dimension
        y_true_padded = tf.pad(y_true_conv, padding, "CONSTANT")
        y_pred_padded = tf.pad(y_pred_binary_conv, padding, "CONSTANT")

        # Convolve y_true: Marks regions within `tolerance` of a true onset
        # Output shape: (batch_size, time_steps, 1)
        true_onset_windows = tf.nn.conv1d(
            y_true_padded,
            filters=kernel,
            stride=1,
            padding="VALID",  # Use 'VALID' padding with manually padded input
        )
        # Result > 0 means a true onset is within the window at that point
        true_onset_windows_bool = true_onset_windows > 0

        # Convolve y_pred_binary: Marks regions within `tolerance` of a predicted onset
        pred_onset_windows = tf.nn.conv1d(
            y_pred_padded, filters=kernel, stride=1, padding="VALID"
        )
        # Result > 0 means a predicted onset is within the window
        pred_onset_windows_bool = pred_onset_windows > 0

        # --- Calculate TP, FP, FN ---
        # Perform calculations using the 3D tensors (y_pred_binary_conv, y_true_conv)

        # True Positives (TP): Predicted onset falls within the tolerance window of a true onset.
        # Check where y_pred_binary_conv is 1 AND a true onset is nearby (true_onset_windows > 0)
        tp = tf.reduce_sum(
            tf.cast(
                tf.logical_and(
                    tf.cast(y_pred_binary_conv, tf.bool), true_onset_windows_bool
                ),
                tf.float32,
            )
        )

        # False Positives (FP): Predicted onset does NOT fall within the tolerance window of any true onset.
        # Check where y_pred_binary_conv is 1 AND no true onset is nearby (true_onset_windows == 0)
        fp = tf.reduce_sum(
            tf.cast(
                tf.logical_and(
                    tf.cast(y_pred_binary_conv, tf.bool),
                    tf.logical_not(true_onset_windows_bool),
                ),
                tf.float32,
            )
        )

        # False Negatives (FN): True onset does NOT have any predicted onsets within its tolerance window.
        # Check where y_true_conv is 1 AND no predicted onset is nearby (pred_onset_windows == 0)
        fn = tf.reduce_sum(
            tf.cast(
                tf.logical_and(
                    tf.cast(y_true_conv, tf.bool),
                    tf.logical_not(pred_onset_windows_bool),
                ),
                tf.float32,
            )
        )

        # Update state variables
        self.true_positives.assign_add(tp)
        self.false_positives.assign_add(fp)
        self.false_negatives.assign_add(fn)

    def result(self):
        """
        Computes and returns the F1 score.
        """
        # Calculate Precision
        precision = self.true_positives / (
            self.true_positives + self.false_positives + K.epsilon()
        )
        # Calculate Recall
        recall = self.true_positives / (
            self.true_positives + self.false_negatives + K.epsilon()
        )
        # Calculate F1 Score
        f1 = 2 * (precision * recall) / (precision + recall + K.epsilon())
        return f1

    def reset_state(self):
        """
        Resets all state variables to zero.
        """
        self.true_positives.assign(0.0)
        self.false_positives.assign(0.0)
        self.false_negatives.assign(0.0)

    def get_config(self):
        """Returns the serializable config of the metric."""
        config = super().get_config()
        config.update(
            {
                "tolerance": self.tolerance,
                "threshold": self.threshold,
            }
        )
        return config


@keras.saving.register_keras_serializable()
class ArrowDistributionMatchMetric(keras.metrics.Metric):
    """
    Metric that measures how closely the predicted arrow-type distribution
    matches the ground-truth distribution via 1 - Jensen-Shannon divergence.

    Higher values (closer to 1) mean the model's pattern of arrow choices
    matches the chart; lower values mean the distributions differ. Uses the
    same ignore_class=0 convention as the arrow loss (padding positions are
    excluded).

    Expects y_true shape (batch, seq_len) and y_pred shape (batch, seq_len, N_ARROW_TYPES).
    """

    def __init__(
        self,
        num_classes: int = constants.N_ARROW_TYPES,
        ignore_class: int = 0,
        name: str = "arrow_dist_match",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.ignore_class = ignore_class
        self.eps = 1e-7
        self.pred_counts = self.add_weight(
            name="pred_counts",
            shape=(num_classes,),
            initializer="zeros",
        )
        self.true_counts = self.add_weight(
            name="true_counts",
            shape=(num_classes,),
            initializer="zeros",
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.int32)
        y_pred = tf.cast(y_pred, tf.float32)
        # Only count positions that are not padding (same as loss).
        mask = tf.cast(tf.not_equal(y_true, self.ignore_class), tf.float32)
        pred_classes = tf.argmax(y_pred, axis=-1)
        pred_classes = tf.cast(pred_classes, tf.int32)
        # One-hot (batch, seq, num_classes), then mask and sum over batch/seq.
        true_onehot = tf.one_hot(
            y_true, depth=self.num_classes, axis=-1, dtype=tf.float32
        )
        pred_onehot = tf.one_hot(
            pred_classes, depth=self.num_classes, axis=-1, dtype=tf.float32
        )
        mask_exp = tf.expand_dims(mask, axis=-1)
        true_inc = tf.reduce_sum(mask_exp * true_onehot, axis=[0, 1])
        pred_inc = tf.reduce_sum(mask_exp * pred_onehot, axis=[0, 1])
        self.true_counts.assign_add(true_inc)
        self.pred_counts.assign_add(pred_inc)

    def result(self):
        p = self.pred_counts / (tf.reduce_sum(self.pred_counts) + self.eps)
        q = self.true_counts / (tf.reduce_sum(self.true_counts) + self.eps)
        m = 0.5 * (p + q)
        kl_p = tf.reduce_sum(
            p * (tf.math.log(p + self.eps) - tf.math.log(m + self.eps))
        )
        kl_q = tf.reduce_sum(
            q * (tf.math.log(q + self.eps) - tf.math.log(m + self.eps))
        )
        jsd = 0.5 * (kl_p + kl_q)
        return 1.0 - jsd

    def reset_state(self):
        self.pred_counts.assign(tf.zeros_like(self.pred_counts))
        self.true_counts.assign(tf.zeros_like(self.true_counts))

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_classes": self.num_classes,
                "ignore_class": self.ignore_class,
            }
        )
        return config


# Note-kind labels: 0=empty, 1=single, 2=chord, 3=hold_start, 4=hold_end, 5=hold_both
def _build_arrow_note_kind_table() -> tf.Tensor:
    """Build a lookup table mapping arrow code (0..255) to note kind (0..5)."""
    table = []
    for n in range(constants.N_ARROW_TYPES):
        d0 = (n // 64) % 4
        d1 = (n // 16) % 4
        d2 = (n // 4) % 4
        d3 = n % 4
        n1 = int(d0 == 1) + int(d1 == 1) + int(d2 == 1) + int(d3 == 1)
        n2 = int(d0 == 2) + int(d1 == 2) + int(d2 == 2) + int(d3 == 2)
        n3 = int(d0 == 3) + int(d1 == 3) + int(d2 == 3) + int(d3 == 3)
        if n == 0:
            kind = 0  # empty
        elif n2 >= 1 and n3 >= 1:
            kind = 5  # hold_both
        elif n2 >= 1:
            kind = 3  # hold_start
        elif n3 >= 1:
            kind = 4  # hold_end
        elif n1 == 1:
            kind = 1  # single
        elif n1 >= 2:
            kind = 2  # chord
        else:
            kind = 0  # empty (no tap/hold)
        table.append(kind)
    return tf.constant(table, dtype=tf.int32)  # type: ignore[return-value]


_ARROW_NOTE_KIND_TABLE = _build_arrow_note_kind_table()


@keras.saving.register_keras_serializable()
class ArrowNoteKindDistributionMetric(keras.metrics.Metric):
    """
    Metric that measures how well the predicted distribution over note kinds
    (single, chord, hold_start, hold_end, hold_both) matches the ground truth,
    via 1 - JSD over the 6 note-kind categories.

    Uses the same ignore_class=0 convention as the arrow loss.
    Expects y_true shape (batch, seq_len) and y_pred shape (batch, seq_len, N_ARROW_TYPES).
    """

    def __init__(
        self,
        num_note_kinds: int = _N_NOTE_KINDS,
        ignore_class: int = 0,
        name: str = "arrow_note_kind_dist_match",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.num_note_kinds = num_note_kinds
        self.ignore_class = ignore_class
        self.eps = 1e-7
        self.pred_counts = self.add_weight(
            name="pred_note_kind_counts",
            shape=(num_note_kinds,),
            initializer="zeros",
        )
        self.true_counts = self.add_weight(
            name="true_note_kind_counts",
            shape=(num_note_kinds,),
            initializer="zeros",
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.int32)
        y_pred = tf.cast(y_pred, tf.float32)
        mask = tf.cast(tf.not_equal(y_true, self.ignore_class), tf.float32)
        pred_classes = tf.cast(tf.argmax(y_pred, axis=-1), tf.int32)
        # Map arrow codes to note kinds via lookup (batch, seq) -> (batch, seq)
        pred_kinds = tf.gather(
            _ARROW_NOTE_KIND_TABLE,
            tf.clip_by_value(pred_classes, 0, constants.N_ARROW_TYPES - 1),
        )
        true_kinds = tf.gather(
            _ARROW_NOTE_KIND_TABLE,
            tf.clip_by_value(y_true, 0, constants.N_ARROW_TYPES - 1),
        )
        true_onehot = tf.one_hot(
            true_kinds, depth=self.num_note_kinds, axis=-1, dtype=tf.float32
        )
        pred_onehot = tf.one_hot(
            pred_kinds, depth=self.num_note_kinds, axis=-1, dtype=tf.float32
        )
        mask_exp = tf.expand_dims(mask, axis=-1)
        true_inc = tf.reduce_sum(mask_exp * true_onehot, axis=[0, 1])
        pred_inc = tf.reduce_sum(mask_exp * pred_onehot, axis=[0, 1])
        self.true_counts.assign_add(true_inc)
        self.pred_counts.assign_add(pred_inc)

    def result(self):
        p = self.pred_counts / (tf.reduce_sum(self.pred_counts) + self.eps)
        q = self.true_counts / (tf.reduce_sum(self.true_counts) + self.eps)
        m = 0.5 * (p + q)
        kl_p = tf.reduce_sum(
            p * (tf.math.log(p + self.eps) - tf.math.log(m + self.eps))
        )
        kl_q = tf.reduce_sum(
            q * (tf.math.log(q + self.eps) - tf.math.log(m + self.eps))
        )
        jsd = 0.5 * (kl_p + kl_q)
        return 1.0 - jsd

    def reset_state(self):
        self.pred_counts.assign(tf.zeros_like(self.pred_counts))
        self.true_counts.assign(tf.zeros_like(self.true_counts))

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_note_kinds": self.num_note_kinds,
                "ignore_class": self.ignore_class,
            }
        )
        return config


def compute_hold_validity_violations(
    arrow_codes: np.ndarray,
    *,
    max_examples: int = 5,
) -> tuple[int, int, list[tuple[int, int, str]]]:
    """Check hold validity on a 1D sequence of arrow codes (0..255).

    Uses the same rules as ArrowHoldValidityMetric: (1) every 3 (hold end) must
    have a preceding 2 (hold start) in that column; (2) 3 cannot immediately
    follow 1 (tap) in the same column.

    Args:
        arrow_codes: 1D array of integer arrow codes, shape (seq_len,).
        max_examples: Maximum number of example violations to return (default 5).

    Returns:
        (num_violations, num_hold_ends, examples). num_hold_ends is the total
        count of 3s. examples is a list of (step_index, column, kind_str) with
        kind_str "unmatched_3" or "3_after_1", usable for lookup in the chart file.
    """
    arrow_codes = np.asarray(arrow_codes, dtype=np.int32).ravel()
    seq_len = len(arrow_codes)
    # (seq_len, 4) per-column digits 0..3
    digits = np.zeros((seq_len, 4), dtype=np.int32)
    for i in range(seq_len):
        n = int(np.clip(arrow_codes[i], 0, 255))
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
            if col[i] == 2:
                run += 1
            elif col[i] == 3:
                hold_ends += 1
                if run < 1:
                    violations += 1
                    if len(examples) < max_examples:
                        examples.append((i, c, "unmatched_3"))
                else:
                    run -= 1
                if i > 0 and col[i - 1] == 1:
                    violations += 1
                    if len(examples) < max_examples:
                        examples.append((i, c, "3_after_1"))
    return int(violations), int(hold_ends), examples


def _arrow_indices_to_column_digits(indices: tf.Tensor) -> tf.Tensor:
    """Convert (batch, seq) of arrow codes to (batch, seq, 4) of per-column values 0..3."""
    # indices: (batch, seq). Each value n: d0=(n//64)%4, d1=(n//16)%4, d2=(n//4)%4, d3=n%4
    n = tf.cast(tf.clip_by_value(indices, 0, 255), tf.int32)
    d0 = tf.math.mod(tf.math.floordiv(n, 64), 4)
    d1 = tf.math.mod(tf.math.floordiv(n, 16), 4)
    d2 = tf.math.mod(tf.math.floordiv(n, 4), 4)
    d3 = tf.math.mod(n, 4)
    return tf.stack([d0, d1, d2, d3], axis=-1)


@keras.saving.register_keras_serializable()
class ArrowHoldValidityMetric(keras.metrics.Metric):
    """
    Metric that measures hold/release validity of predicted arrow sequences.
    Rules: (1) Every 3 (hold end) must have a preceding 2 (hold start) in that column.
    (2) 3 cannot immediately follow 1 (tap) in the same column.
    Returns 1 - (violations / max(1, total_hold_ends)); 1.0 when no hold ends.
    Uses ignore_class=0 (padding positions are skipped when building per-column sequences).
    """

    def __init__(
        self,
        ignore_class: int = 0,
        name: str = "arrow_hold_validity",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.ignore_class = ignore_class
        self.eps = 1e-7
        self.total_violations = self.add_weight(
            name="total_violations", initializer="zeros"
        )
        self.total_hold_ends = self.add_weight(
            name="total_hold_ends", initializer="zeros"
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.int32)
        y_pred = tf.cast(y_pred, tf.float32)
        pred_classes = tf.cast(tf.argmax(y_pred, axis=-1), tf.int32)
        # Only consider positions that are not padding
        mask = tf.not_equal(y_true, self.ignore_class)
        # Shape (batch, seq, 4): digit value 0..3 per column
        digits = _arrow_indices_to_column_digits(pred_classes)  # type: ignore[arg-type]
        # For padding positions, treat as 0 so they don't create spurious 2/3
        mask_exp = tf.expand_dims(tf.cast(mask, tf.float32), axis=-1)
        digits_f = tf.cast(digits, tf.float32)
        digits_masked = tf.where(
            tf.broadcast_to(mask_exp > 0.5, tf.shape(digits_f)),
            digits_f,
            tf.zeros_like(digits_f),
        )
        digits = tf.cast(digits_masked, tf.int32)
        # Running balance per column: +1 for 2, -1 for 3. (batch, seq, 4)
        delta = tf.where(
            tf.equal(digits, 2),
            tf.ones_like(digits, dtype=tf.float32),
            tf.where(
                tf.equal(digits, 3),
                -tf.ones_like(digits, dtype=tf.float32),
                tf.zeros_like(digits, dtype=tf.float32),
            ),
        )
        run = tf.cumsum(delta, axis=1)
        run_prev = tf.concat(
            [tf.zeros_like(run[:, :1, :]), run[:, :-1, :]],  # type: ignore[call-overload]
            axis=1,
        )
        # Violation 1: 3 with no preceding 2 (balance before this step < 1)
        violation_unmatched = tf.cast(
            tf.equal(digits, 3) & (run_prev < 1.0), tf.float32
        )
        # Violation 2: 3 immediately after 1 in same column
        digits_prev = tf.concat(
            [tf.zeros_like(digits[:, :1, :]), digits[:, :-1, :]],  # type: ignore[index]
            axis=1,
        )
        violation_after_tap = tf.cast(
            tf.equal(digits, 3) & tf.equal(digits_prev, 1), tf.float32
        )
        violations_batch = tf.reduce_sum(
            tf.add(violation_unmatched, violation_after_tap)
        )
        hold_ends_batch = tf.reduce_sum(tf.cast(tf.equal(digits, 3), tf.float32))
        self.total_violations.assign_add(violations_batch)
        self.total_hold_ends.assign_add(hold_ends_batch)

    def result(self):
        denom = self.total_hold_ends + self.eps
        return 1.0 - (self.total_violations / denom)

    def reset_state(self):
        self.total_violations.assign(0.0)
        self.total_hold_ends.assign(0.0)

    def get_config(self):
        config = super().get_config()
        config.update({"ignore_class": self.ignore_class})
        return config
