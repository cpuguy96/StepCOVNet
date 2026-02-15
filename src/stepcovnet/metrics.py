"""Custom metrics and divergence functions for evaluating step detection models."""

import keras
import tensorflow as tf
from keras import backend as K


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

    def __init__(self, tolerance=1, threshold=0.5, name='onset_f1_score', **kwargs):
        super().__init__(name=name, **kwargs)
        self.tolerance = tolerance
        self.threshold = threshold
        # Calculate the full window size for convolution
        self.window_size = 2 * self.tolerance + 1

        # State variables to accumulate counts across batches
        self.true_positives = self.add_weight(name='tp', initializer='zeros')
        self.false_positives = self.add_weight(name='fp', initializer='zeros')
        self.false_negatives = self.add_weight(name='fn', initializer='zeros')

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
        y_true_conv = tf.case([
            (tf.equal(rank_true, 2), lambda: tf.reshape(y_true, [shape_true[0], shape_true[1], 1])),
            (tf.equal(rank_true, 4), lambda: tf.reshape(y_true, [shape_true[0], shape_true[1], 1]))
            # Squeeze rank 4 -> 3
        ],
            default=lambda: y_true,  # Assumes rank 3 is already correct
            exclusive=True,
            name='reshape_y_true'
        )
        # Set a general shape hint after reshaping
        y_true_conv.set_shape([None, None, 1])

        # Reshape y_pred
        rank_pred = tf.rank(y_pred)
        shape_pred = tf.shape(y_pred)
        # Define reshaping logic based on rank
        y_pred_proc = tf.case([
            (tf.equal(rank_pred, 2), lambda: tf.reshape(y_pred, [shape_pred[0], shape_pred[1], 1])),
            (tf.equal(rank_pred, 4), lambda: tf.reshape(y_pred, [shape_pred[0], shape_pred[1], 1]))
            # Squeeze rank 4 -> 3
        ],
            default=lambda: y_pred,  # Assumes rank 3 is already correct
            exclusive=True,
            name='reshape_y_pred'
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
        padding = [[0, 0], [self.tolerance, self.tolerance], [0, 0]]  # Pad only the time dimension
        y_true_padded = tf.pad(y_true_conv, padding, "CONSTANT")
        y_pred_padded = tf.pad(y_pred_binary_conv, padding, "CONSTANT")

        # Convolve y_true: Marks regions within `tolerance` of a true onset
        # Output shape: (batch_size, time_steps, 1)
        true_onset_windows = tf.nn.conv1d(
            y_true_padded,
            filters=kernel,
            stride=1,
            padding='VALID'  # Use 'VALID' padding with manually padded input
        )
        # Result > 0 means a true onset is within the window at that point
        true_onset_windows_bool = true_onset_windows > 0

        # Convolve y_pred_binary: Marks regions within `tolerance` of a predicted onset
        pred_onset_windows = tf.nn.conv1d(
            y_pred_padded,
            filters=kernel,
            stride=1,
            padding='VALID'
        )
        # Result > 0 means a predicted onset is within the window
        pred_onset_windows_bool = pred_onset_windows > 0

        # --- Calculate TP, FP, FN ---
        # Perform calculations using the 3D tensors (y_pred_binary_conv, y_true_conv)

        # True Positives (TP): Predicted onset falls within the tolerance window of a true onset.
        # Check where y_pred_binary_conv is 1 AND a true onset is nearby (true_onset_windows > 0)
        tp = tf.reduce_sum(
            tf.cast(tf.logical_and(tf.cast(y_pred_binary_conv, tf.bool), true_onset_windows_bool), tf.float32))

        # False Positives (FP): Predicted onset does NOT fall within the tolerance window of any true onset.
        # Check where y_pred_binary_conv is 1 AND no true onset is nearby (true_onset_windows == 0)
        fp = tf.reduce_sum(
            tf.cast(tf.logical_and(tf.cast(y_pred_binary_conv, tf.bool), tf.logical_not(true_onset_windows_bool)),
                    tf.float32))

        # False Negatives (FN): True onset does NOT have any predicted onsets within its tolerance window.
        # Check where y_true_conv is 1 AND no predicted onset is nearby (pred_onset_windows == 0)
        fn = tf.reduce_sum(
            tf.cast(tf.logical_and(tf.cast(y_true_conv, tf.bool), tf.logical_not(pred_onset_windows_bool)), tf.float32))

        # Update state variables
        self.true_positives.assign_add(tp)
        self.false_positives.assign_add(fp)
        self.false_negatives.assign_add(fn)

    def result(self):
        """
        Computes and returns the F1 score.
        """
        # Calculate Precision
        precision = self.true_positives / (self.true_positives + self.false_positives + K.epsilon())
        # Calculate Recall
        recall = self.true_positives / (self.true_positives + self.false_negatives + K.epsilon())
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
        config.update({
            "tolerance": self.tolerance,
            "threshold": self.threshold,
        })
        return config
