"""Arrow training loss functions: focal, label-smoothed cross-entropy, and aux interval MSE."""

import keras
import tensorflow as tf

from stepcovnet import config, constants, metrics


def sparse_focal_loss(
    y_true: tf.Tensor,
    y_pred: tf.Tensor,
    gamma: float,
    ignore_class: int = constants.ARROW_PADDING_CLASS,
) -> tf.Tensor:
    """Sparse categorical focal loss: - (1 - p_t)^gamma * log(p_t), masked for ignore_class.

    Args:
        y_true: (batch, steps) int class indices.
        y_pred: (batch, steps, num_classes) float probabilities.
        gamma: Focusing parameter (higher down-weights easy examples).
        ignore_class: Class index to exclude from loss (e.g. padding).

    Returns:
        Scalar mean loss over valid (non-ignored) positions.
    """
    y_true_int = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
    indices = tf.range(tf.size(y_true_int))
    flat_pred = tf.reshape(y_pred, [-1, constants.N_ARROW_TYPES])
    p_t = tf.gather_nd(flat_pred, tf.stack([indices, y_true_int], axis=1))
    p_t = tf.reshape(p_t, tf.shape(y_true))
    _max_p = 1.0 - 1e-7
    p_t = tf.clip_by_value(p_t, 1e-7, _max_p)
    focal_weight = tf.pow(tf.subtract(1.0, p_t), gamma)
    ce = tf.negative(tf.math.log(p_t))
    loss_per_step = focal_weight * ce
    mask = tf.cast(tf.not_equal(y_true, ignore_class), tf.float32)
    loss_sum = tf.reduce_sum(loss_per_step * mask)
    count = tf.maximum(tf.reduce_sum(mask), 1.0)
    return loss_sum / count


def arrow_label_smoothed_crossentropy(
    y_true: tf.Tensor, y_pred: tf.Tensor, smoothing: float
) -> tf.Tensor:
    """Cross-entropy with label smoothing over valid (non-ignore) positions.

    Args:
        y_true: (batch, steps) int class indices.
        y_pred: (batch, steps, num_classes) logits or probabilities.
        smoothing: Label smoothing factor in (0, 1).

    Returns:
        Scalar mean loss over valid (non-zero) positions.
    """
    one_hot = tf.one_hot(
        tf.cast(tf.reshape(y_true, [-1]), tf.int32),
        constants.N_ARROW_TYPES,
    )
    one_hot = tf.reshape(
        one_hot,
        tf.concat([tf.shape(y_true), [constants.N_ARROW_TYPES]], axis=0),
    )
    smoothed = one_hot * (1.0 - smoothing) + smoothing / constants.N_ARROW_TYPES
    mask = tf.cast(tf.not_equal(y_true, constants.ARROW_PADDING_CLASS), tf.float32)
    cat_ce = keras.losses.CategoricalCrossentropy(label_smoothing=0.0, reduction="none")
    per_step = cat_ce(smoothed, y_pred)
    return tf.reduce_sum(per_step * mask) / tf.maximum(tf.reduce_sum(mask), 1.0)


def masked_mse_aux_interval(
    y_true: tf.Tensor, y_pred: tf.Tensor, sample_weight: tf.Tensor | None = None
) -> tf.Tensor:
    """MSE for aux_interval regression; when sample_weight given, mask invalid steps.

    Args:
        y_true: (batch, steps, 1) target next-interval.
        y_pred: (batch, steps, 1) predicted next-interval.
        sample_weight: (batch, steps, 1) mask (1 = valid step, 0 = last step / padding).

    Returns:
        Scalar: mean squared error over valid (masked) positions.
    """
    sq = tf.square(tf.subtract(y_pred, y_true))
    if sample_weight is None:
        return tf.reduce_mean(sq)
    return tf.reduce_sum(sq * sample_weight) / tf.maximum(
        tf.reduce_sum(sample_weight), 1.0
    )


def build_arrow_combined_loss(
    run_config: config.ArrowRunConfig,
):
    """Build the combined arrow loss (main + validity + diversity, optional rejection gate).

    Args:
        run_config: Arrow run configuration (loss type, weights, rejection params).

    Returns:
        A callable (y_true, y_pred) -> scalar Tensor suitable for use as a Keras loss.
    """
    w_val = run_config.chart_validity_aux_weight
    w_div = run_config.diversity_aux_weight
    rej_threshold = run_config.chart_validity_rejection_threshold
    rej_scale = run_config.chart_validity_rejection_scale
    rej_temp = run_config.chart_validity_rejection_temperature

    if run_config.loss_type == "crossentropy":
        if run_config.label_smoothing > 0:
            _smoothing = run_config.label_smoothing

            def _main_loss_fn(y_true, y_pred):
                return arrow_label_smoothed_crossentropy(y_true, y_pred, _smoothing)
        else:
            _main_loss_fn = keras.losses.SparseCategoricalCrossentropy(
                ignore_class=constants.ARROW_PADDING_CLASS
            )  # type: ignore
    else:
        _gamma = run_config.focal_gamma

        def _main_loss_fn(y_true, y_pred):
            return sparse_focal_loss(
                y_true,
                y_pred,
                gamma=_gamma,
                ignore_class=constants.ARROW_PADDING_CLASS,
            )

    def combined(y_true, y_pred):
        main = _main_loss_fn(y_true, y_pred)
        validity = metrics.chart_validity_auxiliary_loss(
            y_true, y_pred, ignore_class=constants.ARROW_PADDING_CLASS
        )
        diversity = metrics.note_kind_balance_auxiliary_loss(
            y_true, y_pred, ignore_class=constants.ARROW_PADDING_CLASS
        )
        if rej_threshold is None:
            return main + tf.multiply(validity, w_val) + tf.multiply(diversity, w_div)
        validity_score = tf.subtract(1.0, validity)
        gate = tf.sigmoid((validity_score - rej_threshold) * rej_temp)
        return (
            gate * (main + tf.multiply(diversity, w_div))
            + (1.0 - gate) * rej_scale * validity
        )

    return combined
