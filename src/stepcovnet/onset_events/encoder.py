"""Temporal encoder on frontend embeddings for event-based onset detection."""

import keras

from stepcovnet import models

# Filters on the last Conv1D before the dense onset head in build_unet_wavenet_model.
_ENCODER_OUTPUT_FEATURES = 16


def build_temporal_encoder(
    input_features: int,
    encoder_config: dict,
) -> keras.Model:
    """Build a U-Net-style temporal encoder on ``(B, T, D)`` embeddings.

    Wraps ``models.build_unet_wavenet_model`` and returns multi-channel features
    per time step (no per-frame sigmoid onset head).

    Args:
        input_features: Channel dimension ``D`` from the audio frontend.
        encoder_config: ``initial_filters``, ``depth``, ``dilation_rates``,
            ``kernel_size``, ``dropout_rate``; optional ``model_name`` suffix
            for the inner U-Net.

    Returns:
        Keras model mapping ``(B, T, input_features)`` to
        ``(B, T, 16)`` feature maps.
    """
    dilation_rates = encoder_config.get("dilation_rates")
    base = models.build_unet_wavenet_model(
        initial_filters=int(encoder_config.get("initial_filters", 16)),
        depth=int(encoder_config.get("depth", 2)),
        dilation_rates=dilation_rates,
        kernel_size=int(encoder_config.get("kernel_size", 3)),
        dropout_rate=float(encoder_config.get("dropout_rate", 0.0)),
        model_name=str(encoder_config.get("model_name", "onset_event")),
        input_features=input_features,
    )
    features = base.get_layer("output_conv_1").output
    return keras.Model(
        inputs=base.input,
        outputs=features,
        name="onset_event_temporal_encoder",
    )
