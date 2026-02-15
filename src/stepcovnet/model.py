"""Model architectures and custom Keras components for onset detection."""

import keras
import tensorflow as tf


def _wavenet_residual_block(inputs, residual_channels, skip_channels, dilation_rate, kernel_size, block_id) -> tuple:
    """
    Creates a single residual block from the WaveNet architecture.
    This is the core building block for the U-Net.

    Args:
        inputs: Input tensor of shape (batch, time, channels).
        residual_channels: The number of channels for the residual path.
        skip_channels: The number of channels for the skip connection path.
        dilation_rate: The dilation factor for the causal convolution.
        kernel_size: The size of the convolutional kernel.
        block_id: A unique identifier (string or int) for naming the layers.

    Returns:
        A tuple containing the residual output and the skip connection output.
    """
    prefix = f"wavenet_block_{block_id}_d{dilation_rate}"

    # Gated Activation Unit
    x = inputs
    x_conv = keras.layers.Conv1D(filters=residual_channels * 2, kernel_size=kernel_size, padding='causal',
                                 dilation_rate=dilation_rate, name=f"{prefix}_dilated_conv")(x)

    x_tanh = keras.layers.Activation('tanh', name=f"{prefix}_tanh")(x_conv[:, :, :residual_channels])
    x_sigmoid = keras.layers.Activation('sigmoid', name=f"{prefix}_sigmoid")(x_conv[:, :, residual_channels:])

    gated_output = keras.layers.Multiply(name=f"{prefix}_multiply")([x_tanh, x_sigmoid])

    res_output = keras.layers.Conv1D(filters=residual_channels, kernel_size=1, name=f"{prefix}_residual_conv")(
        gated_output)
    skip_output = keras.layers.Conv1D(filters=skip_channels, kernel_size=1, name=f"{prefix}_skip_conv")(gated_output)

    residual = keras.layers.Add(name=f"{prefix}_add_residual")([inputs, res_output])

    residual = keras.layers.LayerNormalization(name=f"{prefix}_layernorm")(residual)

    return residual, skip_output


def build_unet_wavenet_model(input_shape=(None, 128),
                             initial_filters=32,
                             depth=4,
                             dilation_rates=[1, 2, 4, 8],
                             kernel_size=3,
                             dropout_rate=0.3,
                             experiment_name="") -> keras.Model:
    """
    Builds a U-Net style WaveNet for multi-scale rhythmic analysis.

    The U-Net architecture uses an encoder to learn features at progressively coarser
    time scales and a decoder that uses this context to reconstruct a precise output.
    Skip connections between the encoder and decoder are crucial for combining
    high-level context with low-level timing information.

    Args:
        input_shape: The shape of the input data (time_steps, n_features).
        initial_filters: The number of filters in the first layer. This will double at each encoder level.
        depth: The number of downsampling/upsampling levels in the U-Net.
        dilation_rates: A list of dilation factors for the convolutions within each level.
        kernel_size: The size of the convolutional kernel.
        dropout_rate: The dropout rate for regularization.
        experiment_name: A string to append to the model name.

    Returns:
        A Keras Model instance.
    """
    inputs = keras.Input(shape=input_shape, name="input_features")
    x = inputs

    encoder_outputs = []

    # --- Encoder Path (Downsampling) ---
    for i in range(depth):
        level_prefix = f"encoder_level_{i}"
        current_filters = initial_filters * (2 ** i)

        # Project input to the current filter size if necessary
        x = keras.layers.Conv1D(filters=current_filters, kernel_size=1, name=f"{level_prefix}_projection")(x)

        # Apply a few WaveNet blocks at this resolution
        for rate in dilation_rates:
            x, _ = _wavenet_residual_block(x, current_filters, current_filters, rate, kernel_size,
                                           f"{level_prefix}_{rate}")

        encoder_outputs.append(x)

        # Downsample for the next level using a strided convolution
        x = keras.layers.Conv1D(filters=initial_filters * (2 ** (i + 1)), kernel_size=3, strides=2,
                                padding='same', name=f"{level_prefix}_downsample")(x)

    # --- Bottleneck ---
    bottleneck_prefix = "bottleneck"
    bottleneck_filters = initial_filters * (2 ** depth)
    x = keras.layers.Conv1D(filters=bottleneck_filters, kernel_size=1, name=f"{bottleneck_prefix}_projection")(x)
    for rate in dilation_rates:
        x, _ = _wavenet_residual_block(x, bottleneck_filters, bottleneck_filters, rate, kernel_size,
                                       f"{bottleneck_prefix}_{rate}")

    # The bottleneck contains the most abstract, high-level features. Applying dropout
    # here prevents the model from relying too heavily on any single abstract feature.
    x = keras.layers.Dropout(dropout_rate, name=f"{bottleneck_prefix}_dropout")(x)

    # --- Decoder Path (Upsampling) ---
    for i in reversed(range(depth)):
        level_prefix = f"decoder_level_{i}"
        current_filters = initial_filters * (2 ** i)

        # Upsample using a transposed convolution
        x = keras.layers.Conv1DTranspose(filters=current_filters, kernel_size=3, strides=2,
                                         padding='same', name=f"{level_prefix}_upsample")(x)

        skip_connection = encoder_outputs[i]

        @keras.saving.register_keras_serializable()
        def crop_to_match(inputs):
            # The first input is the tensor to crop, the second is the reference.
            tensor_to_crop, reference_tensor = inputs
            # Get the dynamic sequence length of the reference tensor.
            target_length = tf.shape(reference_tensor)[1]
            # Crop the first tensor to match this length.
            return tensor_to_crop[:, :target_length, :]

        # The Lambda layer takes a list of tensors as input to ensure dynamic shapes are handled correctly.
        x = keras.layers.Lambda(crop_to_match, name=f"{level_prefix}_crop_to_match")([x, skip_connection])

        # Concatenate with the skip connection from the corresponding encoder level
        x = keras.layers.Concatenate(name=f"{level_prefix}_concat_skip")([x, skip_connection])

        # This 1x1 convolution projects it back to the expected number of filters
        x = keras.layers.Conv1D(filters=current_filters, kernel_size=1, name=f"{level_prefix}_post_concat_projection")(
            x)

        # This is another critical location. Dropout here prevents the model from
        # simply learning to pass-through features from the skip connection without
        # properly integrating them with the high-level context from the decoder.
        x = keras.layers.Dropout(dropout_rate, name=f"{level_prefix}_dropout")(x)

        # Apply WaveNet blocks at this resolution to refine features
        for rate in dilation_rates:
            x, _ = _wavenet_residual_block(x, current_filters, current_filters, rate, kernel_size,
                                           f"{level_prefix}_{rate}")

    # --- Output Head ---
    # Applying dropout before the final refinement layers is a standard practice
    # that helps regularize the final classification/regression stage.
    x = keras.layers.Dropout(dropout_rate, name="pre_output_dropout")(x)

    x = keras.layers.Conv1D(filters=16, kernel_size=1, activation='gelu', name="output_conv_1")(x)
    outputs = keras.layers.Conv1D(filters=1, kernel_size=1, activation='sigmoid', name="output_sigmoid")(x)

    model_name = "UNet-WaveNet"
    if experiment_name:
        model_name += f"-{experiment_name}"

    return keras.Model(inputs, outputs, name=model_name)
