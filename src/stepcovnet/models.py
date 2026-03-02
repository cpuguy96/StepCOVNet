"""Model architectures and custom Keras components for onset detection."""

from __future__ import annotations

from collections.abc import Callable

import keras
import tensorflow as tf

from stepcovnet import config, constants


@keras.saving.register_keras_serializable()
class SnippetCNN(keras.layers.Layer):
    """2D CNN over each (n_frames, n_mels) snippet without TimeDistributed.

    Reshapes (batch, steps, n_frames, n_mels) to (batch*steps, n_frames, n_mels),
    applies Conv2D layers and global average pooling, then reshapes back to
    (batch, steps, filters). Avoids TimeDistributed for XLA compatibility.
    """

    def __init__(self, n_frames, n_mels, filters=32, **kwargs):
        super().__init__(**kwargs)
        self.n_frames = n_frames
        self.n_mels = n_mels
        self.filters = filters
        self.conv1 = keras.layers.Conv2D(
            filters, (3, 3), activation="relu", padding="same", name="snippet_conv2d_1"
        )
        self.conv2 = keras.layers.Conv2D(
            filters, (3, 3), activation="relu", padding="same", name="snippet_conv2d_2"
        )
        self.pool = keras.layers.GlobalAveragePooling2D(name="snippet_pool")

    def call(self, inputs):
        # (batch, steps, n_frames, n_mels) -> (batch*steps, n_frames, n_mels, 1)
        shape = tf.shape(inputs)
        b = tf.gather(shape, 0)
        s = tf.gather(shape, 1)
        flat = tf.reshape(inputs, (b * s, self.n_frames, self.n_mels, 1))
        x = self.conv1(flat)
        x = self.conv2(x)
        x = self.pool(x)  # (batch*steps, filters)
        return tf.reshape(x, (b, s, self.filters))

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "n_frames": self.n_frames,
                "n_mels": self.n_mels,
                "filters": self.filters,
            }
        )
        return config


@keras.saving.register_keras_serializable()
class PositionalEncoding(keras.layers.Layer):
    def __init__(self, position, d_model, **kwargs):
        # Add **kwargs to accept base Layer arguments like 'name'
        super().__init__(**kwargs)
        # Ensure d_model is compatible with potential float16 usage later
        if d_model % 2 != 0:
            raise ValueError(
                "PositionalEncoding requires an even d_model so that sine and cosine "
                "components can be interleaved without shape mismatch."
            )
        self.d_model = d_model
        self.position = position
        # Pre-calculate the positional encoding matrix.
        # Calculate using float32 for precision, will cast later if needed.
        self.pos_encoding = self.positional_encoding(position, d_model)

    def get_angles(self, position, i, d_model):
        # Ensure d_model is float for the calculation
        d_model_float = tf.cast(d_model, tf.float32)
        # Calculate the angles for the positional encoding formula
        # Original formula: angle = pos / (10000^(2i / d_model))
        # Use floating point literals and casting for compatibility
        angles = 1.0 / tf.pow(
            10000.0,
            (2.0 * tf.cast(i // 2, tf.float32)) / d_model_float,  # type: ignore
        )
        return tf.cast(position, tf.float32) * angles

    def positional_encoding(self, position, d_model):
        # Create angle radians matrix (using float32 for calculation precision)
        angle_rads = self.get_angles(
            tf.range(position, dtype=tf.float32)[:, tf.newaxis],  # Use float32 range
            tf.range(d_model, dtype=tf.float32)[tf.newaxis, :],  # Use float32 range
            d_model,  # d_model is passed to get_angles which casts it
        )  # Shape: (position, d_model)

        # Apply sin to even indices in the array; 2i
        sines = tf.math.sin(angle_rads[:, 0::2])  # Shape: (position, d_model/2)

        # Apply cos to odd indices in the array; 2i+1
        cosines = tf.math.cos(angle_rads[:, 1::2])  # Shape: (position, d_model/2)

        # Interleave sines and cosines
        pos_encoding = tf.stack(
            [sines, cosines], axis=-1
        )  # Shape: (position, d_model/2, 2)
        pos_encoding = tf.reshape(
            pos_encoding, [position, d_model]
        )  # Shape: (position, d_model)

        # Add batch dimension for broadcasting
        pos_encoding = pos_encoding[tf.newaxis, ...]  # Shape: (1, position, d_model)
        # Return as float32, will be cast in call() if necessary
        return tf.cast(pos_encoding, tf.float32)

    def call(self, inputs):
        # inputs shape: (batch_size, seq_len, d_model)
        seq_len = tf.shape(inputs)[1]  # type: ignore
        input_dtype = inputs.dtype
        pos_encoding_sliced = self.pos_encoding[:, :seq_len, :]  # type: ignore
        pos_encoding_casted = tf.cast(pos_encoding_sliced, dtype=input_dtype)
        return inputs + pos_encoding_casted

    # Optional: Implement compute_output_shape for better static shape inference
    def compute_output_shape(self, input_shape):
        return input_shape


def _wavenet_residual_block(
    inputs, residual_channels, skip_channels, dilation_rate, kernel_size, block_id
) -> tuple:
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
    x_conv = keras.layers.Conv1D(
        filters=residual_channels * 2,
        kernel_size=kernel_size,
        padding="causal",
        dilation_rate=dilation_rate,
        name=f"{prefix}_dilated_conv",
    )(x)

    x_tanh = keras.layers.Activation("tanh", name=f"{prefix}_tanh")(
        x_conv[:, :, :residual_channels]
    )
    x_sigmoid = keras.layers.Activation("sigmoid", name=f"{prefix}_sigmoid")(
        x_conv[:, :, residual_channels:]
    )

    gated_output = keras.layers.Multiply(name=f"{prefix}_multiply")([x_tanh, x_sigmoid])

    res_output = keras.layers.Conv1D(
        filters=residual_channels, kernel_size=1, name=f"{prefix}_residual_conv"
    )(gated_output)
    skip_output = keras.layers.Conv1D(
        filters=skip_channels, kernel_size=1, name=f"{prefix}_skip_conv"
    )(gated_output)

    residual = keras.layers.Add(name=f"{prefix}_add_residual")([inputs, res_output])

    residual = keras.layers.LayerNormalization(name=f"{prefix}_layernorm")(residual)

    return residual, skip_output


def _transformer_encoder(
    inputs,
    d_model: int,
    num_heads: int,
    ff_dim: int,
    dropout_rate: float = 0.1,
    *,
    name: str = "transformer_block",
):
    """
    Creates a single Transformer Encoder block.
    Args:
        inputs: Input tensor shape (batch_size, seq_len, d_model)
        d_model: Dimensionality of the model.
        num_heads: Number of attention heads.
        ff_dim: Inner dimension of the Feed-Forward Network.
        dropout_rate: Dropout rate.
        name: Prefix for layer names in this block.
    Returns:
        Output tensor shape (batch_size, seq_len, d_model)
    """
    prefix = name
    # --- Multi-Head Self-Attention ---
    # Ensure d_model is divisible by num_heads
    assert d_model % num_heads == 0
    kv_dim = d_model // num_heads  # Dimension of each attention head's key/query/value

    attn_output = keras.layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=kv_dim,
        value_dim=kv_dim,
        dtype="float32",  # Needed for numerical stability during inference
        name=f"{prefix}_mha",
    )(inputs, inputs)
    attn_output = keras.layers.Dropout(dropout_rate, name=f"{prefix}_attn_dropout")(
        attn_output
    )
    # Residual connection & Layer Normalization
    out1 = keras.layers.LayerNormalization(
        epsilon=1e-6, name=f"{prefix}_attn_layernorm"
    )(inputs + attn_output)

    # --- Feed-Forward Network ---
    ffn_output = keras.layers.Dense(
        ff_dim, activation="relu", name=f"{prefix}_ffn_dense1"
    )(out1)
    ffn_output = keras.layers.Dense(d_model, name=f"{prefix}_ffn_dense2")(ffn_output)
    ffn_output = keras.layers.Dropout(dropout_rate, name=f"{prefix}_ffn_dropout")(
        ffn_output
    )
    # Residual connection & Layer Normalization
    out2 = keras.layers.LayerNormalization(
        epsilon=1e-6, name=f"{prefix}_ffn_layernorm"
    )(out1 + ffn_output)

    return out2


@keras.saving.register_keras_serializable()
def _crop_to_match(inputs):
    """Crops the first input tensor to match the temporal length of the second tensor.

    Args:
        inputs: A list or tuple containing [tensor_to_crop, reference_tensor].

    Returns:
        The first tensor cropped along the time dimension (axis 1).
    """
    # The first input is the tensor to crop, the second is the reference.
    tensor_to_crop, reference_tensor = inputs
    # Get the dynamic sequence length of the reference tensor.
    target_length = tf.shape(reference_tensor)[1]  # type: ignore
    # Crop the first tensor to match this length.
    return tensor_to_crop[:, :target_length, :]


def build_unet_wavenet_model(
    initial_filters: int = 16,
    depth: int = 2,
    dilation_rates: list[int] | None = None,
    kernel_size: int = 3,
    dropout_rate: float = 0.0,
    model_name: str = "",
) -> keras.Model:
    """Builds a U-Net style WaveNet for multi-scale rhythmic analysis.

    The U-Net architecture uses an encoder to learn features at progressively
    coarser
    time scales and a decoder that uses this context to reconstruct a precise
    output.
    Skip connections between the encoder and decoder are crucial for combining
    high-level context with low-level timing information.

    Args:
        initial_filters: The number of filters in the first layer. This will
        double at each encoder level.
        depth: The number of downsampling/upsampling levels in the U-Net.
        dilation_rates: A list of dilation factors for the convolutions
        within each level.
        kernel_size: The size of the convolutional kernel.
        dropout_rate: The dropout rate for regularization.
        model_name: The name of the model.

    Returns:
        A Keras Model instance.
    """
    if dilation_rates is None:
        dilation_rates = [1, 2, 4, 8]

    inputs = keras.Input(shape=(None, constants.N_MELS), name="input_features")
    x = inputs

    encoder_outputs = []

    # --- Encoder Path (Downsampling) ---
    for i in range(depth):
        level_prefix = f"encoder_level_{i}"
        current_filters = initial_filters * (2**i)

        # Project input to the current filter size if necessary
        x = keras.layers.Conv1D(
            filters=current_filters, kernel_size=1, name=f"{level_prefix}_projection"
        )(x)

        # Apply a few WaveNet blocks at this resolution
        for rate in dilation_rates:
            x, _ = _wavenet_residual_block(
                x,
                current_filters,
                current_filters,
                rate,
                kernel_size,
                f"{level_prefix}_{rate}",
            )

        encoder_outputs.append(x)

        # Downsample for the next level using a strided convolution
        x = keras.layers.Conv1D(
            filters=initial_filters * (2 ** (i + 1)),
            kernel_size=3,
            strides=2,
            padding="same",
            name=f"{level_prefix}_downsample",
        )(x)

    # --- Bottleneck ---
    bottleneck_prefix = "bottleneck"
    bottleneck_filters = initial_filters * (2**depth)
    x = keras.layers.Conv1D(
        filters=bottleneck_filters,
        kernel_size=1,
        name=f"{bottleneck_prefix}_projection",
    )(x)
    for rate in dilation_rates:
        x, _ = _wavenet_residual_block(
            x,
            bottleneck_filters,
            bottleneck_filters,
            rate,
            kernel_size,
            f"{bottleneck_prefix}_{rate}",
        )

    # The bottleneck contains the most abstract, high-level features. Applying dropout
    # here prevents the model from relying too heavily on any single abstract feature.
    x = keras.layers.Dropout(dropout_rate, name=f"{bottleneck_prefix}_dropout")(x)

    # --- Decoder Path (Upsampling) ---
    for i in reversed(range(depth)):
        level_prefix = f"decoder_level_{i}"
        current_filters = initial_filters * (2**i)

        # Upsample using a transposed convolution
        x = keras.layers.Conv1DTranspose(
            filters=current_filters,
            kernel_size=3,
            strides=2,
            padding="same",
            name=f"{level_prefix}_upsample",
        )(x)

        skip_connection = encoder_outputs[i]

        # The Lambda layer takes a list of tensors as input to ensure dynamic shapes are handled correctly.

        x = keras.layers.Lambda(_crop_to_match, name=f"{level_prefix}_crop_to_match")(
            [x, skip_connection]
        )

        # Concatenate with the skip connection from the corresponding encoder level

        x = keras.layers.Concatenate(name=f"{level_prefix}_concat_skip")(
            [x, skip_connection]
        )

        # This 1x1 convolution projects it back to the expected number of filters

        x = keras.layers.Conv1D(
            filters=current_filters,
            kernel_size=1,
            name=f"{level_prefix}_post_concat_projection",
        )(x)

        # This is another critical location. Dropout here prevents the model from
        # simply learning to pass-through features from the skip connection without
        # properly integrating them with the high-level context from the decoder.
        x = keras.layers.Dropout(dropout_rate, name=f"{level_prefix}_dropout")(x)

        # Apply WaveNet blocks at this resolution to refine features
        for rate in dilation_rates:
            x, _ = _wavenet_residual_block(
                x,
                current_filters,
                current_filters,
                rate,
                kernel_size,
                f"{level_prefix}_{rate}",
            )

    # Applying dropout before the final refinement layers is a standard practice
    # that helps regularize the final classification/regression stage.
    x = keras.layers.Dropout(dropout_rate, name="pre_output_dropout")(x)

    x = keras.layers.Conv1D(
        filters=16, kernel_size=1, activation="gelu", name="output_conv_1"
    )(x)
    outputs = keras.layers.Conv1D(
        filters=1, kernel_size=1, activation="sigmoid", name="output_sigmoid"
    )(x)

    _model_name = "stepcovnet_ONSET"
    if model_name:
        _model_name += f"-{model_name}"

    return keras.Model(inputs=inputs, outputs=outputs, name=_model_name)


def build_arrow_model(
    num_layers: int = 1,
    d_model: int = 128,
    num_heads: int = 4,
    ff_dim: int = 512,
    dropout_rate: float = 0.0,
    model_name: str = "",
    snippet_half_frames: int = 0,
):
    """Builds a model for StepMania arrow prediction.

    Args:
        num_layers: Number of stacked encoder layers.
        d_model: The dimensionality of the model's embeddings and layers.
        num_heads: Number of attention heads.
        ff_dim: The inner dimension of the feed-forward networks.
        dropout_rate: The dropout rate used in sublayers.
        model_name: Name for the model.
        snippet_half_frames: Half-window of frames per snippet (total = 2*half+1).
            When > 0, add a second input for mel snippets per step and fuse with timing embedding via a small CNN.

    Returns:
        A Keras Model instance. Inputs accept variable sequence length (None); internally padded to constants.MAX_STEPS.
    """
    timing_input = keras.layers.Input(shape=(None, 1), name="timing_input")
    timing_embed = keras.layers.Dense(d_model, name="input_projection")(timing_input)
    timing_embed *= tf.math.sqrt(tf.cast(d_model, tf.float32), name="sqrt_d_model")

    if snippet_half_frames > 0:
        snippet_n_frames = 2 * snippet_half_frames + 1
        snippet_n_mels = constants.N_MELS
        snippet_input = keras.layers.Input(
            shape=(None, snippet_n_frames, snippet_n_mels),
            name="snippet_input",
        )
        s = SnippetCNN(
            n_frames=snippet_n_frames,
            n_mels=snippet_n_mels,
            filters=32,
            name="snippet_cnn",
        )(snippet_input)
        s = keras.layers.Dense(d_model, name="snippet_projection")(s)
        x = keras.layers.Add(name="fuse_timing_snippet")([timing_embed, s])
        inputs = [timing_input, snippet_input]
    else:
        x = timing_embed
        inputs = timing_input

    x = PositionalEncoding(position=constants.MAX_STEPS, d_model=d_model)(x)
    x = keras.layers.Dropout(dropout_rate)(x)

    for i in range(num_layers):
        x = _transformer_encoder(
            inputs=x,
            d_model=d_model,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout_rate=dropout_rate,
            name=f"transformer_block_{i}",
        )

    outputs = keras.layers.Dense(
        constants.N_ARROW_TYPES, activation="softmax", name="output_probabilities"
    )(x)

    _model_name = "stepcovnet_ARROW"
    if model_name:
        _model_name += f"-{model_name}"

    return keras.Model(inputs=inputs, outputs=outputs, name=_model_name)


def _build_arrow_mlp(
    snippet_half_frames: int,
    params: config.MLPArrowParams,
    model_name: str = "",
) -> keras.Model:
    """Build MLP-based arrow model. Same I/O contract as build_arrow_model."""
    hidden_dims = params.hidden_dims or [256, 128]
    dropout_rate = params.dropout_rate
    d = hidden_dims[0]

    timing_input = keras.layers.Input(shape=(None, 1), name="timing_input")
    timing_embed = keras.layers.Dense(d, name="input_projection")(timing_input)

    if snippet_half_frames > 0:
        snippet_n_frames = 2 * snippet_half_frames + 1
        snippet_n_mels = constants.N_MELS
        snippet_input = keras.layers.Input(
            shape=(None, snippet_n_frames, snippet_n_mels),
            name="snippet_input",
        )
        s = SnippetCNN(
            n_frames=snippet_n_frames,
            n_mels=snippet_n_mels,
            filters=32,
            name="snippet_cnn",
        )(snippet_input)
        s = keras.layers.Dense(d, name="snippet_projection")(s)
        x = keras.layers.Add(name="fuse_timing_snippet")([timing_embed, s])
        inputs = [timing_input, snippet_input]
    else:
        x = timing_embed
        inputs = timing_input

    for i, dim in enumerate(hidden_dims[1:], start=1):
        x = keras.layers.Dense(dim, activation="relu", name=f"mlp_dense_{i}")(x)
        x = keras.layers.Dropout(dropout_rate, name=f"mlp_dropout_{i}")(x)

    outputs = keras.layers.Dense(
        constants.N_ARROW_TYPES, activation="softmax", name="output_probabilities"
    )(x)

    _model_name = "stepcovnet_ARROW"
    if model_name:
        _model_name += f"-{model_name}"
    return keras.Model(inputs=inputs, outputs=outputs, name=_model_name)


def _build_arrow_lstm(
    snippet_half_frames: int,
    params: config.LSTMArrowParams,
    model_name: str = "",
) -> keras.Model:
    """Build LSTM-based arrow model. Same I/O contract as build_arrow_model."""
    units = params.units
    num_layers = params.num_layers
    dropout_rate = params.dropout_rate

    timing_input = keras.layers.Input(shape=(None, 1), name="timing_input")
    timing_embed = keras.layers.Dense(units, name="input_projection")(timing_input)

    if snippet_half_frames > 0:
        snippet_n_frames = 2 * snippet_half_frames + 1
        snippet_n_mels = constants.N_MELS
        snippet_input = keras.layers.Input(
            shape=(None, snippet_n_frames, snippet_n_mels),
            name="snippet_input",
        )
        s = SnippetCNN(
            n_frames=snippet_n_frames,
            n_mels=snippet_n_mels,
            filters=32,
            name="snippet_cnn",
        )(snippet_input)
        s = keras.layers.Dense(units, name="snippet_projection")(s)
        x = keras.layers.Add(name="fuse_timing_snippet")([timing_embed, s])
        inputs = [timing_input, snippet_input]
    else:
        x = timing_embed
        inputs = timing_input

    for i in range(num_layers):
        x = keras.layers.LSTM(
            units,
            return_sequences=True,
            dropout=dropout_rate,
            name=f"lstm_{i}",
        )(x)

    outputs = keras.layers.Dense(
        constants.N_ARROW_TYPES, activation="softmax", name="output_probabilities"
    )(x)

    _model_name = "stepcovnet_ARROW"
    if model_name:
        _model_name += f"-{model_name}"
    return keras.Model(inputs=inputs, outputs=outputs, name=_model_name)


def _build_arrow_transformer_from_config(
    model_config: config.ArrowModelConfig,
    model_name: str = "",
) -> keras.Model:
    if model_config.transformer is None:
        raise ValueError(
            "model_type is 'transformer' but transformer params are missing"
        )
    p = model_config.transformer
    return build_arrow_model(
        num_layers=p.num_layers,
        d_model=p.d_model,
        num_heads=p.num_heads,
        ff_dim=p.ff_dim,
        dropout_rate=p.dropout_rate,
        model_name=model_name,
        snippet_half_frames=model_config.snippet_half_frames,
    )


def _build_arrow_mlp_from_config(
    model_config: config.ArrowModelConfig,
    model_name: str = "",
) -> keras.Model:
    if model_config.mlp is None:
        raise ValueError("model_type is 'mlp' but mlp params are missing")
    return _build_arrow_mlp(
        snippet_half_frames=model_config.snippet_half_frames,
        params=model_config.mlp,
        model_name=model_name,
    )


def _build_arrow_lstm_from_config(
    model_config: config.ArrowModelConfig,
    model_name: str = "",
) -> keras.Model:
    if model_config.lstm is None:
        raise ValueError("model_type is 'lstm' but lstm params are missing")
    return _build_arrow_lstm(
        snippet_half_frames=model_config.snippet_half_frames,
        params=model_config.lstm,
        model_name=model_name,
    )


_ARROW_MODEL_BUILDERS: dict[
    str,
    Callable[[config.ArrowModelConfig, str], keras.Model],
] = {
    "transformer": _build_arrow_transformer_from_config,
    "mlp": _build_arrow_mlp_from_config,
    "lstm": _build_arrow_lstm_from_config,
}


def build_arrow_model_from_config(
    model_config: config.ArrowModelConfig,
    model_name: str = "",
) -> keras.Model:
    """Build an arrow model from ArrowModelConfig. Dispatches on model_type."""
    builder = _ARROW_MODEL_BUILDERS.get(model_config.model_type)
    if builder is None:
        supported = tuple(_ARROW_MODEL_BUILDERS.keys())
        raise ValueError(
            f"Unsupported arrow model_type {model_config.model_type!r}. "
            f"Supported: {supported}"
        )
    return builder(model_config, model_name)
