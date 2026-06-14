"""Model architectures and custom Keras components for onset detection and arrow classification."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable

import keras
import tensorflow as tf

from stepcovnet import config, constants


@dataclasses.dataclass
class ArrowInputOptions:
    """Options for building arrow model inputs.

    Attributes:
        snippet_half_frames: Half-window of frames per snippet (total = 2*half+1).
        use_interval: If True, add interval_input (time since previous step) and fuse with timing embedding.
        interval_encoding: IntervalEncoding for extra interval inputs (DEFAULT, LOG, or MULTI).
        use_step_index: If True, add step_index input.
        use_beat_phase: If True, add beat_phase input.
    """

    snippet_half_frames: int = 0
    use_interval: bool = False
    interval_encoding: config.IntervalEncoding = config.IntervalEncoding.DEFAULT
    use_step_index: bool = False
    use_beat_phase: bool = False


@dataclasses.dataclass
class ArrowOutputOptions:
    """Options for building arrow model outputs.

    Attributes:
        use_aux_interval: If True, add aux_interval output.
        model_name: Name for the model.
    """

    use_aux_interval: bool = False
    model_name: str = ""


@keras.saving.register_keras_serializable()
class SnippetCNN(keras.layers.Layer):
    """2D CNN over each (n_frames, n_mels) snippet without TimeDistributed.

    Reshapes (batch, steps, n_frames, n_mels) to (batch*steps, n_frames, n_mels),
    applies Conv2D layers and global average pooling, then reshapes back to
    (batch, steps, filters). Avoids TimeDistributed for XLA compatibility.

    Attributes:
        n_frames: Time dimension of each snippet.
        n_mels: Mel dimension of each snippet.
        filters: Number of output filters from the Conv2D stack.
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
    """Adds sinusoidal positional encoding to the input sequence.

    Precomputes a (1, position, d_model) encoding matrix and adds it to
    inputs of shape (batch_size, seq_len, d_model). d_model must be even.

    Attributes:
        position: Maximum sequence length for the precomputed encoding.
        d_model: Embedding dimension (must be even).
        pos_encoding: Precomputed encoding tensor, shape (1, position, d_model).
    """

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


def _build_unet_wavenet_stack(
    x: keras.KerasTensor,
    *,
    initial_filters: int,
    depth: int,
    dilation_rates: list[int],
    kernel_size: int,
    dropout_rate: float,
) -> keras.KerasTensor:
    """Run the U-Net WaveNet encoder/decoder on ``(batch, time, features)``."""
    encoder_outputs = []

    for i in range(depth):
        level_prefix = f"encoder_level_{i}"
        current_filters = initial_filters * (2**i)

        x = keras.layers.Conv1D(
            filters=current_filters, kernel_size=1, name=f"{level_prefix}_projection"
        )(x)

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

        x = keras.layers.Conv1D(
            filters=initial_filters * (2 ** (i + 1)),
            kernel_size=3,
            strides=2,
            padding="same",
            name=f"{level_prefix}_downsample",
        )(x)

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

    x = keras.layers.Dropout(dropout_rate, name=f"{bottleneck_prefix}_dropout")(x)

    for i in reversed(range(depth)):
        level_prefix = f"decoder_level_{i}"
        current_filters = initial_filters * (2**i)

        x = keras.layers.Conv1DTranspose(
            filters=current_filters,
            kernel_size=3,
            strides=2,
            padding="same",
            name=f"{level_prefix}_upsample",
        )(x)

        skip_connection = encoder_outputs[i]

        x = keras.layers.Lambda(_crop_to_match, name=f"{level_prefix}_crop_to_match")(
            [x, skip_connection]
        )

        x = keras.layers.Concatenate(name=f"{level_prefix}_concat_skip")(
            [x, skip_connection]
        )

        x = keras.layers.Conv1D(
            filters=current_filters,
            kernel_size=1,
            name=f"{level_prefix}_post_concat_projection",
        )(x)

        x = keras.layers.Dropout(dropout_rate, name=f"{level_prefix}_dropout")(x)

        for rate in dilation_rates:
            x, _ = _wavenet_residual_block(
                x,
                current_filters,
                current_filters,
                rate,
                kernel_size,
                f"{level_prefix}_{rate}",
            )

    return _build_dense_onset_output_head(x, dropout_rate=dropout_rate)


def _build_dense_onset_output_head(
    x: keras.KerasTensor,
    *,
    dropout_rate: float = 0.0,
) -> keras.KerasTensor:
    """Shared per-frame sigmoid onset probability head."""
    x = keras.layers.Dropout(dropout_rate, name="pre_output_dropout")(x)
    x = keras.layers.Conv1D(
        filters=16, kernel_size=1, activation="gelu", name="output_conv_1"
    )(x)
    return keras.layers.Conv1D(
        filters=1, kernel_size=1, activation="sigmoid", name="output_sigmoid"
    )(x)


def _onset_model_name(model_name: str) -> str:
    base = "stepcovnet_ONSET"
    if model_name:
        return f"{base}-{model_name}"
    return base


def build_tcn_onset_model(
    initial_filters: int = 16,
    dilation_rates: list[int] | None = None,
    kernel_size: int = 3,
    dropout_rate: float = 0.0,
    tcn_blocks: int = 4,
    model_name: str = "",
    input_features: int | None = None,
) -> keras.Model:
    """Temporal convolution network for dense frame onset detection (no U-Net skips)."""
    if dilation_rates is None:
        dilation_rates = [1, 2, 4, 8]

    n_features = input_features if input_features is not None else constants.N_MELS
    inputs = keras.Input(shape=(None, n_features), name="input_features")
    x = keras.layers.Conv1D(
        filters=initial_filters,
        kernel_size=1,
        name="tcn_input_projection",
    )(inputs)
    for block_idx in range(tcn_blocks):
        for rate in dilation_rates:
            x, _ = _wavenet_residual_block(
                x,
                initial_filters,
                initial_filters,
                rate,
                kernel_size,
                f"tcn_block_{block_idx}_{rate}",
            )
        x = keras.layers.Dropout(
            dropout_rate,
            name=f"tcn_block_{block_idx}_dropout",
        )(x)
    outputs = _build_dense_onset_output_head(x, dropout_rate=dropout_rate)
    return keras.Model(
        inputs=inputs,
        outputs=outputs,
        name=_onset_model_name(model_name),
    )


def build_bilstm_onset_model(
    initial_filters: int = 16,
    depth: int = 2,
    dropout_rate: float = 0.0,
    recurrent_units: int = 128,
    model_name: str = "",
    input_features: int | None = None,
) -> keras.Model:
    """BiLSTM stack over per-frame features for dense onset detection."""
    n_features = input_features if input_features is not None else constants.N_MELS
    inputs = keras.Input(shape=(None, n_features), name="input_features")
    x = keras.layers.Conv1D(
        filters=initial_filters,
        kernel_size=1,
        name="bilstm_input_projection",
    )(inputs)
    for layer_idx in range(depth):
        x = keras.layers.Bidirectional(
            keras.layers.LSTM(
                recurrent_units,
                return_sequences=True,
                name=f"bilstm_{layer_idx}",
            ),
            name=f"bilstm_bidir_{layer_idx}",
        )(x)
        x = keras.layers.Dropout(
            dropout_rate,
            name=f"bilstm_dropout_{layer_idx}",
        )(x)
    outputs = _build_dense_onset_output_head(x, dropout_rate=dropout_rate)
    return keras.Model(
        inputs=inputs,
        outputs=outputs,
        name=_onset_model_name(model_name),
    )


def build_transformer_onset_model(
    initial_filters: int = 64,
    transformer_layers: int = 2,
    transformer_heads: int = 4,
    dropout_rate: float = 0.0,
    model_name: str = "",
    input_features: int | None = None,
) -> keras.Model:
    """Lightweight transformer encoder over per-frame SSL features."""
    n_features = input_features if input_features is not None else constants.N_MELS
    inputs = keras.Input(shape=(None, n_features), name="input_features")
    x = keras.layers.Conv1D(
        filters=initial_filters,
        kernel_size=1,
        name="transformer_input_projection",
    )(inputs)
    ff_dim = max(initial_filters * 4, 128)
    for layer_idx in range(transformer_layers):
        x = _transformer_encoder(
            x,
            d_model=initial_filters,
            num_heads=transformer_heads,
            ff_dim=ff_dim,
            dropout_rate=dropout_rate,
            name=f"onset_transformer_{layer_idx}",
        )
    outputs = _build_dense_onset_output_head(x, dropout_rate=dropout_rate)
    return keras.Model(
        inputs=inputs,
        outputs=outputs,
        name=_onset_model_name(model_name),
    )


def build_onset_dense_model(
    model_config: config.OnsetModelConfig,
    *,
    model_name: str = "",
    input_features: int | None = None,
) -> keras.Model:
    """Build a dense frame onset model from ``OnsetModelConfig``."""
    arch = model_config.onset_architecture
    n_features = (
        input_features if input_features is not None else model_config.input_features
    )
    if arch == config.OnsetArchitecture.TCN:
        return build_tcn_onset_model(
            initial_filters=model_config.initial_filters,
            dilation_rates=model_config.dilation_rates,
            kernel_size=model_config.kernel_size,
            dropout_rate=model_config.dropout_rate,
            tcn_blocks=model_config.tcn_blocks,
            model_name=model_name,
            input_features=n_features,
        )
    if arch == config.OnsetArchitecture.BILSTM:
        return build_bilstm_onset_model(
            initial_filters=model_config.initial_filters,
            depth=model_config.depth,
            dropout_rate=model_config.dropout_rate,
            recurrent_units=model_config.recurrent_units,
            model_name=model_name,
            input_features=n_features,
        )
    if arch == config.OnsetArchitecture.TRANSFORMER:
        return build_transformer_onset_model(
            initial_filters=model_config.initial_filters,
            transformer_layers=model_config.transformer_layers,
            transformer_heads=model_config.transformer_heads,
            dropout_rate=model_config.dropout_rate,
            model_name=model_name,
            input_features=n_features,
        )
    return build_unet_wavenet_model(
        initial_filters=model_config.initial_filters,
        depth=model_config.depth,
        dilation_rates=model_config.dilation_rates,
        kernel_size=model_config.kernel_size,
        dropout_rate=model_config.dropout_rate,
        model_name=model_name,
        input_features=n_features,
    )


def build_unet_wavenet_model(
    initial_filters: int = 16,
    depth: int = 2,
    dilation_rates: list[int] | None = None,
    kernel_size: int = 3,
    dropout_rate: float = 0.0,
    model_name: str = "",
    input_features: int | None = None,
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
        input_features: Number of input feature channels per time step. Defaults to N_MELS.

    Returns:
        A Keras Model instance.
    """
    if dilation_rates is None:
        dilation_rates = [1, 2, 4, 8]

    n_features = input_features if input_features is not None else constants.N_MELS
    inputs = keras.Input(shape=(None, n_features), name="input_features")
    outputs = _build_unet_wavenet_stack(
        inputs,
        initial_filters=initial_filters,
        depth=depth,
        dilation_rates=dilation_rates,
        kernel_size=kernel_size,
        dropout_rate=dropout_rate,
    )

    _model_name = "stepcovnet_ONSET"
    if model_name:
        _model_name += f"-{model_name}"

    return keras.Model(inputs=inputs, outputs=outputs, name=_model_name)


def build_unet_wavenet_from_waveform_model(
    initial_filters: int = 16,
    depth: int = 2,
    dilation_rates: list[int] | None = None,
    kernel_size: int = 3,
    dropout_rate: float = 0.0,
    model_name: str = "",
    frontend_filters: int = 32,
) -> keras.Model:
    """Build a dense onset model: mono waveform in, frame-wise onset probs out.

    A learned strided Conv1D frontend maps the waveform to one embedding vector
    per ``constants.WAVEFORM_SAMPLES_PER_FRAME`` hop, aligned with the mel-based
    dense onset grid. The embeddings feed the standard U-Net WaveNet stack.

    Args:
        initial_filters: U-Net initial filter width.
        depth: U-Net depth.
        dilation_rates: WaveNet dilation rates per level.
        kernel_size: Convolution kernel size.
        dropout_rate: Dropout rate.
        model_name: Suffix for the Keras model name.
        frontend_filters: Output channels from the waveform frontend.

    Returns:
        Keras model with input ``waveform`` ``(batch, samples)`` and output
        ``(batch, time, 1)`` onset probabilities.
    """
    if dilation_rates is None:
        dilation_rates = [1, 2, 4, 8]

    hop = constants.WAVEFORM_SAMPLES_PER_FRAME
    waveform_input = keras.Input(shape=(None,), name="waveform", dtype=tf.float32)
    x = keras.layers.Reshape((-1, 1), name="waveform_channel")(waveform_input)
    x = keras.layers.Conv1D(
        frontend_filters,
        kernel_size=hop,
        strides=hop,
        padding="valid",
        activation="gelu",
        name="waveform_frame_embed",
    )(x)
    x = keras.layers.Conv1D(
        frontend_filters,
        kernel_size=3,
        strides=1,
        padding="same",
        activation="gelu",
        name="waveform_frame_refine",
    )(x)
    outputs = _build_unet_wavenet_stack(
        x,
        initial_filters=initial_filters,
        depth=depth,
        dilation_rates=dilation_rates,
        kernel_size=kernel_size,
        dropout_rate=dropout_rate,
    )

    _model_name = "stepcovnet_ONSET"
    if model_name:
        _model_name += f"-{model_name}"

    return keras.Model(inputs=waveform_input, outputs=outputs, name=_model_name)


def _build_arrow_inputs(
    embed_dim: int,
    input_options: ArrowInputOptions,
    scale_timing: bool = False,
) -> tuple:
    """Build shared arrow model inputs and fused embedding tensor.

    Args:
        embed_dim: Dimension for timing, interval, and snippet projections.
        input_options: Options for which inputs to add (snippets, interval, step_index, beat_phase, etc.).
        scale_timing: If True, scale timing embedding by sqrt(embed_dim) (transformer convention).

    Returns:
        (inputs, x, timing_tensor): inputs is a single Input or list of Inputs;
        x is the fused tensor of shape (batch, steps, embed_dim); timing_tensor
        is the raw timing input tensor for use e.g. in timing-based position encoding.
    """

    o = input_options
    use_interval = o.use_interval
    interval_encoding = o.interval_encoding
    use_step_index = o.use_step_index
    use_beat_phase = o.use_beat_phase
    snippet_half_frames = o.snippet_half_frames

    timing_input = keras.layers.Input(shape=(None, 1), name="timing_input")
    timing_embed = keras.layers.Dense(embed_dim, name="input_projection")(timing_input)
    if scale_timing:
        timing_embed *= tf.math.sqrt(
            tf.cast(embed_dim, tf.float32), name="sqrt_d_model"
        )
    x = timing_embed

    inputs_list: list = [timing_input]
    if use_interval:
        if interval_encoding == config.IntervalEncoding.DEFAULT:
            interval_input = keras.layers.Input(shape=(None, 1), name="interval_input")
            interval_embed = keras.layers.Dense(embed_dim, name="interval_projection")(
                interval_input
            )
            x = keras.layers.Add(name="fuse_timing_interval")([x, interval_embed])
            inputs_list.append(interval_input)
        elif interval_encoding == config.IntervalEncoding.LOG:
            interval_log_input = keras.layers.Input(
                shape=(None, 1), name="interval_log_input"
            )
            interval_log_embed = keras.layers.Dense(
                embed_dim, name="interval_log_projection"
            )(interval_log_input)
            x = keras.layers.Add(name="fuse_interval_log")([x, interval_log_embed])
            inputs_list.append(interval_log_input)
        elif interval_encoding == config.IntervalEncoding.MULTI:
            interval_log_input = keras.layers.Input(
                shape=(None, 1), name="interval_log_input"
            )
            interval_log_embed = keras.layers.Dense(
                embed_dim, name="interval_log_projection"
            )(interval_log_input)
            x = keras.layers.Add(name="fuse_interval_log")([x, interval_log_embed])
            inputs_list.append(interval_log_input)
            interval_next_input = keras.layers.Input(
                shape=(None, 1), name="interval_next_input"
            )
            interval_next_embed = keras.layers.Dense(
                embed_dim, name="interval_next_projection"
            )(interval_next_input)
            x = keras.layers.Add(name="fuse_interval_next")([x, interval_next_embed])
            inputs_list.append(interval_next_input)
        else:
            raise ValueError(f"Invalid interval encoding: {interval_encoding}")

    if use_step_index:
        step_index_input = keras.layers.Input(shape=(None, 1), name="step_index_input")
        step_index_embed = keras.layers.Dense(embed_dim, name="step_index_projection")(
            step_index_input
        )
        x = keras.layers.Add(name="fuse_step_index")([x, step_index_embed])
        inputs_list.append(step_index_input)

    if use_beat_phase:
        beat_phase_input = keras.layers.Input(shape=(None, 1), name="beat_phase_input")
        beat_phase_embed = keras.layers.Dense(embed_dim, name="beat_phase_projection")(
            beat_phase_input
        )
        x = keras.layers.Add(name="fuse_beat_phase")([x, beat_phase_embed])
        inputs_list.append(beat_phase_input)

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
        s = keras.layers.Dense(embed_dim, name="snippet_projection")(s)
        x = keras.layers.Add(name="fuse_timing_snippet")([x, s])
        inputs_list.append(snippet_input)

    inputs = inputs_list[0] if len(inputs_list) == 1 else inputs_list
    timing_tensor = inputs_list[0]
    return inputs, x, timing_tensor


def _wrap_arrow_output(
    inputs,
    x: keras.KerasTensor,
    output_options: ArrowOutputOptions,
) -> keras.Model:
    """Apply arrow classification head and wrap inputs + outputs in a Keras Model.

    Args:
        inputs: Model input(s) from _build_arrow_inputs.
        x: Fused feature tensor (batch, steps, embed_dim).
        output_options: Options for model name and whether to add aux_interval output.

    Returns:
        Keras Model with softmax output over N_ARROW_TYPES. When output_options.use_aux_interval
        is True, the model has a dict output with keys 'output_probabilities' and
        'aux_interval', which aligns with the loss/metrics/sample_weight dicts used
        during compilation and training.
    """
    o = output_options
    use_aux_interval = o.use_aux_interval
    model_name = o.model_name

    arrow_logits = keras.layers.Dense(
        constants.N_ARROW_TYPES, activation="softmax", name="output_probabilities"
    )(x)
    if use_aux_interval:
        aux_interval = keras.layers.Dense(1, name="aux_interval")(x)
        outputs = {
            "output_probabilities": arrow_logits,
            "aux_interval": aux_interval,
        }
    else:
        outputs = arrow_logits
    name = "stepcovnet_ARROW"
    if model_name:
        name += f"-{model_name}"
    return keras.Model(inputs=inputs, outputs=outputs, name=name)


def build_arrow_model(
    input_options: ArrowInputOptions,
    output_options: ArrowOutputOptions,
    params: config.TransformerArrowParams,
):
    """Build a transformer-based model for StepMania arrow prediction.

    Args:
        input_options: Options for building inputs (snippets, interval, step_index, beat_phase, etc.).
        output_options: Options for model name and optional aux_interval output.
        params: Transformer architecture parameters (layers, d_model, heads, ff_dim, dropout, etc.).

    Returns:
        A Keras Model instance. Inputs accept variable sequence length (None); internally padded to constants.MAX_STEPS.
    """
    p = params
    d_model = p.d_model
    num_layers = p.num_layers
    num_heads = p.num_heads
    ff_dim = p.ff_dim
    dropout_rate = p.dropout_rate
    use_timing_position = p.use_timing_position

    inputs, x, timing_tensor = _build_arrow_inputs(
        embed_dim=d_model,
        input_options=input_options,
        scale_timing=True,
    )

    if use_timing_position:
        timing_pos_bias = keras.layers.Dense(d_model, name="timing_position_bias")(
            timing_tensor
        )
        x = keras.layers.Add(name="add_timing_position")([x, timing_pos_bias])
    else:
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

    return _wrap_arrow_output(inputs, x, output_options=output_options)


def _build_arrow_mlp(
    input_options: ArrowInputOptions,
    output_options: ArrowOutputOptions,
    params: config.MLPArrowParams,
) -> keras.Model:
    """Build MLP-based arrow model. Same I/O contract as build_arrow_model."""
    hidden_dims = params.hidden_dims or [256, 128]
    dropout_rate = params.dropout_rate
    d = hidden_dims[0]

    inputs, x, _ = _build_arrow_inputs(
        embed_dim=d,
        input_options=input_options,
        scale_timing=False,
    )

    for i, dim in enumerate(hidden_dims[1:], start=1):
        x = keras.layers.Dense(dim, activation="relu", name=f"mlp_dense_{i}")(x)
        x = keras.layers.Dropout(dropout_rate, name=f"mlp_dropout_{i}")(x)

    return _wrap_arrow_output(inputs, x, output_options=output_options)


def _build_arrow_lstm(
    input_options: ArrowInputOptions,
    output_options: ArrowOutputOptions,
    params: config.LSTMArrowParams,
) -> keras.Model:
    """Build LSTM-based arrow model. Same I/O contract as build_arrow_model."""
    units = params.units
    num_layers = params.num_layers
    dropout_rate = params.dropout_rate

    inputs, x, _ = _build_arrow_inputs(
        embed_dim=units,
        input_options=input_options,
        scale_timing=False,
    )

    for i in range(num_layers):
        lstm_layer = keras.layers.LSTM(
            units,
            return_sequences=True,
            dropout=dropout_rate,
            name=f"lstm_{i}",
        )
        if params.bidirectional:
            x = keras.layers.Bidirectional(lstm_layer, name=f"bidirectional_{i}")(x)
        else:
            x = lstm_layer(x)

    # Classification head: dense + dropout before softmax (adds capacity and regularization)
    x = keras.layers.Dense(
        max(units // 2, constants.N_ARROW_TYPES),
        activation="relu",
        name="head_dense",
    )(x)
    x = keras.layers.Dropout(dropout_rate, name="head_dropout")(x)
    return _wrap_arrow_output(inputs, x, output_options=output_options)


def _build_arrow_gru(
    input_options: ArrowInputOptions,
    output_options: ArrowOutputOptions,
    params: config.GRUArrowParams,
) -> keras.Model:
    """Build GRU-based arrow model. Same I/O contract as build_arrow_model.

    When params.add_attention_layer is True, adds one multi-head self-attention
    layer after the GRU stack (no positional encoding). Optional aux_interval
    output is controlled by output_options.use_aux_interval.
    """
    units = params.units
    num_layers = params.num_layers
    dropout_rate = params.dropout_rate

    inputs, x, _ = _build_arrow_inputs(
        embed_dim=units,
        input_options=input_options,
        scale_timing=False,
    )

    for i in range(num_layers):
        gru_layer = keras.layers.GRU(
            units,
            return_sequences=True,
            dropout=dropout_rate,
            name=f"gru_{i}",
        )
        if params.bidirectional:
            x = keras.layers.Bidirectional(gru_layer, name=f"bidirectional_{i}")(x)
        else:
            x = gru_layer(x)

    if params.add_attention_layer:
        attn_heads = params.attention_heads
        attn_dim = params.attention_dim
        if units != attn_dim:
            x = keras.layers.Dense(attn_dim, name="gru_attn_projection")(x)
        kv_dim = attn_dim // attn_heads
        if kv_dim * attn_heads != attn_dim:
            kv_dim = max(1, attn_dim // attn_heads)
        x = keras.layers.MultiHeadAttention(
            num_heads=attn_heads,
            key_dim=kv_dim,
            value_dim=kv_dim,
            name="gru_attn_mha",
        )(x, x)
        x = keras.layers.Dropout(dropout_rate, name="gru_attn_dropout")(x)
        head_dim = attn_dim
    else:
        head_dim = units

    # Classification head: dense + dropout before softmax (adds capacity and regularization)
    x = keras.layers.Dense(
        max(head_dim // 2, constants.N_ARROW_TYPES),
        activation="relu",
        name="head_dense",
    )(x)
    x = keras.layers.Dropout(dropout_rate, name="head_dropout")(x)
    return _wrap_arrow_output(inputs, x, output_options=output_options)


def _build_arrow_tcn(
    params: config.TCNArrowParams,
    input_options: ArrowInputOptions,
    output_options: ArrowOutputOptions,
) -> keras.Model:
    """Build TCN-based arrow model with causal dilated Conv1D stack.

    Same I/O contract as build_arrow_model. Uses dilation_base^layer_idx per layer.
    """
    filters = params.filters
    kernel_size = params.kernel_size
    num_layers = params.num_layers
    dilation_base = params.dilation_base
    dropout_rate = params.dropout_rate

    inputs, x, _ = _build_arrow_inputs(
        embed_dim=filters,
        input_options=input_options,
        scale_timing=False,
    )

    for i in range(num_layers):
        dilation = dilation_base**i
        x = keras.layers.Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            padding="causal",
            dilation_rate=dilation,
            activation="relu",
            name=f"tcn_conv_{i}_d{dilation}",
        )(x)
        x = keras.layers.Dropout(dropout_rate, name=f"tcn_dropout_{i}")(x)

    x = keras.layers.Dense(
        max(filters // 2, constants.N_ARROW_TYPES),
        activation="relu",
        name="head_dense",
    )(x)
    x = keras.layers.Dropout(dropout_rate, name="head_dropout")(x)
    return _wrap_arrow_output(inputs, x, output_options=output_options)


def _build_arrow_cnn1d(
    input_options: ArrowInputOptions,
    params: config.CNN1DArrowParams,
    output_options: ArrowOutputOptions,
) -> keras.Model:
    """Build 1D CNN-based arrow model with causal Conv1D stack.

    Same I/O contract as build_arrow_model.
    """
    filters = params.filters
    kernel_sizes = params.kernel_sizes or [3, 3, 3]
    dropout_rate = params.dropout_rate

    inputs, x, _ = _build_arrow_inputs(
        embed_dim=filters,
        scale_timing=False,
        input_options=input_options,
    )

    for i, k in enumerate(kernel_sizes):
        x = keras.layers.Conv1D(
            filters=filters,
            kernel_size=k,
            padding="causal",
            activation="relu",
            name=f"cnn1d_conv_{i}",
        )(x)
    x = keras.layers.Dropout(dropout_rate, name="cnn1d_dropout")(x)

    x = keras.layers.Dense(
        max(filters // 2, constants.N_ARROW_TYPES),
        activation="relu",
        name="head_dense",
    )(x)
    x = keras.layers.Dropout(dropout_rate, name="head_dropout")(x)
    return _wrap_arrow_output(inputs, x, output_options=output_options)


_ARROW_MODEL_BUILDERS: dict[
    str,
    Callable[..., keras.Model],
] = {
    "transformer": build_arrow_model,
    "mlp": _build_arrow_mlp,
    "lstm": _build_arrow_lstm,
    "gru": _build_arrow_gru,
    "tcn": _build_arrow_tcn,
    "cnn1d": _build_arrow_cnn1d,
}


def build_arrow_model_from_config(
    model_config: config.ArrowModelConfig,
    input_options: ArrowInputOptions,
    output_options: ArrowOutputOptions,
) -> keras.Model:
    """Build an arrow model from ArrowModelConfig. Dispatches on model_type.

    Args:
        model_config: Arrow model config specifying model_type and the active params block.
        input_options: Options for building inputs (snippets, interval, step_index, beat_phase, etc.).
        output_options: Options for model name and optional aux_interval output.

    Returns:
        Keras Model. When output_options.use_aux_interval is True, the model output
        is a dict with keys 'output_probabilities' and 'aux_interval'.
    """
    builder = _ARROW_MODEL_BUILDERS.get(model_config.model_type)
    if builder is None:
        supported = ", ".join(sorted(_ARROW_MODEL_BUILDERS.keys()))
        raise ValueError(
            f"Unsupported arrow model_type {model_config.model_type!r}. "
            f"Supported: {supported}"
        )
    params = model_config.get_active_params_block()
    if params is None:
        raise ValueError(
            f"Arrow model_type {model_config.model_type!r} has no params block set"
        )
    return builder(
        input_options=input_options,
        output_options=output_options,
        params=params,
    )
