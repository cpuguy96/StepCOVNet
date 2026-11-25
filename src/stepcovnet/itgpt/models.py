"""Hierarchical transformer placement from ITGPT (`omalley2026itgpt`).

Layer graph follows ``OnsetModel`` in
https://github.com/miguelomalley/ITGPT/blob/main/onset.py
(CNN beat encoder, five hierarchical attention levels, global encoder).
"""

from __future__ import annotations

from typing import Any

import keras
from keras import ops

from stepcovnet.ddcl import constants as ddcl_constants
from stepcovnet.itgpt import constants


def grid_importance_weights(
    n_slots: int = constants.N_SLOTS,
    *,
    weight_24th: float = constants.GRID_WEIGHT_24TH,
) -> Any:
    """Return the ITGPT 48-slot BCE weights.

    Args:
        n_slots: Slot count (paper: 48).
        weight_24th: Loss weight for 24th/triplet slots (paper: 1.0).

    Returns:
        Vector of length ``n_slots``.
    """
    weights = [constants.GRID_WEIGHT_MICRO] * n_slots
    for index in constants.INDICES_16TH:
        weights[index] = constants.GRID_WEIGHT_16TH
    for index in constants.INDICES_24TH:
        weights[index] = float(weight_24th)
    for index in constants.INDICES_32ND:
        weights[index] = 1.0
    return ops.convert_to_tensor(weights, dtype="float32")


class ItgptRmsNorm(keras.layers.Layer):
    """RMSNorm as in ITGPT ``nn.RMSNorm``.

    Args:
        epsilon: Stabilizer.
        **kwargs: Keras layer kwargs.
    """

    def __init__(self, epsilon: float = 1e-6, **kwargs) -> None:
        super().__init__(**kwargs)
        self.epsilon = epsilon

    def build(self, input_shape: object) -> None:
        """Create the scale vector.

        Args:
            input_shape: Incoming shape.
        """
        self.scale = self.add_weight(
            name="scale",
            shape=(int(input_shape[-1]),),  # type: ignore[index]
            initializer="ones",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs: Any) -> Any:
        """Normalize last dimension.

        Args:
            inputs: Activation tensor.

        Returns:
            Normalized tensor.
        """
        rms = ops.sqrt(
            ops.mean(ops.square(inputs), axis=-1, keepdims=True) + self.epsilon  # type: ignore[operator]
        )
        return inputs / rms * self.scale


class ItgptFFN(keras.layers.Layer):
    """GELU MLP with residual dropout (ITGPT ``FFN``).

    Args:
        d_model: Model width.
        dropout: Dropout rate.
        **kwargs: Keras layer kwargs.
    """

    def __init__(self, d_model: int, dropout: float, **kwargs) -> None:
        super().__init__(**kwargs)
        hidden = 4 * d_model
        self.dense1 = keras.layers.Dense(hidden, activation="gelu")
        self.drop1 = keras.layers.Dropout(dropout)
        self.dense2 = keras.layers.Dense(d_model)
        self.drop2 = keras.layers.Dropout(dropout)

    def call(self, inputs: Any, training: bool | None = None) -> Any:
        """Apply the MLP.

        Args:
            inputs: ``(..., d_model)``.
            training: Keras training flag.

        Returns:
            Same shape as ``inputs``.
        """
        hidden = self.drop1(self.dense1(inputs), training=training)
        return self.drop2(self.dense2(hidden), training=training)


class ItgptEncoderBlock(keras.layers.Layer):
    """Pre-norm bidirectional self-attention block (ITGPT ``EncoderBlock``).

    Args:
        d_model: Model width.
        n_heads: Attention heads.
        dropout: Dropout rate.
        **kwargs: Keras layer kwargs.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float, **kwargs) -> None:
        super().__init__(**kwargs)
        self.ln1 = ItgptRmsNorm()
        self.attn = keras.layers.MultiHeadAttention(
            num_heads=n_heads,
            key_dim=d_model // n_heads,
            dropout=dropout,
        )
        self.ln2 = ItgptRmsNorm()
        self.ffn = ItgptFFN(d_model, dropout)

    def call(self, inputs: Any, training: bool | None = None) -> Any:
        """Apply pre-norm attention + FFN.

        Args:
            inputs: ``(batch, time, d_model)``.
            training: Keras training flag.

        Returns:
            Same shape as ``inputs``.
        """
        normed = self.ln1(inputs)
        attended = self.attn(normed, normed, training=training)
        hidden = inputs + attended
        return hidden + self.ffn(self.ln2(hidden), training=training)


class ItgptNormlessEncoderBlock(keras.layers.Layer):
    """ITGPT ``NormlessEncoderBlock`` (attention then RMSNorm FFN).

    Args:
        d_model: Model width.
        n_heads: Attention heads.
        dropout: Dropout rate.
        **kwargs: Keras layer kwargs.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float, **kwargs) -> None:
        super().__init__(**kwargs)
        self.attn = keras.layers.MultiHeadAttention(
            num_heads=n_heads,
            key_dim=d_model // n_heads,
            dropout=dropout,
        )
        self.ffn = ItgptFFN(d_model, dropout)
        self.norm = ItgptRmsNorm()

    def call(self, inputs: Any, training: bool | None = None) -> Any:
        """Apply attention then FFN.

        Args:
            inputs: ``(batch, time, d_model)``.
            training: Keras training flag.

        Returns:
            Same shape as ``inputs``.
        """
        attended = inputs + self.attn(inputs, inputs, training=training)
        return attended + self.ffn(self.norm(attended), training=training)


class ItgptGatedCompression(keras.layers.Layer):
    """ITGPT ``GatedCompression`` (value × sigmoid gate).

    Args:
        out_features: Compressed width.
        **kwargs: Keras layer kwargs.
    """

    def __init__(self, out_features: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.out_features = out_features
        self.proj = keras.layers.Dense(out_features * 2)
        self.norm = ItgptRmsNorm()

    def call(self, inputs: Any) -> Any:
        """Compress the last axis.

        Args:
            inputs: ``(..., in_features)``.

        Returns:
            ``(..., out_features)``.
        """
        projected = self.proj(inputs)
        value, gate = ops.split(projected, 2, axis=-1)  # type: ignore[misc]
        return self.norm(value * ops.sigmoid(gate))


class ItgptHierarchicalAttnBlock(keras.layers.Layer):
    """ITGPT ``HierarchicalAttnBlock`` over ``beats_per_chunk`` groups.

    Args:
        d_model: Model width.
        n_heads: Attention heads.
        dropout: Dropout rate.
        frames_per_beat: CNN frames kept per beat (24 or 6).
        beats_per_chunk: Beats attending together.
        max_chunks: Chunk positional embedding table size.
        **kwargs: Keras layer kwargs.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float,
        *,
        frames_per_beat: int,
        beats_per_chunk: int,
        max_chunks: int = 2048,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.d_model = d_model
        self.frames_per_beat = frames_per_beat
        self.beats_per_chunk = beats_per_chunk
        self._total_frames = beats_per_chunk * frames_per_beat
        self.chunk_pos_emb = keras.layers.Embedding(
            max_chunks,
            d_model,
            embeddings_initializer=keras.initializers.TruncatedNormal(stddev=0.02),  # type: ignore[arg-type]
        )
        self.attn_block = ItgptNormlessEncoderBlock(d_model, n_heads, dropout)
        self.norm1 = ItgptRmsNorm()
        self.norm2 = ItgptRmsNorm()

    def build(self, input_shape: object) -> None:
        """Create the intra-chunk position table.

        Args:
            input_shape: Incoming ``(B*T, frames, d)`` shape.
        """
        self.pos_emb = self.add_weight(
            name="pos_emb",
            shape=(1, self._total_frames, self.d_model),
            initializer=keras.initializers.RandomNormal(stddev=0.02),
            trainable=True,
        )
        super().build(input_shape)

    def call(
        self,
        h_beat: Any,
        *,
        batch_size: Any,
        n_beats: Any,
        training: bool | None = None,
    ) -> Any:
        """Chunk frames and apply intra-chunk attention.

        Diagnostic 48-slot aux heads from upstream are omitted:
        ``--lambda_diag`` defaults to 0.

        Args:
            h_beat: ``(B * T, frames_per_beat, d_model)``.
            batch_size: Batch size ``B``.
            n_beats: Padded beat count ``T`` (multiple of ``beats_per_chunk``).
            training: Keras training flag.

        Returns:
            ``h_out`` with shape ``(B*T, F, d)``.
        """
        hidden = self.norm1(h_beat)
        chunk = self.beats_per_chunk
        frames = self.frames_per_beat
        num_chunks = n_beats // chunk  # type: ignore[operator]
        n_chunk_rows = ops.multiply(batch_size, num_chunks)
        n_beat_rows = ops.multiply(batch_size, n_beats)
        hidden = ops.reshape(
            hidden,
            (batch_size, num_chunks, chunk * frames, self.d_model),
        )
        hidden = ops.reshape(
            hidden,
            (n_chunk_rows, chunk * frames, self.d_model),
        )
        hidden = hidden + self.pos_emb
        chunk_idx = ops.arange(num_chunks, dtype="int32")
        chunk_emb = self.chunk_pos_emb(chunk_idx)
        chunk_emb = ops.repeat(chunk_emb, batch_size, axis=0)
        hidden = hidden + ops.expand_dims(chunk_emb, axis=1)
        hidden = self.attn_block(hidden, training=training)
        hidden = ops.reshape(
            hidden,
            (batch_size, num_chunks, chunk, frames, self.d_model),
        )
        hidden = ops.reshape(
            hidden,
            (n_beat_rows, frames, self.d_model),
        )
        return self.norm2(hidden)


class ItgptBeatCNNEncoder(keras.layers.Layer):
    """ITGPT ``BeatCNNEncoder`` on ``(32, 80, 3)`` log-mel beats.

    Args:
        d_model: Projected width.
        hidden: Base convolution channels.
        **kwargs: Keras layer kwargs.
    """

    def __init__(
        self, d_model: int, hidden: int = constants.CNN_HIDDEN, **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.hidden = hidden
        self.conv0 = keras.layers.Conv2D(
            hidden, (7, 3), padding="valid", activation="gelu"
        )
        self.pad1 = keras.layers.ZeroPadding2D(((1, 1), (0, 0)))
        self.conv1 = keras.layers.Conv2D(
            hidden, (3, 3), strides=(1, 3), padding="valid", activation="gelu"
        )
        self.conv2 = keras.layers.Conv2D(
            hidden * 2, (3, 3), padding="valid", activation="gelu"
        )
        self.pad3 = keras.layers.ZeroPadding2D(((1, 1), (0, 0)))
        self.conv3 = keras.layers.Conv2D(
            hidden * 2, (3, 3), strides=(1, 3), padding="valid", activation="gelu"
        )
        self.proj0 = keras.layers.Dense(hidden * 16, activation="gelu")
        self.proj1 = keras.layers.Dense(d_model)

    def call(self, inputs: Any) -> Any:
        """Encode each beat spectrogram.

        Args:
            inputs: ``(batch * beats, 32, 80, 3)``.

        Returns:
            ``(batch * beats, 24, d_model)``.
        """
        hidden = self.conv0(inputs)
        hidden = self.conv1(self.pad1(hidden))
        hidden = self.conv2(hidden)
        hidden = self.conv3(self.pad3(hidden))
        # (BT, 24, 8, hidden*2) -> (BT, 24, hidden*16)
        hidden = ops.reshape(
            hidden,
            (-1, constants.CNN_FRAMES_OUT, self.hidden * 16),
        )
        return self.proj1(self.proj0(hidden))


class ItgptBeatMertEncoder(keras.layers.Layer):
    """Beat-aligned MERT encoder producing ``(batch * beats, 24, d_model)``.

    Replaces :class:`ItgptBeatCNNEncoder` when ``audio_feature_type == "mert"``
    so pretrained MERT beat frames plug directly into ITGPT's hierarchical
    attention stack.

    Args:
        d_model: Projected width.
        dropout_rate: Dropout after the first linear projection.
        **kwargs: Keras layer kwargs.
    """

    def __init__(
        self,
        d_model: int,
        dropout_rate: float = constants.DROPOUT_RATE,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.d_model = d_model
        self.dropout_rate = dropout_rate
        self.norm = ItgptRmsNorm()
        self.proj0 = keras.layers.Dense(d_model, activation="gelu")
        self.drop = keras.layers.Dropout(dropout_rate)
        self.conv = keras.layers.Conv1D(
            d_model, kernel_size=3, padding="same", activation="gelu"
        )
        self.proj1 = keras.layers.Dense(d_model)

    def call(self, inputs: Any, training: bool | None = None) -> Any:
        """Encode each beat's MERT frame sequence.

        Args:
            inputs: ``(batch * beats, n_frames_per_beat, feature_dim)``.
            training: Keras training flag.

        Returns:
            ``(batch * beats, n_frames_per_beat, d_model)``.
        """
        hidden = self.drop(self.proj0(self.norm(inputs)), training=training)
        hidden = self.conv(hidden)
        return self.proj1(hidden)


class ItgptConvHead(keras.layers.Layer):
    """ITGPT ``ConvHead`` residual depthwise/pointwise conv over beats.

    Args:
        d_model: Channel width.
        dropout: Dropout after the pointwise conv.
        **kwargs: Keras layer kwargs.
    """

    def __init__(self, d_model: int, dropout: float, **kwargs) -> None:
        super().__init__(**kwargs)
        self.norm = ItgptRmsNorm()
        self.dw = keras.layers.Conv1D(
            d_model,
            kernel_size=3,
            padding="same",
            groups=d_model,
            activation="gelu",
        )
        self.pw = keras.layers.Conv1D(d_model, kernel_size=1)
        self.drop = keras.layers.Dropout(dropout)

    def call(self, inputs: Any, training: bool | None = None) -> Any:
        """Add a local conv residual.

        Args:
            inputs: ``(batch, beats, d_model)``.
            training: Keras training flag.

        Returns:
            Same shape as ``inputs``.
        """
        hidden = self.norm(inputs)
        hidden = self.drop(self.pw(self.dw(hidden)), training=training)
        return inputs + hidden


class ItgptConditionProj(keras.layers.Layer):
    """Min-max normalize a scalar and map to a token (ITGPT BPM/diff MLPs).

    Args:
        d_model: Token width.
        min_val: Lower clip bound.
        max_val: Upper clip bound.
        **kwargs: Keras layer kwargs.
    """

    def __init__(self, d_model: int, min_val: float, max_val: float, **kwargs) -> None:
        super().__init__(**kwargs)
        self.min_val = min_val
        self.max_val = max_val
        self.dense0 = keras.layers.Dense(d_model // 2, activation="gelu")
        self.dense1 = keras.layers.Dense(d_model)

    def call(self, inputs: Any) -> Any:
        """Project ``(batch, 1)`` to ``(batch, 1, d_model)``.

        Args:
            inputs: Scalar condition per chart.

        Returns:
            Token sequence of length 1.
        """
        scaled = ops.clip(
            (inputs - self.min_val) / (self.max_val - self.min_val),
            0.0,
            1.0,
        )
        return ops.expand_dims(self.dense1(self.dense0(scaled)), axis=1)


class ItgptConditionFilm(keras.layers.Layer):
    """Zero-initialized FiLM modulation ``(1 + gamma) * h + beta`` from condition tokens.

    Starts as an exact identity transform at initialization so hierarchical
    audio features are undisturbed at step 0 and smoothly adapt to target
    difficulty and BPM during training.

    Args:
        d_model: Feature channel width to modulate.
        **kwargs: Keras layer kwargs.
    """

    def __init__(self, d_model: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.d_model = d_model
        self.hidden = keras.layers.Dense(d_model, activation="gelu")
        self.gamma_beta = keras.layers.Dense(
            d_model * 2,
            kernel_initializer="zeros",
            bias_initializer="zeros",
        )

    def call(
        self,
        features: Any,
        cond_vec: Any,
        *,
        batch_size: Any,
        n_beats: Any,
        frames_per_beat: int | None = None,
    ) -> Any:
        """Modulate beat or sub-beat features with chart-level FiLM parameters.

        Args:
            features: Either ``(B * T, F, d_model)`` when ``frames_per_beat``
                is set, or ``(B, T, d_model)`` when ``frames_per_beat`` is None.
            cond_vec: Condition embedding ``(B, cond_dim)``.
            batch_size: Batch size ``B``.
            n_beats: Padded beat count ``T``.
            frames_per_beat: Optional sub-beat frame count ``F``.

        Returns:
            Modulated tensor with the same shape as ``features``.
        """
        params = self.gamma_beta(self.hidden(cond_vec))
        delta_gamma, beta = ops.split(params, 2, axis=-1)  # type: ignore[misc]
        one = ops.cast(1.0, delta_gamma.dtype)
        scale = one + delta_gamma
        if frames_per_beat is None:
            scale_b = ops.expand_dims(scale, axis=1)
            beta_b = ops.expand_dims(beta, axis=1)
            return features * scale_b + beta_b
        reshaped = ops.reshape(
            features,
            (batch_size, n_beats, frames_per_beat, self.d_model),
        )
        scale_b = ops.reshape(scale, (batch_size, 1, 1, self.d_model))
        beta_b = ops.reshape(beta, (batch_size, 1, 1, self.d_model))
        modulated = reshaped * scale_b + beta_b
        return ops.reshape(
            modulated,
            (ops.multiply(batch_size, n_beats), frames_per_beat, self.d_model),
        )


@keras.saving.register_keras_serializable(package="stepcovnet.itgpt")
class ItgptPlacementModel(keras.Model):
    """Subclassed ITGPT placement network (dynamic beat length).

    Args:
        d_model: Transformer width.
        n_heads: Attention heads.
        n_enc_layers: Global encoder depth.
        cnn_hidden: Beat CNN base channels.
        dropout_rate: Dropout.
        max_beats: Positional table length.
        audio_feature_type: Beat audio frontend (``"logmel"`` or ``"mert"``).
        n_frames_per_beat: Input frames per beat (``32`` for log-mel, ``24``
            for MERT).
        feature_dim: Per-frame feature width when ``audio_feature_type == "mert"``.
        early_film: If True, apply zero-initialized FiLM modulation from
            ``[difficulty, bpm]`` before the sub-beat attention stack and before
            the global encoder.
        cyclic_measure_pos: If True, add zero-initialized cyclic ``beat % 4``
            and ``beat % 16`` positional embeddings before ``level6``.
        **kwargs: Keras model kwargs.

    Raises:
        ValueError: If frontend arguments are invalid.
    """

    def __init__(
        self,
        *,
        d_model: int = constants.D_MODEL,
        n_heads: int = constants.N_HEADS,
        n_enc_layers: int = constants.N_ENC_LAYERS,
        cnn_hidden: int = constants.CNN_HIDDEN,
        dropout_rate: float = constants.DROPOUT_RATE,
        max_beats: int = constants.MAX_BEATS,
        audio_feature_type: str = ddcl_constants.AUDIO_FEATURE_LOGMEL,
        n_frames_per_beat: int = constants.N_FRAMES_PER_BEAT,
        feature_dim: int = ddcl_constants.MERT_FEATURE_DIM,
        early_film: bool = False,
        cyclic_measure_pos: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if audio_feature_type not in ddcl_constants.AUDIO_FEATURE_TYPES:
            raise ValueError(
                "audio_feature_type must be one of "
                f"{ddcl_constants.AUDIO_FEATURE_TYPES}, got {audio_feature_type!r}"
            )
        if (
            audio_feature_type == ddcl_constants.AUDIO_FEATURE_MERT
            and n_frames_per_beat != constants.CNN_FRAMES_OUT
        ):
            raise ValueError(
                "MERT ITGPT requires n_frames_per_beat == "
                f"{constants.CNN_FRAMES_OUT}, got {n_frames_per_beat}"
            )
        if feature_dim < 1:
            raise ValueError(f"feature_dim must be positive, got {feature_dim}")
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_enc_layers = n_enc_layers
        self.cnn_hidden = cnn_hidden
        self.dropout_rate = dropout_rate
        self.max_beats = max_beats
        self.audio_feature_type = audio_feature_type
        self.n_frames_per_beat = n_frames_per_beat
        self.feature_dim = feature_dim
        self.early_film = bool(early_film)
        self.cyclic_measure_pos = bool(cyclic_measure_pos)
        self.beat_cnn: ItgptBeatCNNEncoder | None = None
        self.beat_mert: ItgptBeatMertEncoder | None = None
        if audio_feature_type == ddcl_constants.AUDIO_FEATURE_MERT:
            self.beat_mert = ItgptBeatMertEncoder(
                d_model, dropout_rate=dropout_rate, name="beat_mert"
            )
        else:
            self.beat_cnn = ItgptBeatCNNEncoder(d_model, cnn_hidden, name="beat_cnn")
        self.diff_proj = ItgptConditionProj(
            d_model,
            constants.MIN_DIFFICULTY,
            constants.MAX_DIFFICULTY,
            name="diff_proj",
        )
        self.bpm_proj = ItgptConditionProj(
            d_model,
            constants.MIN_BPM,
            constants.MAX_BPM,
            name="bpm_proj",
        )
        self.early_film_subbeat: ItgptConditionFilm | None = None
        self.early_film_beat: ItgptConditionFilm | None = None
        if self.early_film:
            self.early_film_subbeat = ItgptConditionFilm(
                d_model, name="early_film_subbeat"
            )
            self.early_film_beat = ItgptConditionFilm(
                d_model, name="early_film_beat"
            )
        self.level1 = ItgptHierarchicalAttnBlock(
            d_model,
            n_heads,
            dropout_rate,
            frames_per_beat=constants.CNN_FRAMES_OUT,
            beats_per_chunk=1,
            name="level1",
        )
        self.level2 = ItgptHierarchicalAttnBlock(
            d_model,
            n_heads,
            dropout_rate,
            frames_per_beat=constants.CNN_FRAMES_OUT,
            beats_per_chunk=4,
            name="level2",
        )
        self.level3 = ItgptHierarchicalAttnBlock(
            d_model,
            n_heads,
            dropout_rate,
            frames_per_beat=constants.CNN_FRAMES_OUT,
            beats_per_chunk=16,
            name="level3",
        )
        self.level4 = ItgptHierarchicalAttnBlock(
            d_model,
            n_heads,
            dropout_rate,
            frames_per_beat=constants.CNN_FRAMES_OUT,
            beats_per_chunk=32,
            name="level4",
        )
        self.frame_compress = ItgptGatedCompression(d_model, name="frame_compress")
        self.level5 = ItgptHierarchicalAttnBlock(
            d_model,
            n_heads,
            dropout_rate,
            frames_per_beat=6,
            beats_per_chunk=64,
            name="level5",
        )
        self.beat_compress = ItgptGatedCompression(d_model, name="beat_compress")
        self.level6_pos = keras.layers.Embedding(
            max_beats + 200,
            d_model,
            embeddings_initializer=keras.initializers.RandomNormal(stddev=0.02),  # type: ignore[arg-type]
            name="level6_pos",
        )
        self.bar_pos_emb: keras.layers.Embedding | None = None
        self.phrase_pos_emb: keras.layers.Embedding | None = None
        if self.cyclic_measure_pos:
            self.bar_pos_emb = keras.layers.Embedding(
                4,
                d_model,
                embeddings_initializer="zeros",
                name="bar_pos_emb",
            )
            self.phrase_pos_emb = keras.layers.Embedding(
                16,
                d_model,
                embeddings_initializer="zeros",
                name="phrase_pos_emb",
            )
        self.level6 = [
            ItgptEncoderBlock(d_model, n_heads, dropout_rate, name=f"level6_{index}")
            for index in range(n_enc_layers)
        ]
        self.conv_head = ItgptConvHead(d_model, dropout_rate, name="conv_head")
        self.onset_head = keras.layers.Dense(
            constants.N_SLOTS, name="onset_head", dtype="float32"
        )
        self.onset_probs = keras.layers.Activation(
            "sigmoid", name="onset_probs", dtype="float32"
        )

    def call(self, inputs: dict[str, Any], training: bool | None = None) -> Any:
        """Run the hierarchical encoder.

        Args:
            inputs: Dict with ``audio``, ``bpm``, ``difficulty``.
            training: Keras training flag.

        Returns:
            Slot probabilities ``(batch, beats, 48)``.

        Raises:
            RuntimeError: If the active beat encoder was not initialized.
        """
        audio = inputs["audio"]
        batch_size = ops.shape(audio)[0]  # type: ignore[index]
        n_beats = ops.shape(audio)[1]  # type: ignore[index]
        n_beat_rows = ops.multiply(batch_size, n_beats)
        if self.audio_feature_type == ddcl_constants.AUDIO_FEATURE_MERT:
            if self.beat_mert is None:
                raise RuntimeError("beat_mert encoder is not initialized")
            mert_in = ops.reshape(
                audio,
                (n_beat_rows, self.n_frames_per_beat, self.feature_dim),
            )
            hidden = self.beat_mert(mert_in, training=training)
        else:
            if self.beat_cnn is None:
                raise RuntimeError("beat_cnn encoder is not initialized")
            conv_in = ops.reshape(
                audio,
                (
                    n_beat_rows,
                    constants.N_FRAMES_PER_BEAT,
                    constants.N_MELS,
                    constants.N_CHANNELS,
                ),
            )
            hidden = self.beat_cnn(conv_in)
        diff_tok = self.diff_proj(inputs["difficulty"])
        bpm_tok = self.bpm_proj(inputs["bpm"])
        cond_vec = ops.concatenate(
            [ops.squeeze(diff_tok, axis=1), ops.squeeze(bpm_tok, axis=1)],
            axis=-1,
        )
        if self.early_film_subbeat is not None:
            hidden = self.early_film_subbeat(
                hidden,
                cond_vec,
                batch_size=batch_size,
                n_beats=n_beats,
                frames_per_beat=constants.CNN_FRAMES_OUT,
            )
        hidden = self.level1(
            hidden, batch_size=batch_size, n_beats=n_beats, training=training
        )
        hidden = self.level2(
            hidden, batch_size=batch_size, n_beats=n_beats, training=training
        )
        hidden = self.level3(
            hidden, batch_size=batch_size, n_beats=n_beats, training=training
        )
        hidden = self.level4(
            hidden, batch_size=batch_size, n_beats=n_beats, training=training
        )
        hidden = ops.reshape(hidden, (n_beat_rows, 6, 4, self.d_model))
        hidden = ops.reshape(hidden, (n_beat_rows, 6, 4 * self.d_model))
        hidden = self.frame_compress(hidden)
        hidden = self.level5(
            hidden, batch_size=batch_size, n_beats=n_beats, training=training
        )
        global_h = self.beat_compress(
            ops.reshape(hidden, (n_beat_rows, 6 * self.d_model))
        )
        global_h = ops.reshape(global_h, (batch_size, n_beats, self.d_model))
        if self.early_film_beat is not None:
            global_h = self.early_film_beat(
                global_h,
                cond_vec,
                batch_size=batch_size,
                n_beats=n_beats,
            )
        beat_idx = ops.arange(n_beats, dtype="int32")
        pos = self.level6_pos(beat_idx)
        if self.bar_pos_emb is not None and self.phrase_pos_emb is not None:
            pos = (
                pos
                + self.bar_pos_emb(ops.mod(beat_idx, 4))
                + self.phrase_pos_emb(ops.mod(beat_idx, 16))
            )
        global_h = global_h + pos
        global_h = ops.concatenate([diff_tok, bpm_tok, global_h], axis=1)
        for block in self.level6:
            global_h = block(global_h, training=training)
        global_h = global_h[:, 2:, :]  # type: ignore[index]
        global_h = self.conv_head(global_h, training=training)
        return self.onset_probs(self.onset_head(global_h))

    def get_config(self) -> dict:
        """Return constructor kwargs for Keras serialization.

        Returns:
            Config mapping.
        """
        config = super().get_config()
        config.update(
            {
                "d_model": self.d_model,
                "n_heads": self.n_heads,
                "n_enc_layers": self.n_enc_layers,
                "cnn_hidden": self.cnn_hidden,
                "dropout_rate": self.dropout_rate,
                "max_beats": self.max_beats,
                "audio_feature_type": self.audio_feature_type,
                "n_frames_per_beat": self.n_frames_per_beat,
                "feature_dim": self.feature_dim,
                "early_film": self.early_film,
                "cyclic_measure_pos": self.cyclic_measure_pos,
            }
        )
        return config


def build_itgpt_placement_model(
    *,
    d_model: int = constants.D_MODEL,
    n_heads: int = constants.N_HEADS,
    n_enc_layers: int = constants.N_ENC_LAYERS,
    cnn_hidden: int = constants.CNN_HIDDEN,
    dropout_rate: float = constants.DROPOUT_RATE,
    max_beats: int = constants.MAX_BEATS,
    audio_feature_type: str = ddcl_constants.AUDIO_FEATURE_LOGMEL,
    n_frames_per_beat: int = constants.N_FRAMES_PER_BEAT,
    feature_dim: int = ddcl_constants.MERT_FEATURE_DIM,
    early_film: bool = False,
    cyclic_measure_pos: bool = False,
    model_name: str = "itgpt_placement",
) -> keras.Model:
    """Build the ITGPT placement network.

    Inputs are a padded beat tensor ``audio`` (``(T, 32, 80, 3)`` for log-mel
    or ``(T, 24, feature_dim)`` for MERT) with ``T % 64 == 0``, plus scalar
    ``bpm`` / ``difficulty``. Output is a 48-way sigmoid per beat (padded tail
    included; mask it in the loss).

    Args:
        d_model: Transformer width.
        n_heads: Attention heads.
        n_enc_layers: Global encoder depth.
        cnn_hidden: Beat CNN base channels.
        dropout_rate: Dropout.
        max_beats: Maximum padded beat length.
        audio_feature_type: Beat audio frontend (``"logmel"`` or ``"mert"``).
        n_frames_per_beat: Input frames per beat.
        feature_dim: Per-frame width when ``audio_feature_type == "mert"``.
        early_film: If True, apply zero-initialized FiLM conditioning from
            ``[difficulty, bpm]`` before the sub-beat and global stacks.
        cyclic_measure_pos: If True, add zero-initialized cyclic ``beat % 4``
            and ``beat % 16`` positional embeddings before ``level6``.
        model_name: Keras model name.

    Returns:
        Uncompiled Keras model.

    Raises:
        ValueError: If sizes or frontend parameters are invalid.
    """
    if d_model < 1 or n_heads < 1 or d_model % n_heads != 0:
        raise ValueError(f"n_heads must divide d_model, got {n_heads}, {d_model}")
    if n_enc_layers < 1:
        raise ValueError(f"n_enc_layers must be at least 1, got {n_enc_layers}")
    return ItgptPlacementModel(
        d_model=d_model,
        n_heads=n_heads,
        n_enc_layers=n_enc_layers,
        cnn_hidden=cnn_hidden,
        dropout_rate=dropout_rate,
        max_beats=max_beats,
        audio_feature_type=audio_feature_type,
        n_frames_per_beat=n_frames_per_beat,
        feature_dim=feature_dim,
        early_film=early_film,
        cyclic_measure_pos=cyclic_measure_pos,
        name=model_name,
    )
