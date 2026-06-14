"""Full Keras model for event-based onset detection."""

import keras
import numpy as np
import tensorflow as tf

from stepcovnet.onset_events import config
from stepcovnet.onset_events import encoder
from stepcovnet.onset_events import frontend
from stepcovnet.onset_events import preprocess


@keras.saving.register_keras_serializable(package="stepcovnet")
class BroadcastQueryEmbeddings(keras.layers.Layer):
    """Tile learned query embeddings across the batch dimension.

    Attributes:
        num_queries: Fixed query slot count ``K``.
        embed_dim: Query vector dimension.
    """

    def __init__(self, num_queries: int, embed_dim: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self._num_queries = num_queries
        self._embed = keras.layers.Embedding(num_queries, embed_dim)

    def build(self, input_shape) -> None:
        """Build the query embedding table."""
        self._embed.build((self._num_queries,))

    def call(self, encoder_out: tf.Tensor) -> tf.Tensor:
        indices = tf.range(self._num_queries, dtype=tf.int32)
        queries = self._embed(indices)
        queries = tf.expand_dims(queries, axis=0)
        batch = tf.shape(encoder_out)[0]
        return tf.tile(queries, [batch, 1, 1])


@keras.saving.register_keras_serializable(package="stepcovnet")
class QuerySpreadTimeNorm(keras.layers.Layer):
    """Per-query normalized onset times from reference logits plus a learned delta.

    Attributes:
        num_queries: Fixed query slot count ``K``.
        ref_normalized: Optional per-slot reference times in ``[0, 1]``; defaults to a
            uniform grid when ``None``.
    """

    def __init__(
        self,
        num_queries: int,
        ref_normalized: tuple[float, ...] | list[float] | None = None,
        learn_time_delta: bool = True,
        ref_weight: float | None = None,
        **kwargs,
    ) -> None:
        _ = ref_weight
        super().__init__(**kwargs)
        self._num_queries = num_queries
        self._learn_time_delta = learn_time_delta
        if ref_normalized is not None and len(ref_normalized) != num_queries:
            raise ValueError(
                f"ref_normalized length {len(ref_normalized)} != num_queries "
                f"{num_queries}"
            )
        self._ref_normalized = (
            None if ref_normalized is None else tuple(float(v) for v in ref_normalized)
        )
        self._delta_dense = None
        if learn_time_delta:
            self._delta_dense = keras.layers.Dense(
                1,
                kernel_initializer=keras.initializers.Zeros(),
                bias_initializer=keras.initializers.Zeros(),
            )

    def build(self, input_shape) -> None:
        """Create the per-query delta projection."""
        if self._delta_dense is not None:
            self._delta_dense.build(input_shape)
        if self._ref_normalized is None:
            ref = (np.arange(self._num_queries, dtype=np.float32) + 0.5) / float(
                self._num_queries
            )
        else:
            ref = np.asarray(self._ref_normalized, dtype=np.float32)
        ref = np.clip(ref, 1e-4, 1.0 - 1e-4)
        self._ref_logit = tf.constant(np.log(ref / (1.0 - ref)), dtype=tf.float32)
        super().build(input_shape)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        """Return sigmoid(reference logit + delta) with shape ``(batch, K)``."""
        ref_logit = tf.tile(self._ref_logit[tf.newaxis, :], [tf.shape(x)[0], 1])
        if not self._learn_time_delta:
            return tf.sigmoid(ref_logit)
        assert self._delta_dense is not None
        delta = self._delta_dense(x)
        delta = tf.reshape(delta, (-1, self._num_queries))
        return tf.sigmoid(ref_logit + delta)

    def get_config(self) -> dict:
        """Return layer configuration for serialization."""
        config_dict = super().get_config()
        config_dict.update(
            {
                "num_queries": self._num_queries,
                "ref_normalized": self._ref_normalized,
                "learn_time_delta": self._learn_time_delta,
            }
        )
        return config_dict


def _build_query_head(
    encoder_out: keras.KerasTensor,
    *,
    num_queries: int,
    embed_dim: int,
    decoder_layers: int,
    query_ref_normalized: tuple[float, ...] | None = None,
    learn_time_delta: bool = True,
) -> tuple[keras.KerasTensor, keras.KerasTensor]:
    """Cross-attention query decoder producing normalized times and confidence."""
    query_emb = BroadcastQueryEmbeddings(
        num_queries, embed_dim, name="query_broadcast"
    )(encoder_out)
    x = query_emb
    for layer_idx in range(decoder_layers):
        attn_out = keras.layers.MultiHeadAttention(
            num_heads=4,
            key_dim=embed_dim // 4,
            name=f"query_cross_attn_{layer_idx}",
        )(query=x, value=encoder_out, key=encoder_out)
        x = keras.layers.Add(name=f"query_residual_{layer_idx}")([x, attn_out])
        x = keras.layers.LayerNormalization(name=f"query_norm_{layer_idx}")(x)
    pred_confidence = keras.layers.Dense(
        1,
        activation="sigmoid",
        bias_initializer=keras.initializers.Constant(-2.0),
        name="pred_confidence_logits",
    )(x)
    pred_confidence = keras.layers.Reshape((num_queries,), name="pred_confidence")(
        pred_confidence
    )
    pred_times_norm = QuerySpreadTimeNorm(
        num_queries,
        ref_normalized=query_ref_normalized,
        learn_time_delta=learn_time_delta,
        name="pred_times_norm",
    )(x)
    return pred_times_norm, pred_confidence


def build_onset_event_model(
    model_config: config.OnsetEventModelConfig,
    *,
    query_ref_normalized: tuple[float, ...] | None = None,
    learn_time_delta: bool = True,
) -> keras.Model:
    """Build the event onset model: waveform in, query times and confidence out.

    When ``model_config.include_duration_input`` is True, the model expects
    ``duration`` (seconds, scalar per batch item) and returns ``pred_times`` in
    ``[0, duration]`` via ``sigmoid * duration``. Otherwise ``pred_times`` are
    normalized to ``[0, 1]`` and should be scaled outside the graph.

    Args:
        model_config: Architecture and input sizing options.
        query_ref_normalized: Optional per-query reference times in ``[0, 1]`` for
            :class:`QuerySpreadTimeNorm`; length must equal ``num_queries``.
        learn_time_delta: When ``False``, predicted times follow the reference grid
            exactly and the per-query delta head is skipped.

    Returns:
        Keras model with outputs ``pred_times`` and ``pred_confidence`` of
        shape ``(batch, K)`` where ``K = num_queries``.

    Raises:
        ValueError: If ``frontend`` is not supported.
    """
    frontend_name = preprocess.validate_frontend(model_config.frontend)
    max_frames = frontend.target_encoder_frames(
        model_config.max_audio_seconds,
        model_config.frame_hop_sec,
    )

    if frontend_name == preprocess.FRONTEND_CONV1D:
        frontend_model = frontend.build_audio_frontend(
            target_sample_rate=model_config.target_sample_rate,
            max_audio_seconds=model_config.max_audio_seconds,
            frame_hop_sec=model_config.frame_hop_sec,
            base_filters=model_config.base_filters,
            name="onset_event_frontend",
        )
        encoder_input = keras.Input(
            shape=frontend_model.input_shape[1:],
            name="audio",
            dtype=tf.float32,
        )
        embeddings = frontend_model(encoder_input)
    else:
        input_features = preprocess.encoder_feature_dim(frontend_name)
        frontend_model = frontend.build_cached_feature_frontend(
            input_features=input_features,
            output_features=model_config.base_filters,
            max_frames=max_frames,
            name="onset_event_frontend",
        )
        encoder_input = keras.Input(
            shape=(max_frames, input_features),
            name="features",
            dtype=tf.float32,
        )
        embeddings = frontend_model(encoder_input)

    encoder_model = encoder.build_temporal_encoder(
        model_config.base_filters,
        model_config.encoder.as_dict(),
    )
    encoder_out = encoder_model(embeddings)
    pred_times_norm, pred_confidence = _build_query_head(
        encoder_out,
        num_queries=model_config.num_queries,
        embed_dim=model_config.embed_dim,
        decoder_layers=model_config.decoder_layers,
        query_ref_normalized=query_ref_normalized,
        learn_time_delta=learn_time_delta,
    )

    if model_config.include_duration_input:
        duration_input = keras.Input(shape=(), name="duration", dtype=tf.float32)
        duration_broadcast = keras.layers.Reshape((1,), name="duration_broadcast")(
            duration_input
        )
        pred_times = keras.layers.Multiply(name="pred_times")(
            [pred_times_norm, duration_broadcast]
        )
        inputs = [encoder_input, duration_input]
    else:
        pred_times = keras.layers.Identity(name="pred_times")(pred_times_norm)
        inputs = encoder_input

    return keras.Model(
        inputs=inputs,
        outputs={
            "pred_times": pred_times,
            "pred_confidence": pred_confidence,
        },
        name="onset_event_model",
    )
