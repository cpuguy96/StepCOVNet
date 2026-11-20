"""Branched ConvLSTM placement model from Dance Dance ConvLSTM (`omalley2025ddcl`).

Layer graph follows ``get_onset_model`` (default branch: not
``full_bidirectional``, not ``conv3d``) in
https://github.com/miguelomalley/DDCL/blob/5b1375c642bb708b3c66baf5d880fbf865b85097/models.py
"""

from __future__ import annotations

import keras

from stepcovnet.ddcl import constants


def build_convlstm_placement_model(
    *,
    memlen: int = constants.MEMLEN,
    n_frames: int = constants.N_FRAMES_PER_BEAT,
    lstm_units: int = constants.LSTM_UNITS,
    dropout_rate: float = constants.DROPOUT_RATE,
    dense_sizes: tuple[int, ...] = constants.DENSE_SIZES,
    model_name: str = "ddcl_convlstm_placement",
) -> keras.Model:
    """Build the default DDCL onset/placement network.

    Forward and backward ConvLSTM2D branches (16 then 32 filters, kernels
    ``(7,3)`` / ``(3,3)``, freq pool 3) each see ``memlen+1`` beats of
    ``(n_frames, 80, 3)`` audio plus ``[meter, bpm]`` stream features. Two
    LSTMs per branch, concatenate, then Dense 512/256 leaky ReLU and a 48-way
    sigmoid.

    Args:
        memlen: Context beats besides the current one (paper: 15).
        n_frames: Audio frames per beat (paper: 32).
        lstm_units: LSTM width (paper: 200).
        dropout_rate: Dropout on LSTM and dense layers (paper: 0.2).
        dense_sizes: Hidden widths (paper: 512, 256).
        model_name: Keras model name.

    Returns:
        Uncompiled Keras model.

    Raises:
        ValueError: If sizes are invalid.
    """
    if memlen < 0:
        raise ValueError(f"memlen must be >= 0, got {memlen}")
    if n_frames < 1:
        raise ValueError(f"n_frames must be at least 1, got {n_frames}")
    if lstm_units < 1:
        raise ValueError(f"lstm_units must be at least 1, got {lstm_units}")
    context = memlen + 1
    audio_fwd = keras.Input(
        shape=(
            context,
            n_frames,
            constants.N_MELS,
            constants.N_CHANNELS,
        ),
        name="audio_fwd",
    )
    audio_bwd = keras.Input(
        shape=(
            context,
            n_frames,
            constants.N_MELS,
            constants.N_CHANNELS,
        ),
        name="audio_bwd",
    )
    stream_fwd = keras.Input(
        shape=(context, constants.STREAM_DIM),
        name="stream_fwd",
    )
    stream_bwd = keras.Input(
        shape=(context, constants.STREAM_DIM),
        name="stream_bwd",
    )

    def _audio_branch(audio: object, *, go_backwards: bool, name: str) -> object:
        conv = keras.layers.ConvLSTM2D(
            constants.CONV1_FILTERS,
            constants.CONV1_KERNEL,
            return_sequences=True,
            go_backwards=go_backwards,
            name=f"{name}_convlstm0",
        )(audio)
        conv = keras.layers.MaxPooling3D(
            constants.POOL_SIZE,
            strides=constants.POOL_SIZE,
            name=f"{name}_pool0",
        )(conv)
        conv = keras.layers.ConvLSTM2D(
            constants.CONV2_FILTERS,
            constants.CONV2_KERNEL,
            return_sequences=True,
            name=f"{name}_convlstm1",
        )(conv)
        conv = keras.layers.MaxPooling3D(
            constants.POOL_SIZE,
            strides=constants.POOL_SIZE,
            name=f"{name}_pool1",
        )(conv)
        return keras.layers.Reshape((context, -1), name=f"{name}_flat")(conv)

    audio_out = _audio_branch(audio_fwd, go_backwards=False, name="fwd")
    audio_out_b = _audio_branch(audio_bwd, go_backwards=True, name="bwd")
    merge = keras.layers.Concatenate(axis=-1, name="fwd_merge")([audio_out, stream_fwd])
    merge_b = keras.layers.Concatenate(axis=-1, name="bwd_merge")(
        [audio_out_b, stream_bwd]
    )
    note = keras.layers.LSTM(
        lstm_units,
        return_sequences=True,
        dropout=dropout_rate,
        name="fwd_lstm0",
    )(merge)
    note = keras.layers.LSTM(lstm_units, dropout=dropout_rate, name="fwd_lstm1")(note)
    note_b = keras.layers.LSTM(
        lstm_units,
        return_sequences=True,
        dropout=dropout_rate,
        name="bwd_lstm0",
    )(merge_b)
    note_b = keras.layers.LSTM(lstm_units, dropout=dropout_rate, name="bwd_lstm1")(
        note_b
    )
    combined = keras.layers.Concatenate(axis=1, name="branch_concat")([note, note_b])
    dense = combined
    for dnn_idx, width in enumerate(dense_sizes):
        dense = keras.layers.Dense(
            width,
            activation="leaky_relu",
            name=f"dnn_{dnn_idx}",
        )(dense)
        dense = keras.layers.Dropout(dropout_rate, name=f"dnn_dropout_{dnn_idx}")(dense)
    outputs = keras.layers.Dense(
        constants.N_SLOTS,
        activation="sigmoid",
        name="slots",
    )(dense)
    return keras.Model(
        inputs={
            "audio_fwd": audio_fwd,
            "audio_bwd": audio_bwd,
            "stream_fwd": stream_fwd,
            "stream_bwd": stream_bwd,
        },
        outputs=outputs,
        name=model_name,
    )


def build_mert_placement_model(
    *,
    memlen: int = constants.MEMLEN,
    n_frames: int = constants.MERT_FRAMES_PER_BEAT,
    feature_dim: int = constants.MERT_FEATURE_DIM,
    proj_dim: int = constants.MERT_PROJ_DIM,
    conv_filters: int = constants.MERT_CONV_FILTERS,
    lstm_units: int = constants.LSTM_UNITS,
    dropout_rate: float = constants.DROPOUT_RATE,
    dense_sizes: tuple[int, ...] = constants.DENSE_SIZES,
    model_name: str = "ddcl_mert_placement",
) -> keras.Model:
    """Build a beat-aligned MERT placement model matching DDCL's downstream head.

    Replaces the scratch 2D log-mel ConvLSTM frontend with a shared per-beat
    sub-beat encoder (``LayerNormalization -> Dense -> Dropout -> Conv1D ->
    AveragePooling1D -> Flatten``) applied across ``memlen+1`` beats. Forward
    and backward audio LSTMs align with ``stream_fwd`` / ``stream_bwd``
    (``[meter, bpm]``) before the same second LSTM, dense stack, and 48-way
    sigmoid output.

    Args:
        memlen: Context beats besides the current one (paper: 15).
        n_frames: Sub-beat MERT frames per beat (default: 16).
        feature_dim: MERT embedding width (default: 1024).
        proj_dim: Per-frame linear projection width (default: 128).
        conv_filters: Sub-beat 1D conv filters (default: 64).
        lstm_units: Hidden width per LSTM layer (paper: 200).
        dropout_rate: Dropout rate on projection, LSTM, and dense layers.
        dense_sizes: Fully-connected hidden widths (paper: 512, 256).
        model_name: Keras model name.

    Returns:
        Uncompiled Keras model with identical input dict keys as DDCL.

    Raises:
        ValueError: If any dimension hyperparameter is invalid.
    """
    if memlen < 0:
        raise ValueError(f"memlen must be >= 0, got {memlen}")
    if n_frames < 1:
        raise ValueError(f"n_frames must be at least 1, got {n_frames}")
    if feature_dim < 1:
        raise ValueError(f"feature_dim must be at least 1, got {feature_dim}")
    if proj_dim < 1:
        raise ValueError(f"proj_dim must be at least 1, got {proj_dim}")
    if conv_filters < 1:
        raise ValueError(f"conv_filters must be at least 1, got {conv_filters}")
    if lstm_units < 1:
        raise ValueError(f"lstm_units must be at least 1, got {lstm_units}")

    context = memlen + 1
    audio_fwd = keras.Input(
        shape=(context, n_frames, feature_dim),
        name="audio_fwd",
    )
    audio_bwd = keras.Input(
        shape=(context, n_frames, feature_dim),
        name="audio_bwd",
    )
    stream_fwd = keras.Input(
        shape=(context, constants.STREAM_DIM),
        name="stream_fwd",
    )
    stream_bwd = keras.Input(
        shape=(context, constants.STREAM_DIM),
        name="stream_bwd",
    )

    mert_ln = keras.layers.LayerNormalization(axis=-1, name="mert_ln")
    mert_proj = keras.layers.Dense(proj_dim, activation="gelu", name="mert_proj")
    mert_drop = keras.layers.Dropout(dropout_rate, name="mert_proj_dropout")
    mert_conv = keras.layers.Conv2D(
        conv_filters,
        kernel_size=(1, 3),
        padding="same",
        activation="gelu",
        name="mert_conv",
    )
    mert_pool = (
        keras.layers.AveragePooling2D(pool_size=(1, 2), name="mert_pool")
        if n_frames >= 2
        else None
    )

    def _encode_mert(audio: object, *, name: str) -> object:
        x = mert_ln(audio)
        x = mert_proj(x)
        x = mert_drop(x)
        x = mert_conv(x)
        if mert_pool is not None:
            x = mert_pool(x)
        return keras.layers.Reshape((context, -1), name=f"{name}_flat")(x)

    audio_out_fwd = _encode_mert(audio_fwd, name="fwd")
    audio_out_bwd = _encode_mert(audio_bwd, name="bwd")
    fwd_audio_seq = keras.layers.LSTM(
        lstm_units,
        return_sequences=True,
        go_backwards=False,
        dropout=dropout_rate,
        name="fwd_lstm0",
    )(audio_out_fwd)
    bwd_audio_seq = keras.layers.LSTM(
        lstm_units,
        return_sequences=True,
        go_backwards=True,
        dropout=dropout_rate,
        name="bwd_lstm0",
    )(audio_out_bwd)
    merge = keras.layers.Concatenate(axis=-1, name="fwd_merge")(
        [fwd_audio_seq, stream_fwd]
    )
    merge_b = keras.layers.Concatenate(axis=-1, name="bwd_merge")(
        [bwd_audio_seq, stream_bwd]
    )
    note = keras.layers.LSTM(
        lstm_units,
        dropout=dropout_rate,
        name="fwd_lstm1",
    )(merge)
    note_b = keras.layers.LSTM(
        lstm_units,
        dropout=dropout_rate,
        name="bwd_lstm1",
    )(merge_b)
    combined = keras.layers.Concatenate(axis=1, name="branch_concat")([note, note_b])
    dense = combined
    for dnn_idx, width in enumerate(dense_sizes):
        dense = keras.layers.Dense(
            width,
            activation="leaky_relu",
            name=f"dnn_{dnn_idx}",
        )(dense)
        dense = keras.layers.Dropout(dropout_rate, name=f"dnn_dropout_{dnn_idx}")(dense)
    outputs = keras.layers.Dense(
        constants.N_SLOTS,
        activation="sigmoid",
        name="slots",
    )(dense)
    return keras.Model(
        inputs={
            "audio_fwd": audio_fwd,
            "audio_bwd": audio_bwd,
            "stream_fwd": stream_fwd,
            "stream_bwd": stream_bwd,
        },
        outputs=outputs,
        name=model_name,
    )
