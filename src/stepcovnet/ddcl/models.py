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
