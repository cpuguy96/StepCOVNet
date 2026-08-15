"""C-LSTM step-placement model from Dance Dance Convolution (`donahue2017ddc`)."""

from __future__ import annotations

import keras
import tensorflow as tf

from stepcovnet.ddc import constants


@keras.saving.register_keras_serializable(package="ddc")
class DdcPerFrameCNN(keras.layers.Layer):
    """Apply the DDC CNN independently at each time step.

    TimeDistributed Conv2D in Keras 3 can freeze the train-time length, so this
    layer reshapes ``(batch, time, 15, 80, 3)`` to a 4D CNN batch and back.

    Args:
        **kwargs: Forwarded to ``keras.layers.Layer``.
    """

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.conv0 = keras.layers.Conv2D(
            constants.CNN_FILTERS[0],
            constants.CNN_KERNELS[0],
            activation="relu",
            padding="valid",
            name="conv0",
        )
        self.pool0 = keras.layers.MaxPool2D(
            constants.CNN_POOL,
            strides=constants.CNN_POOL,
            padding="same",
            name="pool0",
        )
        self.conv1 = keras.layers.Conv2D(
            constants.CNN_FILTERS[1],
            constants.CNN_KERNELS[1],
            activation="relu",
            padding="valid",
            name="conv1",
        )
        self.pool1 = keras.layers.MaxPool2D(
            constants.CNN_POOL,
            strides=constants.CNN_POOL,
            padding="same",
            name="pool1",
        )
        self.flatten = keras.layers.Flatten(name="flatten")

    def call(self, x: tf.Tensor) -> tf.Tensor:
        """Encode each 15×80×3 window.

        Args:
            x: Audio windows, shape ``(batch, time, 15, 80, 3)``.

        Returns:
            Flattened CNN features, shape ``(batch, time, 1120)``.
        """
        batch = tf.shape(x)[0]
        time = tf.shape(x)[1]
        frames = tf.reshape(
            x,
            (
                -1,
                constants.CONTEXT_FRAMES,
                constants.N_MELS,
                constants.N_CHANNELS,
            ),
        )
        hidden = self.flatten(self.pool1(self.conv1(self.pool0(self.conv0(frames)))))
        n_features = tf.shape(hidden)[-1]
        return tf.reshape(hidden, (batch, time, n_features))


def build_clstm_placement_model(
    *,
    lstm_units: int = constants.LSTM_UNITS,
    lstm_layers: int = constants.LSTM_LAYERS,
    dropout_rate: float = constants.DROPOUT_RATE,
    dnn_sizes: tuple[int, ...] = constants.DNN_SIZES,
    model_name: str = "ddc_clstm_placement",
) -> keras.Model:
    """Build the DDC C-LSTM placement network.

    CNN (VALID 7×3 / 3×3, freq max-pool 3) is applied per 15×80×3 context window,
    concatenated with a 5-way difficulty one-hot, then two LSTMs and two ReLU
    dense layers with dropout, ending in a per-frame sigmoid.

    Args:
        lstm_units: Hidden size of each LSTM layer (paper: 200).
        lstm_layers: Number of stacked LSTM layers (paper: 2).
        dropout_rate: Dropout after each LSTM and dense layer (paper: 0.5).
        dnn_sizes: Hidden widths of the fully-connected stack (paper: 256, 128).
        model_name: Keras model name.

    Returns:
        Uncompiled Keras model with inputs ``audio`` ``(T, 15, 80, 3)`` and
        ``difficulty`` ``(T, 5)``, output ``(T, 1)``.

    Raises:
        ValueError: If ``lstm_layers`` or ``lstm_units`` is less than 1.
    """
    if lstm_layers < 1:
        raise ValueError(f"lstm_layers must be at least 1, got {lstm_layers}")
    if lstm_units < 1:
        raise ValueError(f"lstm_units must be at least 1, got {lstm_units}")

    audio = keras.Input(
        shape=(
            None,
            constants.CONTEXT_FRAMES,
            constants.N_MELS,
            constants.N_CHANNELS,
        ),
        name="audio",
    )
    difficulty = keras.Input(
        shape=(None, constants.N_DIFFICULTIES),
        name="difficulty",
    )
    conv_flat = DdcPerFrameCNN(name="ddc_cnn")(audio)
    merged = keras.layers.Concatenate(name="concat_difficulty")([conv_flat, difficulty])
    projected = keras.layers.Dense(lstm_units, name="rnn_proj")(merged)
    recurrent = projected
    for layer_idx in range(lstm_layers):
        recurrent = keras.layers.LSTM(
            lstm_units,
            return_sequences=True,
            name=f"lstm_{layer_idx}",
        )(recurrent)
        recurrent = keras.layers.Dropout(
            dropout_rate,
            name=f"lstm_dropout_{layer_idx}",
        )(recurrent)
    dense = recurrent
    for dnn_idx, width in enumerate(dnn_sizes):
        dense = keras.layers.Dense(width, activation="relu", name=f"dnn_{dnn_idx}")(
            dense
        )
        dense = keras.layers.Dropout(dropout_rate, name=f"dnn_dropout_{dnn_idx}")(dense)
    outputs = keras.layers.Dense(1, activation="sigmoid", name="onset")(dense)
    return keras.Model(
        inputs={"audio": audio, "difficulty": difficulty},
        outputs=outputs,
        name=model_name,
    )
