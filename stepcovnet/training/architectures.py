import tensorflow as tf
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import LayerNormalization
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import RepeatVector
from tensorflow.keras.layers import SpatialDropout1D
from tensorflow.keras.layers import SpatialDropout2D


def time_front(x_input, reshape_dim, channel_order, channel):
    x = Conv2D(8,
               (3, 7),
               padding="same",
               input_shape=reshape_dim,
               data_format=channel_order,
               kernel_regularizer=tf.keras.regularizers.L1L2(l2=1e-6),
               kernel_initializer=tf.keras.initializers.he_uniform(42),
               bias_initializer=tf.keras.initializers.he_uniform(42),
               dtype=tf.float32)(x_input)
    x = BatchNormalization(axis=channel)(x)
    x = Activation('relu')(x)
    x = Conv2D(8,
               (3, 3),
               padding="same",
               data_format=channel_order,
               kernel_initializer=tf.keras.initializers.he_uniform(42),
               bias_initializer=tf.keras.initializers.he_uniform(42))(x)
    x = BatchNormalization(axis=channel)(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(3, 1),
                     strides=3,
                     padding='same',
                     data_format=channel_order)(x)
    x = SpatialDropout2D(0.5)(x)
    x = Conv2D(16,
               (3, 3),
               padding="same",
               data_format=channel_order,
               kernel_initializer=tf.keras.initializers.he_uniform(42),
               bias_initializer=tf.keras.initializers.he_uniform(42))(x)
    x = BatchNormalization(axis=channel)(x)
    x = Activation('relu')(x)
    x = Conv2D(16,
               (3, 3),
               padding="same",
               data_format=channel_order,
               kernel_initializer=tf.keras.initializers.he_uniform(42),
               bias_initializer=tf.keras.initializers.he_uniform(42))(x)
    x = BatchNormalization(axis=channel)(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 1),
                     strides=2,
                     padding='same',
                     data_format=channel_order)(x)
    x = SpatialDropout2D(0.5)(x)
    return x


"""
    x = Conv2D(8,
               (3, 7),
               padding="valid",
               kernel_initializer='glorot_normal',
               input_shape=reshape_dim,
               data_format=channel_order)(x_input)
    x = BatchNormalization(axis=channel)(x)
    x = Activation('relu')(x)
    x = Conv2D(8,
               (3, 3),
               padding="valid",
               kernel_initializer='glorot_normal',
               data_format=channel_order)(x)
    x = BatchNormalization(axis=channel)(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(3, 1),
                     padding='valid',
                     data_format=channel_order)(x)
    x = SpatialDropout2D(0.5)(x)
    x = Conv2D(int(16),
               (3, 3),
               padding="valid",
               kernel_initializer='glorot_normal',
               data_format=channel_order)(x)
    x = BatchNormalization(axis=channel)(x)
    x = Activation('relu')(x)
    x = Conv2D(int(16),
               (3, 3),
               padding="valid",
               kernel_initializer='glorot_normal',
               data_format=channel_order)(x)
    x = BatchNormalization(axis=channel)(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 1),
                     padding='valid',
                     data_format=channel_order)(x)
    x = SpatialDropout2D(0.5)(x)
"""


def time_back(model, lookback, extra_input=None):
    if extra_input is not None:
        x = concatenate([Flatten()(model), extra_input])
        x = RepeatVector(lookback)(x)
    else:
        x = RepeatVector(lookback)(model)
    x = Bidirectional(LSTM(128,
                           return_sequences=True,
                           kernel_initializer=tf.keras.initializers.glorot_uniform(42),
                           kernel_regularizer=tf.keras.regularizers.L1L2(l2=5e-6)))(x)
    x = LayerNormalization(axis=-1, epsilon=1e-6, dtype=tf.float32)(x)
    x = Bidirectional(LSTM(128,
                           return_sequences=False,
                           kernel_initializer=tf.keras.initializers.glorot_uniform(42),
                           kernel_regularizer=tf.keras.regularizers.L1L2(l2=5e-6)))(x)
    x = Dense(256,
              kernel_initializer=tf.keras.initializers.he_uniform(42),
              bias_initializer=tf.keras.initializers.Constant(value=0.1))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(128,
              kernel_initializer=tf.keras.initializers.he_uniform(42),
              kernel_regularizer=tf.keras.regularizers.L1L2(l2=1e-6),
              bias_initializer=tf.keras.initializers.Constant(value=0.1))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)
    return x


""" if extra_input is not None:
    x = concatenate([Flatten()(model), extra_input], dtype=dtype)
    x = RepeatVector(lookback)(x)
else:
    x = RepeatVector(lookback)(model)
x = Bidirectional(CuDNNLSTM(128,
                            return_sequences=True,
                            kernel_regularizer=l2(1e-6),
                            # recurrent_regularizer=tensorflow.keras.regularizers.l2(1e-6),
                            kernel_initializer='glorot_normal'))(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = SpatialDropout1D(0.5)(x)
x = Bidirectional(CuDNNLSTM(128,
                            return_sequences=False,
                            kernel_regularizer=l2(1e-6),
                            # recurrent_regularizer=tensorflow.keras.regularizers.l2(1e-6),
                            kernel_initializer='glorot_normal'))(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Dense(256, kernel_initializer='glorot_normal')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Dropout(0.5)(x)
x = Dense(128, kernel_initializer='glorot_normal')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Dropout(0.5)(x)"""


def front(x_input, reshape_dim, channel_order, channel):
    x = Conv2D(8,
               (3, 7),
               padding="valid",
               kernel_initializer='glorot_normal',
               input_shape=reshape_dim,
               data_format=channel_order)(x_input)
    x = BatchNormalization(axis=channel)(x)
    x = Activation('relu')(x)
    x = Conv2D(8,
               (3, 3),
               padding="valid",
               kernel_initializer='glorot_normal',
               data_format=channel_order)(x)
    x = BatchNormalization(axis=channel)(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(3, 1),
                     padding='valid',
                     data_format=channel_order)(x)
    x = SpatialDropout2D(0.3)(x)
    x = Conv2D(int(16),
               (3, 3),
               padding="valid",
               kernel_initializer='glorot_normal',
               data_format=channel_order)(x)
    x = BatchNormalization(axis=channel)(x)
    x = Activation('relu')(x)
    x = Conv2D(int(16),
               (3, 3),
               padding="valid",
               kernel_initializer='glorot_normal',
               data_format=channel_order)(x)
    x = BatchNormalization(axis=channel)(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 1),
                     padding='valid',
                     data_format=channel_order)(x)
    x = SpatialDropout2D(0.3)(x)
    return x


def back(model, extra_input=None):
    if extra_input is not None:
        x = concatenate([model, extra_input])
        x = Dense(256, kernel_initializer='glorot_normal')(x)
    else:
        x = Dense(256, kernel_initializer='glorot_normal')(model)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    # x = Dropout(0.10)(x)
    x = Dense(128, kernel_initializer='glorot_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.25)(x)

    return x


def pretrained_time_back(model, lookback, extra_input=None, dtype=tf.float32):
    if extra_input is not None:
        x = concatenate([Flatten()(model), extra_input], dtype=dtype)
        x = RepeatVector(lookback)(x)
    else:
        x = RepeatVector(lookback)(model)
    x = Bidirectional(LSTM(128,
                           return_sequences=True,
                           kernel_regularizer=tf.keras.regularizers.L1L2(l2=1e-6),
                           # recurrent_regularizer=tf.keras.regularizers.L1L2(l2=1e-6),
                           kernel_initializer='glorot_normal'))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SpatialDropout1D(0.5)(x)
    x = Bidirectional(LSTM(128,
                           return_sequences=False,
                           kernel_regularizer=tf.keras.regularizers.L1L2(l2=1e-6),
                           # recurrent_regularizer=tf.keras.regularizers..L1L2(l2=1e-6),
                           kernel_initializer='glorot_normal'))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dense(256, kernel_initializer='glorot_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(128, kernel_initializer='glorot_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)
    return x


def pretrained_back(model, extra_input=None):
    if extra_input is not None:
        x = concatenate([model, extra_input])
        x = Dense(256, kernel_initializer='glorot_normal')(x)
    else:
        x = Dense(256, kernel_initializer='glorot_normal')(model)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    # x = Dropout(0.10)(x)
    x = Dense(128, kernel_initializer='glorot_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.25)(x)

    return x


def paper_front(x_input, reshape_dim, channel_order):
    x = Conv2D(10,
               (3, 7),
               padding="same",
               input_shape=reshape_dim,
               data_format=channel_order,
               kernel_initializer=tf.keras.initializers.VarianceScaling(scale=1.43),
               bias_initializer=tf.keras.initializers.Constant(value=0.1))(x_input)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(3, 1),
                     padding='same',
                     data_format=channel_order)(x)
    x = Conv2D(20,
               (3, 3),
               padding="same",
               data_format=channel_order,
               kernel_initializer=tf.keras.initializers.VarianceScaling(scale=1.43),
               bias_initializer=tf.keras.initializers.Constant(value=0.1))(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(3, 1),
                     padding='same',
                     data_format=channel_order)(x)
    return x


def paper_back(model, lookback, extra_input=None):
    if extra_input is not None:
        x = concatenate([Flatten()(model), extra_input])
        x = RepeatVector(lookback)(x)
    else:
        x = RepeatVector(lookback)(model)
    x = LSTM(200,
             return_sequences=True,
             kernel_initializer=tf.keras.initializers.VarianceScaling(scale=1.0),
             recurrent_initializer=tf.keras.initializers.VarianceScaling(scale=1.0),
             bias_initializer=tf.keras.initializers.Constant(value=0.0))(x)
    x = LSTM(200,
             return_sequences=False,

             kernel_initializer=tf.keras.initializers.VarianceScaling(scale=1.0),
             recurrent_initializer=tf.keras.initializers.VarianceScaling(scale=1.0),
             bias_initializer=tf.keras.initializers.Constant(value=0.0))(x)
    x = Dropout(0.5)(x)
    x = Dense(256,
              kernel_initializer=tf.keras.initializers.VarianceScaling(scale=1.15),
              bias_initializer=tf.keras.initializers.Constant(value=0.0))(x)
    x = Activation('relu')(x)
    x = Dense(128,
              kernel_initializer=tf.keras.initializers.VarianceScaling(scale=1.15),
              bias_initializer=tf.keras.initializers.Constant(value=0.0))(x)
    x = Activation('relu')(x)
    return x


"""
    x = Conv2D(16,
               (3, 7),
               padding="valid",
               kernel_initializer='glorot_normal',
               input_shape=reshape_dim,
               data_format=channel_order)(x_input)
    x = Activation('relu')(x)
    x = Conv2D(int(16),
               (3, 3),
               padding="same",
               kernel_initializer='glorot_normal',
               data_format=channel_order)(x)
    x = BatchNormalization(axis=channel)(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(3, 1),
                     padding='valid',
                     data_format=channel_order)(x)
    #x = SpatialDropout2D(0.10)(x)
    x = Conv2D(int(32),
               (3, 3),
               padding="valid",
               kernel_initializer='glorot_normal',
               data_format=channel_order)(x)
    x = Activation('relu')(x)
    x = Conv2D(int(32),
               (3, 3),
               padding="same",
               kernel_initializer='glorot_normal',
               data_format=channel_order)(x)
    x = BatchNormalization(axis=channel)(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 1),
                     padding='valid',
                     data_format=channel_order)(x)
    x = Conv2D(int(64),
               (3, 3),
               padding="same",
               kernel_initializer='glorot_normal',
               data_format=channel_order)(x)
    x = BatchNormalization(axis=channel)(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 1),
                     padding='valid',
                     data_format=channel_order)(x)
    x = Conv2D(int(128),
               (3, 3),
               padding="same",
               kernel_initializer='glorot_normal',
               data_format=channel_order)(x)
    x = BatchNormalization(axis=channel)(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 1),
                     padding='valid',
                     data_format=channel_order)(x)
    x = SpatialDropout2D(0.5)(x)
    x = Flatten()(x)
    return x
"""
