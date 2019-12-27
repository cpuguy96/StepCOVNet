from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Flatten, BatchNormalization, \
    Activation, SpatialDropout2D, concatenate, Bidirectional, RepeatVector, SpatialDropout1D

from tensorflow.compat.v1.keras.layers import CuDNNLSTM
from tensorflow.keras.regularizers import l2


def time_front(x_input, reshape_dim, channel_order, channel):
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
    return x


def time_back(model, lookback, extra_input=None):
    if extra_input is not None:
        x = concatenate([Flatten()(model), extra_input], dtype="float16")
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
    x = Dropout(0.5)(x)
    return x


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


def pretrained_time_back(model, lookback, extra_input=None):
    if extra_input is not None:
        x = concatenate([Flatten()(model), extra_input], dtype="float16")
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
