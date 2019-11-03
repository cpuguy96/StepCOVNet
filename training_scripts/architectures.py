from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Flatten, BatchNormalization, \
    Input, Activation, Add, AveragePooling2D, ConvLSTM2D, Reshape, SpatialDropout2D, GlobalMaxPooling2D


def front(x_input, reshape_dim, channel_order, channel):
    x = Conv2D(16,
               (3, 7),
               padding="valid",
               kernel_initializer='glorot_normal',
               input_shape=reshape_dim,
               data_format=channel_order)(x_input)
    x = BatchNormalization(axis=channel)(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(3, 1),
                     padding='valid',
                     data_format=channel_order)(x)
    x = SpatialDropout2D(0.25)(x)
    x = Conv2D(int(32),
               (3, 3),
               padding="valid",
               kernel_initializer='glorot_normal',
               data_format=channel_order)(x)
    x = BatchNormalization(axis=channel)(x)
    x = MaxPooling2D(pool_size=(2, 1),
                     padding='valid',
                     data_format=channel_order)(x)
    x = SpatialDropout2D(0.25)(x)
    x = Conv2D(int(64),
               (3, 3),
               padding="valid",
               kernel_initializer='glorot_normal',
               data_format=channel_order)(x)
    x = BatchNormalization(axis=channel)(x)
    x = MaxPooling2D(pool_size=(2, 1),
                     padding='valid',
                     data_format=channel_order)(x)
    #x = SpatialDropout2D(0.25)(x)
    return x


def back(model):
    X = Dense(256, kernel_initializer='glorot_normal')(model)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = Dropout(0.15)(X)
    X = Dense(128, kernel_initializer='glorot_normal')(X)
    X = Activation('relu')(X)
    X = Dropout(0.15)(X)

    return X


def front_end_a_fun(x_input, reshape_dim, channel_order, channel):
    X = Conv2D(8,
               (3, 7),
               padding="valid",
               kernel_initializer='glorot_normal',
               input_shape=reshape_dim,
               data_format=channel_order)(x_input)
    X = BatchNormalization(axis=channel)(X)
    X = Activation('relu')(X)
    X = Conv2D(8,
               (3, 7),
               padding="valid",
               kernel_initializer='glorot_normal',
               data_format=channel_order)(X)
    X = BatchNormalization(axis=channel)(X)
    X = Activation('relu')(X)
    X = MaxPooling2D(pool_size=(3, 1),
                     padding='valid',
                     data_format=channel_order)(X)
    X = SpatialDropout2D(0.25)(X)
    X = Conv2D(int(16),
               (3, 3),
               padding="same",
               kernel_initializer='glorot_normal',
               data_format=channel_order)(X)
    X = BatchNormalization(axis=channel)(X)
    X = Conv2D(int(16),
               (3, 3),
               padding="same",
               kernel_initializer='glorot_normal',
               data_format=channel_order,
               activation='relu')(X)
    X = BatchNormalization(axis=channel)(X)
    X = Activation('relu')(X)
    X = MaxPooling2D(pool_size=(3, 1),
                     padding='valid',
                     data_format=channel_order)(X)
    X = SpatialDropout2D(0.25)(X)
    return X


def front_end_b_fun(x_input, reshape_dim, channel_order, channel):
    model = Conv2D(10,
                   (3, 7),
                   padding="valid",
                   kernel_initializer='glorot_normal',
                   input_shape=reshape_dim,
                   data_format=channel_order)(x_input)
    model = BatchNormalization(axis=channel)(model)
    model = Activation('relu')(model)
    model = MaxPooling2D(pool_size=(3, 1),
                         padding='valid',
                         data_format=channel_order)(model)
    model = SpatialDropout2D(0.5)(model)

    model = Conv2D(20,
                   (3, 3),
                   padding="valid",
                   kernel_initializer='glorot_normal',
                   data_format=channel_order)(model)
    model = BatchNormalization(axis=channel)(model)
    model = Activation('relu')(model)
    model = MaxPooling2D(pool_size=(3, 1),
                         padding='valid',
                         data_format=channel_order)(model)
    model = SpatialDropout2D(0.5)(model)

    return model


def front_c(x_input, reshape_dim, channel_order, channel):
    x = Conv2D(16,
               (3, 7),
               padding="valid",
               kernel_initializer='glorot_normal',
               input_shape=reshape_dim,
               data_format=channel_order)(x_input)
    x = BatchNormalization(axis=channel)(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(3, 1),
                     padding='valid',
                     data_format=channel_order)(x)
    x = SpatialDropout2D(0.25)(x)
    x = Conv2D(int(32),
               (3, 3),
               padding="valid",
               kernel_initializer='glorot_normal',
               data_format=channel_order)(x)
    x = BatchNormalization(axis=channel)(x)
    x = MaxPooling2D(pool_size=(2, 1),
                     padding='valid',
                     data_format=channel_order)(x)
    x = SpatialDropout2D(0.25)(x)
    x = Conv2D(int(64),
               (3, 3),
               padding="valid",
               kernel_initializer='glorot_normal',
               data_format=channel_order)(x)
    x = BatchNormalization(axis=channel)(x)
    x = MaxPooling2D(pool_size=(2, 1),
                     padding='valid',
                     data_format=channel_order)(x)
    x = SpatialDropout2D(0.25)(x)
    return x


def back_end_a_fun(model):
    X = Dense(256, kernel_initializer='glorot_normal')(model)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = Dropout(0.5)(X)

    return X


def back_end_b_fun(model):
    X = Dense(256, kernel_initializer='glorot_normal')(model)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = Dropout(0.5)(X)
    X = Dense(128, kernel_initializer='glorot_normal')(X)
    X = Activation('relu')(X)
    X = Dropout(0.5)(X)

    return X


def back_end_c_fun(model, channel_order, channel):
    model = Conv2D(int(60), (3, 3), padding="same",
                   data_format=channel_order)(model)
    model = BatchNormalization(axis=channel)(model)
    model = Activation('relu')(model)
    model = Conv2D(int(60), (3, 3), padding="same",
                   data_format=channel_order)(model)
    model = BatchNormalization(axis=channel)(model)
    model = Activation('relu')(model)
    model = Conv2D(int(60), (3, 3), padding="same",
                   data_format=channel_order)(model)
    model = BatchNormalization(axis=channel)(model)
    model = Activation('relu')(model)
    model = SpatialDropout2D(0.5)(model)

    return model
