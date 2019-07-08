from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Flatten, BatchNormalization, Input
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.optimizers import Nadam

from training_scripts.feature_generator import generator
from training_scripts.data_preparation import load_data
from src.resnet_blocks import *


def front_end_a(model,
                reshape_dim,
                channel_order,
                channel):
    model.add(Conv2D(int(8),
                       (3, 7),
                       padding="same",
                       kernel_initializer='glorot_normal',
                       input_shape=reshape_dim,
                       data_format=channel_order,
                       activation='relu'))
    model.add(Conv2D(int(8),
                     (3, 7),
                     padding="same",
                     kernel_initializer='glorot_normal',
                     data_format=channel_order,
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 1),
                             padding='valid',
                             data_format=channel_order))
    model.add(BatchNormalization(axis=channel))
    model.add(Dropout(0.5))
    model.add(Conv2D(int(16),
                       (3, 3),
                       padding="same",
                       kernel_initializer='glorot_normal',
                       data_format=channel_order,
                       activation='relu'))
    model.add(Conv2D(int(16),
                     (3, 3),
                     padding="same",
                     kernel_initializer='glorot_normal',
                     data_format=channel_order,
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 1),
                             padding='valid',
                             data_format=channel_order))
    model.add(BatchNormalization(axis=channel))
    model.add(Dropout(0.25))
    return model


def front_end_a_fun(x_input, reshape_dim, channel_order, channel):
    X = Conv2D(8,
               (3, 7),
               padding="same",
               kernel_initializer='glorot_normal',
               input_shape=reshape_dim,
               data_format=channel_order,
               activation='relu')(x_input)
    X = BatchNormalization(axis=channel)(X)
    X = Conv2D(8,
               (3, 7),
               padding="same",
               kernel_initializer='glorot_normal',
               data_format=channel_order,
               activation='relu')(X)
    X = BatchNormalization(axis=channel)(X)
    X = MaxPooling2D(pool_size=(3, 1),
                     padding='valid',
                     data_format=channel_order)(X)
    X = Dropout(0.25)(X)
    X = Conv2D(int(16),
               (3, 3),
               padding="same",
               kernel_initializer='glorot_normal',
               data_format=channel_order,
               activation='relu')(X)
    X = BatchNormalization(axis=channel)(X)
    X = Conv2D(int(16),
               (3, 3),
               padding="same",
               kernel_initializer='glorot_normal',
               data_format=channel_order,
               activation='relu')(X)
    X = BatchNormalization(axis=channel)(X)
    X = MaxPooling2D(pool_size=(3, 1),
                     padding='valid',
                     data_format=channel_order)(X)
    X = Dropout(0.25)(X)
    return X


def front_end_b(model,
                reshape_dim,
                channel_order,
                channel):
    model.add(Conv2D(10,
                       (3, 7),
                       padding="valid",
                       kernel_initializer='glorot_normal',
                       input_shape=reshape_dim,
                       data_format=channel_order,
                       activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 1),
                           padding='valid',
                           data_format=channel_order))
   # model.add(BatchNormalization(axis=channel))
    #model.add(Dropout(0.25))
    model.add(Conv2D(20,
                     (3, 3),
                     padding="valid",
                     kernel_initializer='glorot_normal',
                     data_format=channel_order,
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 1),
                           padding='valid',
                           data_format=channel_order))
    #model.add(BatchNormalization(axis=channel))
    model.add(Dropout(0.5))
    return model


def front_end_b_fun(x_input, reshape_dim, channel_order, channel):
    model = Conv2D(10,
                   (3, 7),
                   padding="valid",
                   kernel_initializer='glorot_normal',
                   input_shape=reshape_dim,
                   data_format=channel_order,
                   activation='relu')(x_input)
    model = BatchNormalization(axis=channel)(model)
    model = MaxPooling2D(pool_size=(3, 1),
                         padding='valid',
                         data_format=channel_order)(model)
    model = Conv2D(20,
                   (3, 3),
                   padding="valid",
                   kernel_initializer='glorot_normal',
                   data_format=channel_order,
                   activation='relu')(model)
    model = BatchNormalization(axis=channel)(model)
    model = MaxPooling2D(pool_size=(3, 1),
                         padding='valid',
                         data_format=channel_order)(model)
    model = BatchNormalization(axis=channel)(model)
    model = Dropout(0.5)(model)
    return model


def back_end_a_fun(model):
    X = Dense(256, activation='relu', kernel_initializer='glorot_normal')(model)
    X = BatchNormalization()(X)
    X = Dropout(0.25)(X)

    return X


def back_end_b_fun(model):
    X = Dense(256, activation='relu', kernel_initializer='glorot_normal')(model)
    X = BatchNormalization()(X)
    X = Dense(128, activation='relu', kernel_initializer='glorot_normal')(X)
    X = Dropout(0.5)(X)

    return X


def back_end_c(model, channel_order, channel):
    model.add(Conv2D(int(60), (3, 3), padding="same",
                       data_format=channel_order, activation='relu'))
    model.add(BatchNormalization(axis=channel))

    model.add(Conv2D(int(60), (3, 3), padding="same",
                       data_format=channel_order, activation='relu'))
    model.add(BatchNormalization(axis=channel))

    model.add(Conv2D(int(60), (3, 3), padding="same",
                       data_format=channel_order, activation='relu'))
    model.add(BatchNormalization(axis=channel))
    model.add(Dropout(0.5))

    return model


def back_end_c_fun(model, channel_order, channel):
    model = Conv2D(int(60), (3, 3), padding="same",
                       data_format=channel_order, activation='relu')(model)
    model = BatchNormalization(axis=channel)(model)
    model = Conv2D(int(60), (3, 3), padding="same",
                       data_format=channel_order, activation='relu')(model)
    model = BatchNormalization(axis=channel)(model)
    model = Conv2D(int(60), (3, 3), padding="same",
                       data_format=channel_order, activation='relu')(model)
    model = BatchNormalization(axis=channel)(model)
    model = Dropout(0.5)(model)

    return model


def resnet_model(model, channel):
    model = convolutional_block(model, f=3, filters=[16, 16, 64], stage = 2, block='a', s=1, channel=channel)
    model = identity_block(model, 3, [16, 16, 64], stage=2, block='b', channel=channel)
    model = identity_block(model, 3, [16, 16, 64], stage=2, block='c', channel=channel)

    model = convolutional_block(model, f=3, filters=[32, 32, 128], stage=3, block='a', s=1, channel=channel)
    model = identity_block(model, 3, [32, 32, 128], stage=3, block='b', channel=channel)
    model = identity_block(model, 3, [32, 32, 128], stage=3, block='c', channel=channel)
    model = identity_block(model, 3, [32, 32, 128], stage=3, block='d', channel=channel)

    model = convolutional_block(model, f=3, filters=[64, 64, 256], stage=4, block='a', s=1, channel=channel)
    model = identity_block(model, 3, [64, 64, 256], stage=4, block='b', channel=channel)
    model = identity_block(model, 3, [64, 64, 256], stage=4, block='c', channel=channel)
    model = identity_block(model, 3, [64, 64, 256], stage=4, block='d', channel=channel)

    return model


def build_model(input_shape, channel=1):
    if channel == 1:
        reshape_dim = (1, input_shape[0], input_shape[1])
        channel_order = 'channels_first'
    else:
        reshape_dim = input_shape
        channel_order = 'channels_last'

    x_input = Input(reshape_dim)
    X = front_end_a_fun(x_input, reshape_dim=reshape_dim,
                        channel_order=channel_order,
                        channel=channel)

    X = Flatten()(X)
    X = back_end_a_fun(X)
    X = Dense(1, activation='sigmoid')(X)

    model = Model(inputs=x_input, outputs=X, name='StepNet')

    optimizer = Nadam()

    model.compile(loss='binary_crossentropy',
                  optimizer=optimizer,
                  metrics=["accuracy"])
    print(model.summary())

    return model


def build_pretrained_model(pretrained_model):
    model = load_model(pretrained_model)

    for layer in model.layers:
        layer.trainable = False
    for layer in model.layers[::-1]:
        if isinstance(layer, tf.keras.layers.core.Flatten):
            layer.trainable = True
            break
        model.pop()

    model.add(Dense(256, activation="relu", kernel_initializer='glorot_normal'))
    model.add(Dense(128, activation="relu", kernel_initializer='glorot_normal'))
    #model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(loss='binary_crossentropy',
                  optimizer=Nadam(),
                  metrics=["accuracy"])

    print(model.summary())
    return model


def model_train(model_0,
                batch_size,
                path_feature_data,
                indices_all,
                Y_train_validation,
                sample_weights,
                class_weights,
                scaler,
                file_path_model,
                filename_log,
                channel,
                input_shape,
                pretrained_model=None,
                is_pretrained=False):

    print("start training...")

    callbacks = [EarlyStopping(monitor='val_loss', patience=8, verbose=0),
                 # ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=4, min_lr=0.001)
                 ]

    from sklearn.model_selection import train_test_split

    indices_train, indices_validation, Y_train, Y_validation = \
        train_test_split(indices_all,
                         Y_train_validation,
                         test_size=0.2,
                         random_state=42,
                         stratify=Y_train_validation)

    sample_weights_train = sample_weights[indices_train]
    sample_weights_validation = sample_weights[indices_validation]

    steps_per_epoch_train = int(np.ceil(len(indices_train) / batch_size))
    steps_per_epoch_val = int(np.ceil(len(indices_validation) / batch_size))

    training_scaler = []

    from sklearn.preprocessing import StandardScaler
    with open(path_feature_data, 'rb') as f:
        training_data = np.load(f)['features']

    if channel != 1:
        for i in range(channel):
            training_scaler.append(StandardScaler().fit(training_data[:, :, i]))
    else:
        training_scaler.append(StandardScaler().fit(training_data))

    if channel != 1:
        multi_inputs = True
    else:
        multi_inputs = False

    generator_train = generator(path_feature_data=path_feature_data,
                                indices=indices_train,
                                number_of_batches=steps_per_epoch_train,
                                file_size=batch_size,
                                labels=Y_train,
                                shuffle=False,
                                sample_weights=sample_weights_train,
                                multi_inputs=multi_inputs,
                                scaler=training_scaler,
                                channel=channel)
    generator_val = generator(path_feature_data=path_feature_data,
                              indices=indices_validation,
                              number_of_batches=steps_per_epoch_val,
                              file_size=batch_size,
                              labels=Y_validation,
                              shuffle=False,
                              sample_weights=sample_weights_validation,
                              multi_inputs=multi_inputs,
                              scaler=training_scaler,
                              channel=channel)

    history = model_0.fit_generator(generator=generator_train,
                                    steps_per_epoch=steps_per_epoch_train,
                                    epochs=30,
                                    validation_data=generator_val,
                                    validation_steps=steps_per_epoch_val,
                                    class_weight=class_weights,
                                    callbacks=callbacks,
                                    verbose=2)

    model_0.save(file_path_model)

    callbacks = [# ReduceLROnPlateau(monitor='loss', factor=0.2, patience=5, min_lr=0.001)
                 ]
    # train again use all train and validation set
    epochs_final = len(history.history['val_loss'])

    steps_per_epoch_train_val = int(np.ceil(len(indices_all) / batch_size))

    if is_pretrained:
        model = build_pretrained_model(pretrained_model)
    else:
        model = build_model(input_shape=input_shape, channel=channel)

    generator_train_val = generator(path_feature_data=path_feature_data,
                                    indices=indices_all,
                                    number_of_batches=steps_per_epoch_train_val,
                                    file_size=batch_size,
                                    labels=Y_train_validation,
                                    shuffle=False,
                                    sample_weights=sample_weights,
                                    multi_inputs=multi_inputs,
                                    channel=channel,
                                    scaler=scaler)

    model.fit_generator(generator=generator_train_val,
                        steps_per_epoch=steps_per_epoch_train_val,
                        epochs=epochs_final,
                        callbacks=callbacks,
                        class_weight=class_weights,
                        verbose=2)

    model.save(file_path_model)


def train_model(filename_train_validation_set,
                filename_labels_train_validation_set,
                filename_sample_weights,
                filename_scaler,
                input_shape,
                file_path_model,
                filename_log,
                channel=1,
                pretrained_model=None):
    """
    train final model save to model path
    """

    filenames_features, Y_train_validation, sample_weights, class_weights, scaler = \
        load_data(filename_labels_train_validation_set, filename_sample_weights, filename_scaler)

    is_pretrained = False
    batch_size = 256

    if pretrained_model is not None:
        model = build_pretrained_model(pretrained_model)
        is_pretrained = True
    else:
        model = build_model(input_shape=input_shape, channel=channel)

    model_train(model,
                batch_size,
                filename_train_validation_set,
                filenames_features,
                Y_train_validation,
                sample_weights,
                class_weights,
                scaler,
                file_path_model,
                filename_log,
                channel,
                input_shape,
                pretrained_model=pretrained_model,
                is_pretrained=is_pretrained)
