from __future__ import absolute_import, division, print_function, unicode_literals

import os
os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'


import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import Dense, Flatten, Input, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Nadam
tf.compat.v1.disable_eager_execution()

# disabling until more stable
# from keras_radam import RAdam
# os.environ['TF_KERAS'] = '1'

from training_scripts.architectures import *

from training_scripts.feature_generator import generator
from training_scripts.data_preparation import load_data


def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


def build_pretrained_model(pretrained_model, silent=False):
    model = load_model(pretrained_model, custom_objects={'f1': f1})

    for layer in model.layers:
        layer.trainable = False
    for layer in model.layers[::-1]:
        layer.trainable = True
        if isinstance(layer, tf.keras.layers.Flatten):
            layer.trainable = True
            break
    #    model.layers.pop()
    # x_input = Input(model.)
    # X = Dense(256, kernel_initializer='glorot_normal')(model.outputs)
    # X = BatchNormalization()(X)
    # X = Activation('relu')(X)
    # X = Dropout(0.5)(X)
    # X = Dense(128, kernel_initializer='glorot_normal')(X)
    # X = Activation('relu')(X)
    # X = Dropout(0.5)(X)
    # X = Dense(1, activation="sigmoid")(X)
    # model = Model(inputs=x_input, outputs=X, name='StepNet')
    model.compile(loss='binary_crossentropy',
                  optimizer=Nadam(),
                  metrics=["accuracy", f1])

    if not silent:
        print(model.summary())

    return model


def build_model(input_shape, channel=1, silent=False):
    if channel == 1:
        reshape_dim = (1, input_shape[0], input_shape[1])
        channel_order = 'channels_first'
    else:
        reshape_dim = input_shape
        channel_order = 'channels_last'

    x_input = Input(reshape_dim)
    # x = front_end_a_fun(x_input, reshape_dim=reshape_dim,
    #                    channel_order=channel_order,
    #                   channel=channel)
    # x = back_end_c_fun(x, channel_order, channel)
    x = new_front(x_input,
                  reshape_dim=reshape_dim,
                  channel_order=channel_order,
                  channel=channel)
    x = Flatten()(x)
    x = new_back(x)

    x = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=x_input, outputs=x, name='StepNet')

    optimizer = Nadam()

    model.compile(loss='binary_crossentropy',
                  optimizer=optimizer,
                  metrics=["accuracy", f1])

    if not silent:
        print(model.summary())

    return model


def model_train(model_0,
                batch_size,
                path_feature_data,
                indices_all,
                all_labels,
                sample_weights,
                class_weights,
                scaler,
                file_path_model,
                channel,
                input_shape,
                pretrained_model=None,
                is_pretrained=False):
    print("start training...")

    callbacks = [EarlyStopping(monitor='val_loss', patience=5, verbose=0)]
    from sklearn.model_selection import train_test_split

    print("number of samples:", len(indices_all))

    indices_train, indices_validation, y_train, y_val = \
        train_test_split(indices_all,
                         all_labels,
                         test_size=0.2,
                         random_state=42,
                         # shuffle=False,
                         stratify=all_labels)

    sample_weights_train = sample_weights[indices_train]
    sample_weights_validation = sample_weights[indices_validation]

    steps_per_epoch_train = int(np.ceil(len(indices_train) / batch_size))
    steps_per_epoch_val = int(np.ceil(len(indices_validation) / batch_size))

    print("loading training data")

    from sklearn.preprocessing import StandardScaler
    with open(path_feature_data, 'rb') as f:
        training_data = np.load(f)['features'][indices_train]

    training_scaler = []

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
                                labels=y_train,
                                shuffle=False,
                                sample_weights=sample_weights_train,
                                multi_inputs=multi_inputs,
                                scaler=training_scaler,
                                channel=channel)
    generator_val = generator(path_feature_data=path_feature_data,
                              indices=indices_validation,
                              number_of_batches=steps_per_epoch_val,
                              file_size=batch_size,
                              labels=y_val,
                              shuffle=False,
                              sample_weights=sample_weights_validation,
                              multi_inputs=multi_inputs,
                              scaler=training_scaler,
                              channel=channel)

    history = model_0.fit(generator_train,
                          steps_per_epoch=steps_per_epoch_train,
                          epochs=300,
                          validation_data=generator_val,
                          validation_steps=steps_per_epoch_val,
                          class_weight=class_weights,
                          callbacks=callbacks,
                          verbose=1)

    model_0.save(os.path.join(file_path_model, "trained_model.h5"))

    callbacks = []
    # train again use all train and validation set
    epochs_final = len(history.history['val_loss'])

    steps_per_epoch_train_val = int(np.ceil(len(indices_all) / batch_size))

    if is_pretrained:
        model = build_pretrained_model(pretrained_model, silent=True)
    else:
        model = build_model(input_shape=input_shape, channel=channel, silent=True)

    generator_train_val = generator(path_feature_data=path_feature_data,
                                    indices=indices_all,
                                    number_of_batches=steps_per_epoch_train_val,
                                    file_size=batch_size,
                                    labels=all_labels,
                                    shuffle=False,
                                    sample_weights=sample_weights,
                                    multi_inputs=multi_inputs,
                                    channel=channel,
                                    scaler=scaler)

    model.fit(generator_train_val,
              steps_per_epoch=steps_per_epoch_train_val,
              epochs=epochs_final,
              callbacks=callbacks,
              class_weight=class_weights,
              verbose=1)

    model.save(os.path.join(file_path_model, "retrained_model.h5"))


def train_model(filename_train_validation_set,
                filename_labels_train_validation_set,
                filename_sample_weights,
                filename_scaler,
                input_shape,
                file_path_model,
                channel=1,
                pretrained_model=None):
    """
    train final model save to model path
    """

    filenames_features, Y_train_validation, sample_weights, class_weights, scaler = \
        load_data(filename_labels_train_validation_set, filename_sample_weights, filename_scaler)

    is_pretrained = False
    batch_size = 128

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
                channel,
                input_shape,
                pretrained_model=pretrained_model,
                is_pretrained=is_pretrained)
