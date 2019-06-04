import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.callbacks import EarlyStopping, CSVLogger, LearningRateScheduler, ModelCheckpoint, ReduceLROnPlateau
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dropout, Dense, Flatten, BatchNormalization
from keras.models import Sequential
from keras.optimizers import Adam, Nadam

from feature_generator import generator
from data_preparation import load_data


def front_end_a(model_1,
                filter_density,
                reshape_dim,
                channel_order,
                dropout):
    model_1.add(Conv2D(int(8 * filter_density),
                       (3, 7),
                       padding="valid",
                       kernel_initializer='glorot_normal',
                       input_shape=reshape_dim,
                       data_format=channel_order,
                       activation='relu'))
    model_1.add(BatchNormalization(axis=1))
    model_1.add(MaxPooling2D(pool_size=(3, 1),
                             padding='valid',
                             data_format=channel_order))

    model_1.add(Conv2D(int(16 * filter_density),
                       (3, 3),
                       padding="valid",
                       kernel_initializer='glorot_normal',
                       data_format=channel_order,
                       activation='relu'))
    model_1.add(BatchNormalization(axis=1))
    model_1.add(MaxPooling2D(pool_size=(3, 1),
                             padding='valid',
                             data_format=channel_order))

    if dropout:
        model_1.add(Dropout(dropout))

    return model_1


def auc(y_true, y_pred):
    auc = tf.metrics.auc(y_true, y_pred)[1]
    K.get_session().run(tf.local_variables_initializer())
    return auc


def jan_original_5_layers_cnn(filter_density,
                              dropout,
                              input_shape,
                              channel=1):
    "less deep architecture"
    if channel == 1:
        reshape_dim = (1, input_shape[0], input_shape[1])
        channel_order = 'channels_first'
    else:
        reshape_dim = input_shape
        channel_order = 'channels_last'

    model = Sequential()

    model = front_end_a(model_1=model,
                          filter_density=filter_density,
                          reshape_dim=reshape_dim,
                          channel_order=channel_order,
                          dropout=dropout)
    model.add(BatchNormalization(axis=1))
    model.add(Flatten())
    model.add(Dense(512, activation="relu", kernel_initializer='glorot_normal'))
    model.add(BatchNormalization())

    if dropout:
        model.add(Dropout(dropout))

    model.add(Dense(1, activation='sigmoid'))

    optimizer = Nadam()

    model.compile(loss='binary_crossentropy',
                    optimizer=optimizer,
                    metrics=["accuracy", auc])

    print(model.summary())

    return model


def model_train(model_0,
                batch_size,
                path_feature_data,
                indices_all,
                Y_train_validation,
                sample_weights,
                class_weights,
                file_path_model,
                filename_log,
                channel):

    print("start training...")

    model_0.save_weights(file_path_model)

    callbacks = [EarlyStopping(monitor='val_loss', patience=10, verbose=0),
                 ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001),
                 CSVLogger(filename=filename_log, separator=';')]

    from sklearn.model_selection import train_test_split

    weights_index_train, weights_index_val, Y_train, Y_validation = \
        train_test_split(np.asarray([sample_weights, indices_all]).T,
                         Y_train_validation,
                         test_size=0.2,
                         random_state=42,
                         stratify=Y_train_validation)

    indices_train = [int(xt[1]) for xt in weights_index_train]
    sample_weights_train = np.array([xt[0] for xt in weights_index_train])
    indices_validation = [int(xv[1]) for xv in weights_index_val]
    sample_weights_validation = np.array([xv[0] for xv in weights_index_val])

    steps_per_epoch_train = int(np.ceil(len(indices_train) / batch_size))
    steps_per_epoch_val = int(np.ceil(len(indices_validation) / batch_size))

    generator_train = generator(path_feature_data=path_feature_data,
                                indices=indices_train,
                                number_of_batches=steps_per_epoch_train,
                                file_size=batch_size,
                                labels=Y_train,
                                shuffle=False,
                                sample_weights=sample_weights_train,
                                multi_inputs=False)
    generator_val = generator(path_feature_data=path_feature_data,
                              indices=indices_validation,
                              number_of_batches=steps_per_epoch_val,
                              file_size=batch_size,
                              labels=Y_validation,
                              shuffle=False,
                              sample_weights=sample_weights_validation,
                              multi_inputs=False)

    history = model_0.fit_generator(generator=generator_train,
                                    steps_per_epoch=steps_per_epoch_train,
                                    epochs=200,
                                    validation_data=generator_val,
                                    validation_steps=steps_per_epoch_val,
                                    class_weight=class_weights,
                                    callbacks=callbacks,
                                    verbose=2)

    model_0.load_weights(file_path_model)

    callbacks = [ReduceLROnPlateau(monitor='loss', factor=0.2, patience=5, min_lr=0.001),
                 CSVLogger(filename=filename_log, separator=';')]
    # train again use all train and validation set
    epochs_final = len(history.history['val_loss'])

    steps_per_epoch_train_val = int(np.ceil(len(indices_all) / batch_size))

    generator_train_val = generator(path_feature_data=path_feature_data,
                                    indices=indices_all,
                                    number_of_batches=steps_per_epoch_train_val,
                                    file_size=batch_size,
                                    labels=Y_train_validation,
                                    shuffle=False,
                                    sample_weights=sample_weights,
                                    multi_inputs=False,
                                    channel=channel)

    model_0.fit_generator(generator=generator_train_val,
                          steps_per_epoch=steps_per_epoch_train_val,
                          epochs=epochs_final,
                          callbacks=callbacks,
                          class_weight=class_weights,
                          verbose=2)

    model_0.save(file_path_model)


def train_model(filename_train_validation_set,
                filename_labels_train_validation_set,
                filename_sample_weights,
                filter_density,
                dropout,
                input_shape,
                file_path_model,
                filename_log,
                channel=1):
    """
    train final model save to model path
    """

    filenames_features, Y_train_validation, sample_weights, class_weights = \
        load_data(filename_labels_train_validation_set,
                       filename_sample_weights)
    model = jan_original_5_layers_cnn  (filter_density=filter_density,
                                        dropout=dropout,
                                        input_shape=input_shape,
                                        channel=channel)
    batch_size = 256

    model_train(model,
                 batch_size,
                 filename_train_validation_set,
                 filenames_features,
                 Y_train_validation,
                 sample_weights,
                 class_weights,
                 file_path_model,
                 filename_log,
                 channel)
