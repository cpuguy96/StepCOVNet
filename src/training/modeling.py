from __future__ import absolute_import, division, print_function, unicode_literals

import os

import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Nadam

from configuration.parameters import BATCH_SIZE, MAX_EPOCHS
from training.data_preparation import load_data, pre_process
from training.network import build_stepcovnet

os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH '] = 'true'
tf.compat.v1.disable_eager_execution()

tf.random.set_seed(42)


# tf.keras.backend.set_floatx('float16')

# disabling until more stable
# from keras_radam import RAdam
# os.environ['TF_KERAS'] = '1'


def train_model(model, features, extra_features, labels, sample_weights, class_weights, all_scalers, model_name,
                model_out_path, lookback=1):
    indices_all = range(len(features))
    print("Number of samples: %s" % len(indices_all))

    if lookback > 2:
        indices_train, indices_validation, y_train, y_train = \
            train_test_split(indices_all,
                             labels,
                             test_size=0.2,
                             shuffle=False,
                             random_state=42)
    else:
        indices_train, indices_validation, y_train, y_train = \
            train_test_split(indices_all,
                             labels,
                             test_size=0.2,
                             stratify=labels,
                             shuffle=True,
                             random_state=42)

    if extra_features is not None:
        training_extra_features = extra_features[indices_train]
        testing_extra_features = extra_features[indices_validation]
    else:
        training_extra_features = None
        testing_extra_features = None

    print("\nCreating training scalers..")

    training_scaler = []

    if len(features.shape) > 2:
        for i in range(3):
            training_scaler.append(StandardScaler().fit(features[indices_train, :, i]))
    else:
        training_scaler.append(StandardScaler().fit(features[indices_train]))

    training_callbacks = [EarlyStopping(monitor='val_loss', patience=3, verbose=0),
                          ModelCheckpoint(filepath=os.path.join(model_out_path, model_name + '_callback.h5'),
                                          monitor='val_loss',
                                          verbose=0,
                                          save_best_only=True)]

    print("\nCreating training and test sets...")

    weights = model.get_weights()

    x_train, y_train, sample_weights_train = pre_process(features[indices_train],
                                                         labels[indices_train],
                                                         training_extra_features,
                                                         sample_weights[indices_train],
                                                         training_scaler)

    x_test, y_test, sample_weights_test = pre_process(features[indices_validation],
                                                      labels[indices_validation],
                                                      testing_extra_features,
                                                      sample_weights[indices_validation],
                                                      training_scaler)

    print("\nStarting training...")

    history = model.fit(x=x_train,
                        y=y_train,
                        batch_size=BATCH_SIZE,
                        epochs=MAX_EPOCHS,
                        callbacks=training_callbacks,
                        class_weight=class_weights,
                        sample_weight=sample_weights_train,
                        validation_data=(x_test, y_test, sample_weights_test),
                        shuffle="batch",
                        verbose=1)

    print("\n*****************************")
    print("***** TRAINING FINISHED *****")
    print("*****************************\n")

    model.save(os.path.join(model_out_path, model_name + ".h5"))

    callbacks = [  # ModelCheckpoint(
        # filepath=os.path.join(model_out_path, prefix + 'retrained_callback_timing_model.h5'),
        # monitor='loss',
        # verbose=0,
        # save_best_only=True)
    ]
    # train again use all train and validation set
    epochs_final = len(history.history['val_loss'])

    print("\nUsing entire dataset for training...")

    all_x, all_y, sample_weights_all = pre_process(features,
                                                   labels,
                                                   extra_features,
                                                   sample_weights,
                                                   all_scalers)

    model.set_weights(weights)

    print("\nStarting retraining...")

    model.fit(x=all_x,
              y=all_y,
              batch_size=BATCH_SIZE,
              epochs=epochs_final,
              callbacks=callbacks,
              sample_weight=sample_weights_all,
              class_weight=class_weights,
              shuffle="batch",
              verbose=1)

    print("\n*******************************")
    print("***** RETRAINING FINISHED *****")
    print("*******************************\n")

    model.save(os.path.join(model_out_path, model_name + "_retrained.h5"))


def prepare_model(filename_features, filename_labels, filename_sample_weights, filename_scaler, input_shape, model_name,
                  model_out_path, extra_input_shape, path_extra_features, lookback, limit=-1,
                  filename_pretrained_model=None):
    print("Loading data...")
    features, extra_features, labels, sample_weights, class_weights, all_scalers, pretrained_model = \
        load_data(filename_features, path_extra_features, filename_labels, filename_sample_weights, filename_scaler,
                  filename_pretrained_model)

    if limit > 0:
        features = features[:limit]
        if extra_features is not None:
            extra_features = extra_features[:limit]
        labels = labels[:limit]
        sample_weights = sample_weights[:limit]

        if labels.sum() == 0:
            raise ValueError("Not enough positive labels. Increase limit!")

    timeseries = True if lookback > 1 else False

    print("Building StepCOVNet...")
    model = build_stepcovnet(input_shape, timeseries, extra_input_shape, pretrained_model)

    model.compile(loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.15),
                  optimizer=Nadam(beta_1=0.99),
                  metrics=["accuracy"])

    print(model.summary())

    train_model(model, features, extra_features, labels, sample_weights, class_weights, all_scalers, model_name,
                model_out_path, lookback)
