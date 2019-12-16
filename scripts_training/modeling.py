from __future__ import absolute_import, division, print_function, unicode_literals

from scripts_training.data_preparation import load_data, preprocess
from scripts_training.network import build_stepcovnet


from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from tensorflow.keras.optimizers import Nadam

import os
import tensorflow as tf

os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH '] = 'true'
tf.compat.v1.disable_eager_execution()

tf.random.set_seed(42)
#tf.keras.backend.set_floatx('float16')

# disabling until more stable
# from keras_radam import RAdam
# os.environ['TF_KERAS'] = '1'


def build_pretrained_model(pretrained_model, silent=False):
    model = load_model(pretrained_model, custom_objects={}, compile=False)

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
                  optimizer=Nadam(beta_1=0.99),
                  metrics=["accuracy"])

    if not silent:
        print(model.summary())

    return model


def model_train(model,
                batch_size,
                max_epochs,
                features,
                extra_features,
                labels,
                sample_weights,
                class_weights,
                all_scalers,
                prefix,
                model_out_path,
                lookback=1):

    training_callbacks = [EarlyStopping(monitor='val_loss', patience=3, verbose=0),
                          ModelCheckpoint(filepath=os.path.join(model_out_path, prefix + '_callback.h5'),
                                          monitor='val_loss',
                                          verbose=0,
                                          save_best_only=True)]
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    indices_all = range(len(features))
    print("number of samples:", len(indices_all))

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

    training_scaler = []

    if len(features.shape) > 2:
        for i in range(3):
            training_scaler.append(StandardScaler().fit(features[indices_train, :, i]))
    else:
        training_scaler.append(StandardScaler().fit(features[indices_train]))

    print(model.summary())

    print("\nStarting training...")

    weights = model.get_weights()

    x_train, y_train, sample_weights_train = preprocess(features[indices_train],
                                                        labels[indices_train],
                                                        training_extra_features,
                                                        sample_weights[indices_train],
                                                        training_scaler)

    x_test, y_test, sample_weights_test = preprocess(features[indices_validation],
                                                     labels[indices_validation],
                                                     testing_extra_features,
                                                     sample_weights[indices_validation],
                                                     training_scaler)

    history = model.fit(x=x_train,
                        y=y_train,
                        batch_size=batch_size,
                        epochs=max_epochs,
                        callbacks=training_callbacks,
                        class_weight=class_weights,
                        sample_weight=sample_weights_train,
                        validation_data=(x_test, y_test, sample_weights_test),
                        shuffle="batch",
                        verbose=1)

    model.save(os.path.join(model_out_path, prefix + ".h5"))

    print("\n*****************************")
    print("***** TRAINING FINISHED *****")
    print("*****************************\n")

    callbacks = [# ModelCheckpoint(
                 # filepath=os.path.join(model_out_path, prefix + 'retrained_callback_timing_model.h5'),
                 # monitor='loss',
                 # verbose=0,
                 # save_best_only=True)
                ]
    # train again use all train and validation set
    epochs_final = len(history.history['val_loss'])

    all_x, all_y, sample_weights_all = preprocess(features,
                                                  labels,
                                                  extra_features,
                                                  sample_weights,
                                                  all_scalers)

    model.set_weights(weights)

    print("Starting retraining...")

    history = model.fit(x=all_x,
                        y=all_y,
                        batch_size=batch_size,
                        epochs=epochs_final,
                        callbacks=callbacks,
                        sample_weight=sample_weights_all,
                        class_weight=class_weights,
                        shuffle="batch",
                        verbose=1)

    model.save(os.path.join(model_out_path, prefix + "_retrained.h5"))


def train_model(filename_features,
                filename_labels,
                filename_sample_weights,
                filename_scaler,
                input_shape,
                prefix,
                model_out_path,
                extra_input_shape,
                path_extra_features,
                lookback,
                limit=-1,
                filename_pretrained_model=None):

    print("Loading data...")
    features, extra_features, labels, sample_weights, class_weights, all_scalers, pretrained_model = \
        load_data(filename_features, path_extra_features, filename_labels, filename_sample_weights, filename_scaler, filename_pretrained_model)

    if limit > 0:
        features = features[:limit]
        if extra_features is not None:
            extra_features = extra_features[:limit]
        labels = labels[:limit]
        sample_weights = sample_weights[:limit]

        assert labels.sum() > 0, "Not enough positive labels. Increase limit!"

    timeseries = True if lookback > 1 else False

    print("Building StepNet...")
    model = build_stepcovnet(input_shape, timeseries, extra_input_shape, pretrained_model)

    model.compile(loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.1),
                  optimizer=Nadam(beta_1=0.99),
                  metrics=["accuracy"])

    batch_size = 256
    max_epochs = 30

    model_train(model,
                batch_size,
                max_epochs,
                features,
                extra_features,
                labels,
                sample_weights,
                class_weights,
                all_scalers,
                prefix,
                model_out_path,
                lookback)
