from __future__ import absolute_import, division, print_function, unicode_literals

import os

import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from stepcovnet.common.utils import get_scalers, feature_reshape, pre_process
from stepcovnet.configuration.parameters import BATCH_SIZE, MAX_EPOCHS
from stepcovnet.training.data_preparation import load_data
from stepcovnet.training.network import build_stepcovnet

os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH '] = 'true'
tf.config.optimizer.set_jit(True)

tf.random.set_seed(42)


def train_model(model, features, extra_features, labels, sample_weights, class_weights, all_scalers, model_name,
                model_out_path, multi, log_path):
    indices_all = range(len(features))
    print("Number of samples: %s" % len(indices_all))

    if multi:
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

    training_scaler = get_scalers(features[indices_train], multi)

    training_callbacks = [EarlyStopping(monitor='val_pr_auc', patience=3, verbose=0, mode="max"),
                          ModelCheckpoint(filepath=os.path.join(model_out_path, model_name + '_callback.h5'),
                                          monitor='pr_auc',
                                          verbose=0,
                                          save_best_only=True)]

    if log_path is not None:
        os.makedirs(os.path.join(log_path, "split_dataset"), exist_ok=True)
        training_callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=os.path.join(log_path, "split_dataset"), histogram_freq=1, profile_batch=100000000))

    print("\nCreating training and test sets...")

    weights = model.get_weights()

    x_train, y_train = pre_process(features[indices_train], multi, labels[indices_train], training_extra_features,
                                   training_scaler)

    x_test, y_test = pre_process(features[indices_validation], multi, labels[indices_validation],
                                 testing_extra_features, training_scaler)

    print("\nStarting training...")

    history = model.fit(x=x_train,
                        y=y_train,
                        batch_size=BATCH_SIZE,
                        epochs=MAX_EPOCHS,
                        callbacks=training_callbacks,
                        class_weight=class_weights,
                        sample_weight=sample_weights[indices_train],
                        validation_data=(x_test, y_test, sample_weights[indices_validation]),
                        shuffle="batch",
                        verbose=1)

    print("\n*****************************")
    print("***** TRAINING FINISHED *****")
    print("*****************************\n")

    model.save(os.path.join(model_out_path, model_name + ".h5"))

    callbacks = []

    if log_path is not None:
        os.makedirs(os.path.join(log_path, "whole_dataset"), exist_ok=True)
        callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=os.path.join(log_path, "whole_dataset"), histogram_freq=1, profile_batch=100000000))

    # train again use all train and validation set
    epochs_final = len(history.history['val_loss'])

    print("\nUsing entire dataset for training...")

    all_x, all_y = pre_process(features, multi, labels, extra_features, all_scalers)

    model.set_weights(weights)

    print("\nStarting retraining...")

    model.fit(x=all_x,
              y=all_y,
              batch_size=BATCH_SIZE,
              epochs=epochs_final,
              callbacks=callbacks,
              sample_weight=sample_weights,
              class_weight=class_weights,
              shuffle="batch",
              verbose=1)

    print("\n*******************************")
    print("***** RETRAINING FINISHED *****")
    print("*******************************\n")

    model.save(os.path.join(model_out_path, model_name + "_retrained.h5"))


def prepare_model(filename_features, filename_labels, filename_sample_weights, filename_scaler, input_shape, model_name,
                  model_out_path, extra_input_shape, path_extra_features, lookback, multi, limit=-1, log_path=None,
                  filename_pretrained_model=None, model_type="normal"):
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

    print("Reshaping features...")
    features = feature_reshape(features, multi)

    timeseries = True if lookback > 1 else False

    print("Building StepCOVNet...")

    metrics = [
        tf.keras.metrics.TruePositives(name='tp'),
        tf.keras.metrics.FalsePositives(name='fp'),
        tf.keras.metrics.TrueNegatives(name='tn'),
        tf.keras.metrics.FalseNegatives(name='fn'),
        tf.keras.metrics.BinaryAccuracy(name='acc'),
        tf.keras.metrics.Precision(name='pre'),
        tf.keras.metrics.Recall(name='rec'),
        tf.keras.metrics.AUC(curve="PR", name='pr_auc'),
    ]

    import numpy as np

    b0 = np.log(labels.sum()/(len(labels) - labels.sum()))

    model = build_stepcovnet(input_shape, timeseries, extra_input_shape, pretrained_model, model_type=model_type, output_bias_init=tf.keras.initializers.Constant(value=b0))

    model.compile(loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.1),
                  optimizer=tf.keras.optimizers.Nadam(beta_1=0.99, clipvalue=5),
                  metrics=metrics)

    print(model.summary())

    p0 = labels.sum()/len(labels)

    # Best practices mentioned in https://www.tensorflow.org/tutorials/structured_data/imbalanced_data
    expected_init_loss = -p0*np.log(p0)-(1-p0)*np.log(1-p0)
    result_init_loss = model.evaluate(features[:1000], labels[:1000], batch_size=BATCH_SIZE, verbose=0)[0]
    if not np.array_equal(np.around([expected_init_loss], decimals=1), np.around([result_init_loss], decimals=1)):
        print("WARNING! Expected loss is not close enough to initial loss!")
    print("Bias init b0 %s " % b0)
    print("Loss should be %s. Actually is %s" % (expected_init_loss, result_init_loss))

    train_model(model, features, extra_features, labels, sample_weights, class_weights, all_scalers, model_name,
                model_out_path, multi, log_path)
