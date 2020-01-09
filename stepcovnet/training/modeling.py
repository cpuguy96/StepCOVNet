from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os

import numpy as np
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint

from stepcovnet.common.utils import feature_reshape
from stepcovnet.common.utils import get_scalers
from stepcovnet.training.data_preparation import FeatureGenerator
from stepcovnet.training.data_preparation import get_init_bias_correction
from stepcovnet.training.data_preparation import get_init_expected_loss
from stepcovnet.training.data_preparation import get_split_indexes
from stepcovnet.training.data_preparation import load_data
from stepcovnet.training.network import build_stepcovnet
from stepcovnet.training.parameters import BATCH_SIZE
from stepcovnet.training.parameters import MAX_EPOCHS
from stepcovnet.training.parameters import PATIENCE
from stepcovnet.training.tf_config import *


def train_model(model, features, extra_features, labels, sample_weights, class_weights, all_scalers, model_name,
                model_out_path, multi, log_path):
    indices_all, indices_train, indices_validation = get_split_indexes(labels, multi)

    print("Number of samples: %s" % len(indices_all))

    if extra_features is not None:
        training_extra_features = extra_features[indices_train]
        testing_extra_features = extra_features[indices_validation]
    else:
        training_extra_features = None
        testing_extra_features = None

    print("\nCreating training scalers...")

    training_scaler = get_scalers(features[indices_train], multi)

    training_callbacks = [EarlyStopping(monitor='val_pr_auc', patience=PATIENCE, verbose=0, mode="max"),
                          ModelCheckpoint(filepath=os.path.join(model_out_path, model_name + '_callback.h5'),
                                          monitor='val_pr_auc',
                                          verbose=0,
                                          save_best_only=True)]

    if log_path is not None:
        os.makedirs(os.path.join(log_path, "split_dataset"), exist_ok=True)
        training_callbacks.append(
            tf.keras.callbacks.TensorBoard(log_dir=os.path.join(log_path, "split_dataset"), histogram_freq=1,
                                           profile_batch=100000000))

    print("\nCreating training and test sets...")

    weights = model.get_weights()

    train_gen = FeatureGenerator(features[indices_train], labels[indices_train],
                                 sample_weights=sample_weights[indices_train], multi=multi, scaler=training_scaler,
                                 extra_features=training_extra_features)
    test_gen = FeatureGenerator(features[indices_validation], labels[indices_validation],
                                sample_weights=sample_weights[indices_validation], multi=multi, scaler=training_scaler,
                                extra_features=testing_extra_features)

    train_enqueuer = tf.keras.utils.OrderedEnqueuer(train_gen,
                                                    use_multiprocessing=False,
                                                    shuffle=False)
    test_enqueuer = tf.keras.utils.OrderedEnqueuer(test_gen,
                                                   use_multiprocessing=False,
                                                   shuffle=False)

    train_steps_per_epoch = int(np.ceil(len(indices_train) / BATCH_SIZE))
    val_steps_per_epoch = int(np.ceil(len(indices_validation) / BATCH_SIZE))

    print("\nStarting training...")

    train_enqueuer.start()
    test_enqueuer.start()
    history = model.fit(train_enqueuer.get(),
                        epochs=MAX_EPOCHS,
                        steps_per_epoch=train_steps_per_epoch,
                        validation_steps=val_steps_per_epoch,
                        callbacks=training_callbacks,
                        class_weight=class_weights,
                        validation_data=test_enqueuer.get(),
                        verbose=1)
    train_enqueuer.stop()
    test_enqueuer.stop()

    print("\n*****************************")
    print("***** TRAINING FINISHED *****")
    print("*****************************\n")

    model.save(os.path.join(model_out_path, model_name + ".h5"))

    callbacks = []

    if log_path is not None:
        os.makedirs(os.path.join(log_path, "whole_dataset"), exist_ok=True)
        callbacks.append(
            tf.keras.callbacks.TensorBoard(log_dir=os.path.join(log_path, "whole_dataset"), histogram_freq=1,
                                           profile_batch=100000000))

    # train again using all data
    epochs_final = len(history.history['val_loss'])

    print("\nUsing entire dataset for training...")

    all_gen = FeatureGenerator(features, labels, sample_weights=sample_weights, multi=multi, scaler=all_scalers,
                               extra_features=extra_features)

    all_enqueuer = tf.keras.utils.OrderedEnqueuer(all_gen,
                                                  use_multiprocessing=False,
                                                  shuffle=False)

    steps_per_epoch = int(np.ceil(len(indices_all) / BATCH_SIZE))

    model.set_weights(weights)

    print("\nStarting retraining...")

    all_enqueuer.start()
    model.fit(all_enqueuer.get(),
              steps_per_epoch=steps_per_epoch,
              epochs=epochs_final,
              callbacks=callbacks,
              class_weight=class_weights,
              verbose=1)
    all_enqueuer.stop()

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
        # tf.keras.metrics.TruePositives(name='tp'),
        # tf.keras.metrics.FalsePositives(name='fp'),
        # tf.keras.metrics.TrueNegatives(name='tn'),
        # tf.keras.metrics.FalseNegatives(name='fn'),
        tf.keras.metrics.BinaryAccuracy(name='acc'),
        tf.keras.metrics.Precision(name='pre'),
        tf.keras.metrics.Recall(name='rec'),
        tf.keras.metrics.AUC(curve="PR", name='pr_auc'),
    ]

    b0 = get_init_bias_correction(labels.sum(), len(labels))

    model = build_stepcovnet(input_shape, timeseries, extra_input_shape, pretrained_model, model_type=model_type,
                             output_bias_init=tf.keras.initializers.Constant(value=b0))

    model.compile(loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.1),
                  optimizer=tf.keras.optimizers.Nadam(beta_1=0.99),
                  metrics=metrics)

    print(model.summary())

    # Best practices mentioned in https://www.tensorflow.org/tutorials/structured_data/imbalanced_data
    expected_init_loss = get_init_expected_loss(labels.sum(), len(labels))
    result_init_loss = model.evaluate(features[:1000], labels[:1000], batch_size=BATCH_SIZE, verbose=0)[0]
    print("Bias init b0 %s " % b0)
    print("Loss should be %s. Actually is %s" % (expected_init_loss, result_init_loss))
    if not np.array_equal(np.around([expected_init_loss], decimals=1), np.around([result_init_loss], decimals=1)):
        print("WARNING! Expected loss is not close enough to initial loss!")

    train_model(model, features, extra_features, labels, sample_weights, class_weights, all_scalers, model_name,
                model_out_path, multi, log_path)
