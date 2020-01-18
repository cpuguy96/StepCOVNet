from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os

import numpy as np
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint

from stepcovnet.common.modeling_dataset import ModelDataset
from stepcovnet.common.utils import get_sklearn_scalers
from stepcovnet.training.data_preparation import FeatureGenerator
from stepcovnet.training.data_preparation import get_split_indexes
from stepcovnet.training.data_preparation import load_data
from stepcovnet.training.network import build_stepcovnet
from stepcovnet.training.network import get_init_bias_correction
from stepcovnet.training.parameters import BATCH_SIZE
from stepcovnet.training.parameters import MAX_EPOCHS
from stepcovnet.training.parameters import PATIENCE
from stepcovnet.training.tf_config import *


def train_model(model, dataset_path, multi, output_shape, output_types, index_all, index_train, index_val, all_scalers,
                training_scaler, class_weights, model_name, model_out_path, log_path):
    training_callbacks = [ModelCheckpoint(filepath=os.path.join(model_out_path, model_name + '_callback.h5'),
                                          monitor='val_pr_auc',
                                          verbose=0,
                                          save_best_only=True)]
    if PATIENCE > 0:
        training_callbacks.append(EarlyStopping(monitor='val_pr_auc', patience=PATIENCE, verbose=0, mode="max"))
    elif PATIENCE == 0:
        raise ValueError("Patience must be > 0")

    if log_path is not None:
        os.makedirs(os.path.join(log_path, "split_dataset"), exist_ok=True)
        training_callbacks.append(
            tf.keras.callbacks.TensorBoard(log_dir=os.path.join(log_path, "split_dataset"), histogram_freq=1,
                                           profile_batch=100000000))

    weights = model.get_weights()

    print("\nCreating training and test sets...")

    train_steps_per_epoch = int(np.ceil(len(index_train) / BATCH_SIZE))
    val_steps_per_epoch = int(np.ceil(len(index_val) / BATCH_SIZE))

    train_gen = FeatureGenerator(dataset_path, indexes=index_train, multi=multi, scaler=training_scaler, shuffle=True)
    val_gen = FeatureGenerator(dataset_path, indexes=index_val, multi=multi, scaler=training_scaler, shuffle=False)

    train_dataset = tf.data.Dataset.from_generator(
        train_gen,
        output_types=output_types,
        output_shapes=output_shape,
    ).prefetch(tf.data.experimental.AUTOTUNE)

    val_dataset = tf.data.Dataset.from_generator(
        val_gen,
        output_types=output_types,
        output_shapes=output_shape,
    ).prefetch(tf.data.experimental.AUTOTUNE)

    print("\nStarting training...")

    history = model.fit(x=train_dataset,
                        epochs=MAX_EPOCHS,
                        steps_per_epoch=train_steps_per_epoch,
                        validation_steps=val_steps_per_epoch,
                        callbacks=training_callbacks,
                        class_weight=class_weights,
                        validation_data=val_dataset,
                        verbose=1)

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

    model.set_weights(weights)

    epochs_final = len(history.history['val_loss'])

    print("\nUsing entire dataset for training...")

    steps_per_epoch = int(np.ceil(len(index_all) / BATCH_SIZE))

    all_gen = FeatureGenerator(dataset_path, indexes=index_all, multi=multi, scaler=all_scalers, shuffle=True)

    all_dataset = tf.data.Dataset.from_generator(
        all_gen,
        output_types=output_types,
        output_shapes=output_shape,
    ).prefetch(tf.data.experimental.AUTOTUNE)

    print("\nStarting retraining...")

    model.fit(all_dataset,
              steps_per_epoch=steps_per_epoch,
              epochs=epochs_final,
              callbacks=callbacks,
              class_weight=class_weights,
              verbose=1)

    print("\n*******************************")
    print("***** RETRAINING FINISHED *****")
    print("*******************************\n")

    model.save(os.path.join(model_out_path, model_name + "_retrained.h5"))


def prepare_model(dataset_path, model_out_path, input_shape, extra_input_shape=None, multi=False, extra=False,
                  filename_scaler=None, filename_pretrained_model=None, limit=-1, lookback=1, log_path=None,
                  model_name=None, model_type="normal"):
    print("Loading data...")
    all_scalers, pretrained_model = load_data(filename_scaler, filename_pretrained_model)

    timeseries = True if lookback > 1 else False

    with ModelDataset(dataset_path) as dataset:
        indices_all, indices_train, indices_validation = get_split_indexes(dataset, timeseries, limit)
        if limit > 0 and dataset.labels[indices_all].sum() == 0:
            raise ValueError("Not enough positive labels. Increase limit!")
        if extra and dataset.extra_features is None:
            raise ValueError("Modeling with extra features requested, but dataset doesn't have extra features.")
        class_weights = {0: (dataset.num_samples / dataset.neg_samples) / 2.0,
                         1: (dataset.num_samples / dataset.pos_samples) / 2.0}
        # Best practices mentioned in https://www.tensorflow.org/tutorials/structured_data/imbalanced_data
        b0 = get_init_bias_correction(dataset.pos_samples, dataset.num_samples)

        print("\nCreating training scalers...")

        if all_scalers is not None:
            training_scaler = get_sklearn_scalers(dataset.features[indices_train], multi)
        else:
            training_scaler = None

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
        tf.keras.metrics.AUC(name='auc'),
    ]

    model = build_stepcovnet(input_shape, timeseries, extra_input_shape, pretrained_model, model_type=model_type,
                             output_bias_init=tf.keras.initializers.Constant(value=b0))

    model.compile(loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.1),
                  optimizer=tf.keras.optimizers.Nadam(beta_1=0.99),
                  metrics=metrics)

    print(model.summary())

    print("Number of samples: %s" % len(indices_all))

    if timeseries:
        input_shape = (None,) + input_shape[1:]
    else:
        input_shape = (None,) + input_shape

    if extra:
        output_types = (
            {"log_mel_input": tf.dtypes.float16, "extra_input": tf.dtypes.int8}, tf.dtypes.int8, tf.dtypes.float16)
        output_shape = (
            {"log_mel_input": tf.TensorShape(input_shape), "extra_input": tf.TensorShape(extra_input_shape)},
            tf.TensorShape([None]), tf.TensorShape([None]))
    else:
        output_types = (tf.dtypes.float16, tf.dtypes.int8, tf.dtypes.float16)
        output_shape = (tf.TensorShape(input_shape), tf.TensorShape([None]), tf.TensorShape([None]))

    train_model(model, dataset_path, multi, output_shape, output_types, indices_all, indices_train, indices_validation,
                all_scalers, training_scaler, class_weights, model_name, model_out_path, log_path)
