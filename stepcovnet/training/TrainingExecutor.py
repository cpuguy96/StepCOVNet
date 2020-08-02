from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import TensorBoard


class TrainingExecutor(object):
    def __init__(self, training_input, stepcovnet_model):
        self.training_input = training_input
        self.stepcovnet_model = stepcovnet_model

    def execute(self):
        retrain = self.training_input.training_config.hyperparameters.retrain
        weights = self.stepcovnet_model.model.get_weights() if retrain else None
        loss = self.training_input.training_config.hyperparameters.loss
        metrics = self.training_input.training_config.hyperparameters.metrics
        optimizer = self.training_input.training_config.hyperparameters.optimizer

        self.configure_tensorflow()

        self.stepcovnet_model.model.compile(loss=loss, metrics=metrics, optimizer=optimizer)
        print(self.stepcovnet_model.model.summary())
        history = self.train(self.get_training_callbacks())
        self.save(retrained=False)

        if retrain:
            epochs_final = len(history.history['val_loss'])
            self.retrain(saved_original_weights=weights, epochs=epochs_final, callbacks=self.get_retraining_callbacks())
            self.save(retrained=True)

    def get_training_callbacks(self):
        model_out_path = self.stepcovnet_model.model_path
        model_name = self.stepcovnet_model.model_name
        log_path = self.training_input.training_config.hyperparameters.log_path
        patience = self.training_input.training_config.hyperparameters.patience
        callbacks = [
            ModelCheckpoint(filepath=os.path.join(model_out_path, model_name + '_callback.h5'), monitor='val_pr_auc',
                            verbose=0,
                            save_best_only=True)]
        if patience > 0:
            callbacks.append(EarlyStopping(monitor='val_pr_auc', patience=patience, verbose=0, mode="max"))
        elif patience == 0:
            raise ValueError("Patience must be > 0")

        if log_path is not None:
            os.makedirs(os.path.join(log_path, "split_dataset"), exist_ok=True)
            callbacks.append(
                TensorBoard(log_dir=os.path.join(log_path, "split_dataset"), histogram_freq=1,
                            profile_batch=100000000))

        return callbacks

    def get_retraining_callbacks(self):
        log_path = self.training_input.training_config.hyperparameters.log_path
        callbacks = []

        if log_path is not None:
            os.makedirs(os.path.join(log_path, "whole_dataset"), exist_ok=True)
            callbacks.append(
                TensorBoard(log_dir=os.path.join(log_path, "whole_dataset"), histogram_freq=1,
                            profile_batch=100000000))

        return callbacks

    def train(self, callbacks):
        print("\nStarting training...")
        history = self.stepcovnet_model.model.fit(x=self.training_input.train_generator,
                                                  epochs=self.training_input.training_config.hyperparameters.epochs,
                                                  steps_per_epoch=len(self.training_input.train_feature_generator),
                                                  validation_steps=len(self.training_input.val_feature_generator),
                                                  callbacks=callbacks,
                                                  class_weight=self.training_input.training_config.class_weights,
                                                  validation_data=self.training_input.val_generator,
                                                  verbose=1)
        print("\n*****************************")
        print("***** TRAINING FINISHED *****")
        print("*****************************\n")
        return history

    def retrain(self, saved_original_weights, epochs, callbacks):
        print("\nStarting retraining...")
        self.stepcovnet_model.model.set_weights(saved_original_weights)
        history = self.stepcovnet_model.model.fit(x=self.training_input.all_generator,
                                                  epochs=epochs,
                                                  steps_per_epoch=len(self.training_input.all_feature_generator),
                                                  callbacks=callbacks,
                                                  class_weight=self.training_input.training_config.class_weights,
                                                  verbose=1)
        print("\n*****************************")
        print("***** RETRAINING FINISHED *****")
        print("*****************************\n")
        return history

    def save(self, retrained):
        model_out_path = self.stepcovnet_model.model_path
        model_name = self.stepcovnet_model.model_name
        model = self.stepcovnet_model.model
        if retrained:
            model.save(os.path.join(model_out_path, model_name + "_retrained.h5"))
        else:
            model.save(os.path.join(model_out_path, model_name + ".h5"))

    @staticmethod
    def configure_tensorflow():
        gpu = tf.config.experimental.list_physical_devices('GPU')
        tf.config.experimental.set_memory_growth(gpu[0], True)

        policy = tf.keras.mixed_precision.experimental.Policy('mixed_float16')
        tf.keras.mixed_precision.experimental.set_policy(policy)

        tf.config.optimizer.set_jit(True)

        tf.random.set_seed(42)
        tf.compat.v1.set_random_seed(42)
