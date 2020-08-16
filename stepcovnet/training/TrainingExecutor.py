import json
import os

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.python.keras.callbacks import ModelCheckpoint

from stepcovnet.common.tf_config import tf_init


class TrainingExecutor(object):
    def __init__(self, training_input, stepcovnet_model):
        self.training_input = training_input
        self.stepcovnet_model = stepcovnet_model
        tf_init()

    def execute(self):
        retrain = self.training_input.training_config.hyperparameters.retrain
        weights = self.stepcovnet_model.model.get_weights() if retrain else None
        loss = self.training_input.training_config.hyperparameters.loss
        metrics = self.training_input.training_config.hyperparameters.metrics
        optimizer = self.training_input.training_config.hyperparameters.optimizer

        self.stepcovnet_model.model.compile(loss=loss, metrics=metrics, optimizer=optimizer)
        self.stepcovnet_model.model.summary()
        history = self.train(self.get_training_callbacks())
        self.save(training_history=history, retrained=False)

        if retrain:
            epochs_final = len(history.history['val_loss'])
            retraining_history = self.retrain(saved_original_weights=weights, epochs=epochs_final,
                                              callbacks=self.get_retraining_callbacks())
            self.save(training_history=retraining_history, retrained=True)

    def get_training_callbacks(self):
        model_out_path = self.stepcovnet_model.model_path
        model_name = self.stepcovnet_model.model_name
        log_path = self.training_input.training_config.hyperparameters.log_path
        patience = self.training_input.training_config.hyperparameters.patience
        callbacks = [
            ModelCheckpoint(filepath=os.path.join(model_out_path, model_name + '_callback'), monitor='val_loss',
                            verbose=0, save_best_only=True)]
        if patience > 0:
            callbacks.append(EarlyStopping(monitor='val_pr_auc', patience=patience, verbose=0, mode="max"))
        elif patience == 0:
            raise ValueError("Patience must be > 0")

        if log_path is not None:
            os.makedirs(os.path.join(log_path, "split_dataset"), exist_ok=True)
            callbacks.append(TensorBoard(log_dir=os.path.join(log_path, "split_dataset"),
                                         histogram_freq=1, profile_batch=100000000))

        return callbacks

    def get_retraining_callbacks(self):
        log_path = self.training_input.training_config.hyperparameters.log_path
        callbacks = []

        if log_path is not None:
            os.makedirs(os.path.join(log_path, "whole_dataset"), exist_ok=True)
            callbacks.append(TensorBoard(log_dir=os.path.join(log_path, "whole_dataset"),
                                         histogram_freq=1, profile_batch=100000000))

        return callbacks

    def train(self, callbacks):
        print("Training on %d samples (%d songs) and validating on %d samples (%d songs)" % (
            self.training_input.train_feature_generator.num_samples,
            len(self.training_input.train_feature_generator.train_indexes),
            self.training_input.val_feature_generator.num_samples,
            len(self.training_input.val_feature_generator.train_indexes)))
        print("\nStarting training...")
        history = self.stepcovnet_model.model.fit(x=self.training_input.train_generator,
                                                  epochs=self.training_input.training_config.hyperparameters.epochs,
                                                  steps_per_epoch=len(self.training_input.train_feature_generator),
                                                  validation_steps=len(self.training_input.val_feature_generator),
                                                  callbacks=callbacks,
                                                  # class_weight=self.training_input.training_config.class_weights,
                                                  validation_data=self.training_input.val_generator,
                                                  verbose=1)
        print("\n*****************************")
        print("***** TRAINING FINISHED *****")
        print("*****************************\n")
        return history

    def retrain(self, saved_original_weights, epochs, callbacks):
        print("Training on %d samples (%d songs)" % (self.training_input.all_feature_generator.num_samples,
                                                     len(self.training_input.all_feature_generator.train_indexes)))
        print("\nStarting retraining...")
        self.stepcovnet_model.model.set_weights(saved_original_weights)
        history = self.stepcovnet_model.model.fit(x=self.training_input.all_generator,
                                                  epochs=epochs,
                                                  steps_per_epoch=len(self.training_input.all_feature_generator),
                                                  callbacks=callbacks,
                                                  # class_weight=self.training_input.training_config.class_weights,
                                                  verbose=1)
        print("\n*****************************")
        print("***** RETRAINING FINISHED *****")
        print("*****************************\n")
        return history

    def save(self, retrained, training_history):
        model_out_path = self.stepcovnet_model.model_path
        model_name = self.stepcovnet_model.model_name
        model_name += '_retrained' if retrained else ""
        print("Saving model \"%s\" at %s" % (model_name, model_out_path))
        self.stepcovnet_model.model.save(os.path.join(model_out_path, model_name))
        if self.stepcovnet_model.metadata is None:
            self.stepcovnet_model.build_metadata_from_training_config(self.training_input.training_config)
        history_name = "retraining_history" if retrained else "training_history"
        self.stepcovnet_model.metadata[history_name] = training_history.history
        print("Saving model metadata at %s" % model_out_path)
        with open(os.path.join(model_out_path, 'metadata.json'), 'w') as json_file:
            json_file.write(json.dumps(self.stepcovnet_model.metadata))
