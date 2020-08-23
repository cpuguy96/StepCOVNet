import json
import os

import joblib
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.python.keras.callbacks import ModelCheckpoint

from stepcovnet.executor.AbstractExecutor import AbstractExecutor
from stepcovnet.model_input.TrainingInput import TrainingInput


class TrainingExecutor(AbstractExecutor):
    def __init__(self, training_input: TrainingInput, stepcovnet_model):
        super(TrainingExecutor, self).__init__(input_data=training_input, stepcovnet_model=stepcovnet_model)

    def execute(self):
        retrain = self.input_data.config.hyperparameters.retrain
        weights = self.stepcovnet_model.model.get_weights() if retrain else None
        loss = self.input_data.config.hyperparameters.loss
        metrics = self.input_data.config.hyperparameters.metrics
        optimizer = self.input_data.config.hyperparameters.optimizer

        self.stepcovnet_model.model.compile(loss=loss, metrics=metrics, optimizer=optimizer)
        self.stepcovnet_model.model.summary()
        self.save(pretrained=True, retrained=False)  # Saving pretrained model in the case of errors during training
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
        log_path = self.input_data.config.hyperparameters.log_path
        patience = self.input_data.config.hyperparameters.patience
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
        log_path = self.input_data.config.hyperparameters.log_path
        callbacks = []

        if log_path is not None:
            os.makedirs(os.path.join(log_path, "whole_dataset"), exist_ok=True)
            callbacks.append(TensorBoard(log_dir=os.path.join(log_path, "whole_dataset"),
                                         histogram_freq=1, profile_batch=100000000))

        return callbacks

    def train(self, callbacks):
        print("Training on %d samples (%d songs) and validating on %d samples (%d songs)" % (
            self.input_data.train_feature_generator.num_samples,
            len(self.input_data.train_feature_generator.train_indexes),
            self.input_data.val_feature_generator.num_samples,
            len(self.input_data.val_feature_generator.train_indexes)))
        print("\nStarting training...")
        history = self.stepcovnet_model.model.fit(x=self.input_data.train_generator,
                                                  epochs=self.input_data.config.hyperparameters.epochs,
                                                  steps_per_epoch=len(self.input_data.train_feature_generator),
                                                  validation_steps=len(self.input_data.val_feature_generator),
                                                  callbacks=callbacks,
                                                  class_weight=self.input_data.config.train_class_weights,
                                                  validation_data=self.input_data.val_generator,
                                                  verbose=1)
        print("\n*****************************")
        print("***** TRAINING FINISHED *****")
        print("*****************************\n")
        return history

    def retrain(self, saved_original_weights, epochs, callbacks):
        print("Training on %d samples (%d songs)" % (self.input_data.all_feature_generator.num_samples,
                                                     len(self.input_data.all_feature_generator.train_indexes)))
        print("\nStarting retraining...")
        self.stepcovnet_model.model.set_weights(saved_original_weights)
        history = self.stepcovnet_model.model.fit(x=self.input_data.all_generator,
                                                  epochs=epochs,
                                                  steps_per_epoch=len(self.input_data.all_feature_generator),
                                                  callbacks=callbacks,
                                                  class_weight=self.input_data.config.all_class_weights,
                                                  verbose=1)
        print("\n*******************************")
        print("***** RETRAINING FINISHED *****")
        print("*******************************\n")
        return history

    def save(self, retrained, training_history=None, pretrained=False):
        model_out_path = self.stepcovnet_model.model_path
        model_name = self.stepcovnet_model.model_name
        if pretrained:
            if self.input_data.config.all_scalers is not None:
                joblib.dump(self.input_data.config.all_scalers,
                            open(os.path.join(model_out_path, model_name + '_scaler.pkl'), 'wb'))
            model_name += '_pretrained'
        elif retrained:
            model_name += '_retrained'
        print("Saving model \"%s\" at %s" % (model_name, model_out_path))
        self.stepcovnet_model.model.save(os.path.join(model_out_path, model_name))
        if self.stepcovnet_model.metadata is None:
            self.stepcovnet_model.build_metadata_from_training_config(self.input_data.config)
        if training_history is not None and not pretrained:
            history_name = "retraining_history" if retrained else "training_history"
            self.stepcovnet_model.metadata[history_name] = training_history.history
        print("Saving model metadata at %s" % model_out_path)
        with open(os.path.join(model_out_path, 'metadata.json'), 'w') as json_file:
            json_file.write(json.dumps(self.stepcovnet_model.metadata))
