import json
import os

import joblib
from keras import callbacks

from stepcovnet.config.TrainingConfig import TrainingConfig
from stepcovnet.executor.AbstractExecutor import AbstractExecutor
from stepcovnet.inputs.TrainingInput import TrainingInput
from stepcovnet.training.TrainingHyperparameters import TrainingHyperparameters


class TrainingExecutor(AbstractExecutor):
    def __init__(self, stepcovnet_model):
        super(TrainingExecutor, self).__init__(stepcovnet_model=stepcovnet_model)

    def execute(self, input_data: TrainingInput):
        hyperparameters = input_data.config.hyperparameters

        weights = (
            self.stepcovnet_model.model.get_weights()
            if hyperparameters.retrain
            else None
        )

        self.stepcovnet_model.model.compile(
            loss=hyperparameters.loss,
            metrics=hyperparameters.metrics,
            optimizer=hyperparameters.optimizer,
        )
        self.stepcovnet_model.model.summary()
        # Saving scalers and metadata in the case of errors during training
        self.save(input_data.config, pretrained=True, retrained=False)
        history = self.train(input_data, self.get_training_callbacks(hyperparameters))
        self.save(input_data.config, training_history=history, retrained=False)

        if hyperparameters.retrain:
            epochs_final = len(history.history["val_loss"])
            retraining_history = self.retrain(
                input_data,
                saved_original_weights=weights,
                epochs=epochs_final,
                callbacks=self.get_retraining_callbacks(hyperparameters),
            )
            self.save(
                input_data.config, training_history=retraining_history, retrained=True
            )
        return self.stepcovnet_model

    def get_training_callbacks(self, hyperparameters: TrainingHyperparameters):
        model_out_path = self.stepcovnet_model.model_root_path
        model_name = self.stepcovnet_model.model_name
        log_path = hyperparameters.log_path
        patience = hyperparameters.patience
        callback_list = [
            callbacks.ModelCheckpoint(
                filepath=os.path.join(model_out_path, model_name + "_callback"),
                monitor="val_loss",
                verbose=0,
                save_best_only=True,
            )
        ]
        if patience > 0:
            callback_list.append(
                callbacks.EarlyStopping(
                    monitor="val_loss", patience=patience, verbose=0
                )
            )

        if log_path is not None:
            os.makedirs(os.path.join(log_path, "split_dataset"), exist_ok=True)
            callback_list.append(
                callbacks.TensorBoard(
                    log_dir=os.path.join(log_path, "split_dataset"),
                    histogram_freq=1,
                    profile_batch=100000000,
                )
            )
        return callback_list

    @staticmethod
    def get_retraining_callbacks(hyperparameters: TrainingHyperparameters):
        log_path = hyperparameters.log_path
        callbacks_list = []

        if log_path is not None:
            os.makedirs(os.path.join(log_path, "whole_dataset"), exist_ok=True)
            callbacks_list.append(
                callbacks.TensorBoard(
                    log_dir=os.path.join(log_path, "whole_dataset"),
                    histogram_freq=1,
                    profile_batch=100000000,
                )
            )
        return callbacks

    def train(self, input_data, callbacks):
        print(
            "Training on %d samples (%d songs) and validating on %d samples (%d songs)"
            % (
                input_data.train_feature_generator.num_samples,
                len(input_data.train_feature_generator.train_indexes),
                input_data.val_feature_generator.num_samples,
                len(input_data.val_feature_generator.train_indexes),
            )
        )
        print("\nStarting training...")
        history = self.stepcovnet_model.model.fit(
            x=input_data.train_generator,
            epochs=input_data.config.hyperparameters.epochs,
            steps_per_epoch=len(input_data.train_feature_generator),
            validation_steps=len(input_data.val_feature_generator),
            callbacks=callbacks,
            class_weight=input_data.config.train_class_weights,
            validation_data=input_data.val_generator,
            verbose=1,
        )
        print("\n*****************************")
        print("***** TRAINING FINISHED *****")
        print("*****************************\n")
        return history

    def retrain(self, input_data, saved_original_weights, epochs, callbacks):
        print(
            "Training on %d samples (%d songs)"
            % (
                input_data.all_feature_generator.num_samples,
                len(input_data.all_feature_generator.train_indexes),
            )
        )
        print("\nStarting retraining...")
        self.stepcovnet_model.model.set_weights(saved_original_weights)
        history = self.stepcovnet_model.model.fit(
            x=input_data.all_generator,
            epochs=epochs,
            steps_per_epoch=len(input_data.all_feature_generator),
            callbacks=callbacks,
            class_weight=input_data.config.all_class_weights,
            verbose=1,
        )
        print("\n*******************************")
        print("***** RETRAINING FINISHED *****")
        print("*******************************\n")
        return history

    def save(
        self,
        training_config: TrainingConfig,
        retrained,
        training_history=None,
        pretrained=False,
    ):
        model_out_path = self.stepcovnet_model.model_root_path
        model_name = self.stepcovnet_model.model_name
        if pretrained:
            if training_config.all_scalers is not None:
                joblib.dump(
                    training_config.all_scalers,
                    open(
                        os.path.join(model_out_path, model_name + "_scaler.pkl"), "wb"
                    ),
                )
        elif retrained:
            model_name += "_retrained"
        if not pretrained:
            print('Saving model "%s" at %s' % (model_name, model_out_path))
            self.stepcovnet_model.model.save(os.path.join(model_out_path, model_name))
        if self.stepcovnet_model.metadata is None:
            self.stepcovnet_model.build_metadata_from_training_config(training_config)
        if training_history is not None and not pretrained:
            history_name = "retraining_history" if retrained else "training_history"
            self.stepcovnet_model.metadata[history_name] = training_history.history
        print("Saving model metadata at %s" % model_out_path)
        with open(os.path.join(model_out_path, "metadata.json"), "w") as json_file:
            json_file.write(json.dumps(self.stepcovnet_model.metadata))
