import json
import os
from abc import ABC
from abc import abstractmethod

import joblib
import numpy as np
import tensorflow as tf
from tensorflow.python.keras import callbacks

from src.stepcovnet import config
from src.stepcovnet import constants
from src.stepcovnet import encoder
from src.stepcovnet import inputs
from src.stepcovnet import model
from src.stepcovnet import tf_config
from src.stepcovnet import training
from src.stepcovnet import utils


class AbstractExecutor(ABC):
    def __init__(self, stepcovnet_model: model.StepCOVNetModel, *args, **kwargs):
        self.stepcovnet_model = stepcovnet_model
        tf_config.tf_init()

    @abstractmethod
    def execute(self, input_data: inputs.AbstractInput):
        pass


class InferenceExecutor(AbstractExecutor):
    def __init__(self, stepcovnet_model: model.StepCOVNetModel, verbose: bool = False):
        super(InferenceExecutor, self).__init__(stepcovnet_model=stepcovnet_model)
        self.verbose = verbose
        self.binary_arrow_encoder = encoder.BinaryArrowEncoder()
        self.label_arrow_encoder = encoder.LabelArrowEncoder()

    def execute(self, input_data: inputs.InferenceInput) -> list[str]:
        arrow_input = input_data.arrow_input_init
        arrow_mask = input_data.arrow_mask_init
        pred_arrows = []
        inferer = self.stepcovnet_model.model.signatures["serving_default"]
        for audio_features_index in range(len(input_data.audio_features)):
            audio_features = utils.get_samples_ngram_with_mask(
                samples=input_data.audio_features[
                    max(
                        audio_features_index + 1 - input_data.config.lookback, 0
                    ) : audio_features_index
                    + 1
                ],
                lookback=input_data.config.lookback,
                squeeze=False,
            )[0][-1]
            audio_input = utils.apply_scalers(
                features=audio_features, scalers=input_data.config.scalers
            )
            binary_arrows_probs = inferer(
                arrow_input=tf.convert_to_tensor(arrow_input),
                arrow_mask=tf.convert_to_tensor(arrow_mask),
                audio_input=tf.convert_to_tensor(audio_input),
            )
            binary_arrows_probs = (
                next(iter(binary_arrows_probs.values())).numpy().ravel()
            )
            binary_encoded_arrows = []
            for i in range(constants.NUM_ARROWS):
                binary_arrow_prob = binary_arrows_probs[
                    constants.NUM_ARROW_TYPES * i: constants.NUM_ARROW_TYPES * (i + 1)
                ]
                encoded_arrow = np.random.choice(
                    constants.NUM_ARROW_TYPES, 1, p=binary_arrow_prob
                )[0]
                binary_encoded_arrows.append(str(encoded_arrow))
            arrows = "".join(binary_encoded_arrows)
            pred_arrows.append(arrows)
            # Roll and append predicted arrow to input to predict next sample
            arrow_input = np.roll(arrow_input, -1, axis=0)
            arrow_mask = np.roll(arrow_mask, -1, axis=0)
            arrow_input[0][-1] = self.label_arrow_encoder.encode(arrows)
            arrow_mask[0][-1] = 1
            if self.verbose and audio_features_index % 100 == 0:
                print(
                    "[%d/%d] Samples generated"
                    % (audio_features_index, len(input_data.audio_features))
                )
        return pred_arrows


class TrainingExecutor(AbstractExecutor):
    def __init__(self, stepcovnet_model: model.StepCOVNetModel):
        super(TrainingExecutor, self).__init__(stepcovnet_model=stepcovnet_model)

    def execute(self, input_data: inputs.TrainingInput) -> model.StepCOVNetModel:
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
                callback_list=self.get_retraining_callbacks(hyperparameters),
            )
            self.save(
                input_data.config, training_history=retraining_history, retrained=True
            )
        return self.stepcovnet_model

    def get_training_callbacks(
        self, hyperparameters: training.TrainingHyperparameters
    ) -> list[callbacks.Callback]:
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
    def get_retraining_callbacks(
        hyperparameters: training.TrainingHyperparameters,
    ) -> list[callbacks.Callback]:
        log_path = hyperparameters.log_path
        callback_list = []

        if log_path is not None:
            os.makedirs(os.path.join(log_path, "whole_dataset"), exist_ok=True)
            callback_list.append(
                callbacks.TensorBoard(
                    log_dir=os.path.join(log_path, "whole_dataset"),
                    histogram_freq=1,
                    profile_batch=100000000,
                )
            )
        return callback_list

    def train(
        self, input_data: inputs.TrainingInput, callback_list: list[callbacks.Callback]
    ) -> callbacks.History:
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
            callbacks=callback_list,
            class_weight=input_data.config.train_class_weights,
            validation_data=input_data.val_generator,
            verbose=1,
        )
        print("\n*****************************")
        print("***** TRAINING FINISHED *****")
        print("*****************************\n")
        return history

    def retrain(
        self,
        input_data: inputs.TrainingInput,
        saved_original_weights: np.ndarray,
        epochs: int,
        callback_list: list[callbacks.Callback],
    ) -> callbacks.History:
        print(
            "Training on %d samples (%d songs)"
            % (
                input_data.all_feature_generator.num_samples,
                len(input_data.all_feature_generator.train_indexes),
            )
        )
        print("\nStarting retraining...")
        self.stepcovnet_model.model.set_weights(saved_original_weights)
        history: callbacks.History = self.stepcovnet_model.model.fit(
            x=input_data.all_generator,
            epochs=epochs,
            steps_per_epoch=len(input_data.all_feature_generator),
            callbacks=callback_list,
            class_weight=input_data.config.all_class_weights,
            verbose=1,
        )
        print("\n*******************************")
        print("***** RETRAINING FINISHED *****")
        print("*******************************\n")
        return history

    def save(
        self,
        training_config: config.TrainingConfig,
        retrained: bool,
        training_history: callbacks.History | None = None,
        pretrained: bool = False,
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
