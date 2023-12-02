from abc import ABC

import numpy as np
import tensorflow as tf

from stepcovnet import config
from stepcovnet.common.utils import get_samples_ngram_with_mask
from stepcovnet.data_collection.sample_collection_helper import get_audio_features
from stepcovnet.training.TrainingFeatureGenerator import TrainingFeatureGenerator


class AbstractInput(ABC, object):
    def __init__(self, input_config, *args, **kwargs):
        self.config = input_config


class InferenceInput(AbstractInput):
    def __init__(self, inference_config: config.InferenceConfig):
        super(InferenceInput, self).__init__(input_config=inference_config)
        self.audio_features = get_audio_features(
            wav_path=self.config.audio_path,
            file_name=self.config.file_name,
            config=self.config.dataset_config,
        )
        self.arrow_input_init, self.arrow_mask_init = get_samples_ngram_with_mask(
            samples=np.array([0]),
            lookback=self.config.lookback,
            reshape=True,
            mask_padding_value=0,
        )
        self.arrow_input_init = self.arrow_input_init[:-1, 1:]
        self.arrow_mask_init = self.arrow_mask_init[:-1, 1:]


class TrainingInput(AbstractInput):
    def __init__(self, training_config: config.TrainingConfig):
        super(TrainingInput, self).__init__(input_config=training_config)
        self.output_types = (
            {
                "arrow_input": tf.dtypes.int32,
                "arrow_mask": tf.dtypes.int32,
                "audio_input": tf.dtypes.float64,
            },
            tf.dtypes.int8,  # labels
            tf.dtypes.float16,  # sample weights
        )
        self.output_shape = (
            {
                "arrow_input": tf.TensorShape((None,) + self.config.arrow_input_shape),
                "arrow_mask": tf.TensorShape((None,) + self.config.arrow_mask_shape),
                "audio_input": tf.TensorShape((None,) + self.config.audio_input_shape),
            },
            tf.TensorShape((None,) + self.config.label_shape),  # labels
            tf.TensorShape((None,)),  # sample weights
        )
        self.train_feature_generator = TrainingFeatureGenerator(
            dataset_path=self.config.dataset_path,
            dataset_type=self.config.dataset_type,
            lookback=self.config.lookback,
            batch_size=self.config.hyperparameters.batch_size,
            indexes=self.config.train_indexes,
            num_samples=self.config.num_train_samples,
            scalers=self.config.train_scalers,
            difficulty=self.config.difficulty,
            warmup=True,
            tokenizer_name=self.config.tokenizer_name,
        )
        self.val_feature_generator = TrainingFeatureGenerator(
            dataset_path=self.config.dataset_path,
            dataset_type=self.config.dataset_type,
            lookback=self.config.lookback,
            batch_size=self.config.hyperparameters.batch_size,
            indexes=self.config.val_indexes,
            num_samples=self.config.num_val_samples,
            scalers=self.config.train_scalers,
            difficulty=self.config.difficulty,
            shuffle=False,
            tokenizer_name=self.config.tokenizer_name,
        )
        self.all_feature_generator = TrainingFeatureGenerator(
            dataset_path=self.config.dataset_path,
            dataset_type=self.config.dataset_type,
            lookback=self.config.lookback,
            batch_size=self.config.hyperparameters.batch_size,
            indexes=self.config.all_indexes,
            num_samples=self.config.num_samples,
            scalers=self.config.all_scalers,
            difficulty=self.config.difficulty,
            warmup=True,
            tokenizer_name=self.config.tokenizer_name,
        )

    def get_tf_dataset(self, generator):
        return tf.data.Dataset.from_generator(
            generator,
            output_types=self.output_types,
            output_shapes=self.output_shape,
        ).prefetch(tf.data.AUTOTUNE)

    @property
    def train_generator(self):
        return self.get_tf_dataset(self.train_feature_generator)

    @property
    def val_generator(self):
        return self.get_tf_dataset(self.val_feature_generator)

    @property
    def all_generator(self):
        return self.get_tf_dataset(self.all_feature_generator)
