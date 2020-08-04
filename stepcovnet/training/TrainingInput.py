import tensorflow as tf

from stepcovnet.training.TrainingFeatureGenerator import TrainingFeatureGenerator


class TrainingInput(object):
    def __init__(self, dataset, training_config):
        self.training_config = training_config
        self.output_types = (
            {"arrow_input": tf.dtypes.int32,
             "arrow_mask": tf.dtypes.int32,
             "audio_input": tf.dtypes.float64},
            tf.dtypes.int8,  # labels
            tf.dtypes.float16  # sample weights
        )
        self.output_shape = (
            {"arrow_input": tf.TensorShape((None,) + self.training_config.arrow_input_shape),
             "arrow_mask": tf.TensorShape((None,) + self.training_config.arrow_mask_shape),
             "audio_input": tf.TensorShape((None,) + self.training_config.audio_input_shape)},
            tf.TensorShape((None,) + self.training_config.label_shape),  # labels
            tf.TensorShape([None])  # sample weights
        )
        self.train_feature_generator = TrainingFeatureGenerator(dataset,
                                                                lookback=self.training_config.lookback,
                                                                batch_size=self.training_config.hyperparameters.batch_size,
                                                                indexes=self.training_config.train_indexes,
                                                                num_samples=self.training_config.num_train_samples,
                                                                scalers=self.training_config.train_scalers)
        self.val_feature_generator = TrainingFeatureGenerator(dataset,
                                                              lookback=self.training_config.lookback,
                                                              batch_size=self.training_config.hyperparameters.batch_size,
                                                              indexes=self.training_config.val_indexes,
                                                              num_samples=self.training_config.num_val_samples,
                                                              scalers=self.training_config.train_scalers)
        self.all_feature_generator = TrainingFeatureGenerator(dataset,
                                                              lookback=self.training_config.lookback,
                                                              batch_size=self.training_config.hyperparameters.batch_size,
                                                              indexes=self.training_config.all_indexes,
                                                              num_samples=self.training_config.num_samples,
                                                              scalers=self.training_config.all_scalers)

    def get_tf_dataset(self, generator):
        return tf.data.Dataset.from_generator(
            generator,
            output_types=self.output_types,
            output_shapes=self.output_shape,
        ).prefetch(tf.data.experimental.AUTOTUNE)

    @property
    def train_generator(self):
        return self.get_tf_dataset(self.train_feature_generator)

    @property
    def val_generator(self):
        return self.get_tf_dataset(self.val_feature_generator)

    @property
    def all_generator(self):
        return self.get_tf_dataset(self.all_feature_generator)
