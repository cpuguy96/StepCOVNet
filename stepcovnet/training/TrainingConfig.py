import numpy as np
from sklearn.model_selection import train_test_split

from stepcovnet.common.utils import get_channel_scalers


class TrainingConfig(object):
    def __init__(self, dataset, dataset_config, hyperparameters, all_scalers=None, limit=-1, lookback=1,
                 difficulty="challenge"):
        self.dataset_config = dataset_config
        self.hyperparameters = hyperparameters
        self.all_scalers = all_scalers
        self.limit = limit
        self.lookback = lookback
        self.difficulty = difficulty

        # Song level indexes
        self.all_indexes, self.train_indexes, self.val_indexes = self.get_train_val_split(dataset, limit)
        self.num_samples = self.get_num_samples(self.all_indexes, self.all_indexes)
        self.num_train_samples = self.get_num_samples(self.train_indexes, self.train_indexes)
        self.num_val_samples = self.get_num_samples(self.val_indexes, self.val_indexes)
        self.class_weights = self.get_class_weights()
        self.init_bias_correction = self.get_init_bias_correction(dataset, self.train_indexes, self.num_train_samples)
        self.train_scalers = self.get_train_scalers(dataset, self.train_indexes)

    @staticmethod
    def get_train_val_split(training_dataset, limit=-1):
        all_indexes = []
        with training_dataset as dataset:
            total_samples = 0
            for i, song_start_index, song_end_index in enumerate(dataset.song_index_ranges):
                if not any(dataset.labels[song_start_index: song_end_index] < 0):
                    all_indexes.append(i)
                    total_samples += song_end_index - song_start_index
                    if 0 < limit < total_samples:
                        break
        all_indexes = np.array(all_indexes)
        train_indexes, val_indexes, _, _ = \
            train_test_split(all_indexes,
                             all_indexes,
                             test_size=0.2,
                             shuffle=True,
                             random_state=42)
        return all_indexes, train_indexes, val_indexes

    def get_class_weights(self):
        # TODO: Implement later
        # Should be something like this
        #         class_weights = {0: (dataset.num_valid_samples / dataset.neg_samples) / 2.0,
        #                          1: (dataset.num_valid_samples / dataset.pos_samples) / 2.0}
        return None

    @staticmethod
    def get_init_bias_correction(training_dataset, train_indexes, num_train_samples):
        # Best practices mentioned in
        # https://www.tensorflow.org/tutorials/structured_data/imbalanced_data#optional_set_the_correct_initial_bias
        # Not completely correct but works for now
        num_all = num_train_samples
        num_pos = 0
        with training_dataset as dataset:
            for index in train_indexes:
                song_start_index, song_end_index = dataset.song_index_ranges[index]
                num_pos += dataset.labels[song_end_index:song_end_index].sum()
        num_neg = num_all - num_pos
        return np.log(num_pos / num_neg)

    @staticmethod
    def get_train_scalers(training_dataset, train_indexes):
        training_scalers = None
        with training_dataset as dataset:
            for index in train_indexes:
                song_start_index, song_end_index = dataset.song_index_ranges[index]
                features = dataset.features[range(song_start_index, song_end_index)]
                training_scalers = get_channel_scalers(features, existing_scalers=training_scalers, n_jobs=-1)
        return training_scalers

    @staticmethod
    def get_num_samples(training_dataset, indexes):
        num_all = 0
        with training_dataset as dataset:
            for index in indexes:
                song_start_index, song_end_index = dataset.song_index_ranges[index]
                num_all += song_end_index - song_start_index
        return num_all

    @property
    def arrow_input_shape(self):
        return (self.lookback,)

    @property
    def arrow_mask_shape(self):
        return (self.lookback,)

    @property
    def audio_input_shape(self):
        # return lookback + 1 to include current audio sample
        return (self.lookback + 1, self.dataset_config["NUM_TIME_BANDS"], self.dataset_config["NUM_FREQ_BANDS"], 1)

    @property
    def label_shape(self):
        return (4 * self.dataset_config["NUM_ARROW_TYPES"])
