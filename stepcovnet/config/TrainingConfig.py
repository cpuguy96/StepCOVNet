import numpy as np
from sklearn.model_selection import train_test_split

from stepcovnet.common.utils import get_channel_scalers
from stepcovnet.config.AbstractConfig import AbstractConfig


class TrainingConfig(AbstractConfig):
    def __init__(self, dataset_path, dataset_type, dataset_config, hyperparameters, all_scalers=None, limit=-1,
                 lookback=1, difficulty="challenge"):
        super(TrainingConfig, self).__init__(dataset_config=dataset_config, lookback=lookback, difficulty=difficulty)
        self.dataset_path = dataset_path
        self.dataset_type = dataset_type
        self.hyperparameters = hyperparameters
        self.all_scalers = all_scalers
        self.limit = limit

        # Combine some of these to reduce the number of loops and save I/O reads
        self.all_indexes, self.train_indexes, self.val_indexes = self.get_train_val_split()
        self.num_samples = self.get_num_samples(self.all_indexes)
        self.num_train_samples = self.get_num_samples(self.train_indexes)
        self.num_val_samples = self.get_num_samples(self.val_indexes)
        # Disabling class weights since model currently performs better when disabled.
        self.train_class_weights = None  # self.get_class_weights(self.train_indexes)
        self.all_class_weights = None  # self.get_class_weights(self.all_indexes)
        self.init_bias_correction = self.get_init_bias_correction()
        self.train_scalers = self.get_train_scalers()

    def get_train_val_split(self):
        all_indexes = []
        with self.enter_dataset as dataset:
            total_samples = 0
            index = 0
            for song_start_index, song_end_index in dataset.song_index_ranges:
                if not any(dataset.labels[song_start_index: song_end_index] < 0):
                    all_indexes.append(index)
                    total_samples += song_end_index - song_start_index
                    if 0 < self.limit < total_samples:
                        break
                index += 1
        all_indexes = np.array(all_indexes)
        train_indexes, val_indexes, _, _ = \
            train_test_split(all_indexes,
                             all_indexes,
                             test_size=0.2,
                             shuffle=True,
                             random_state=42)
        return all_indexes, train_indexes, val_indexes

    def get_class_weights(self, indexes):
        labels = None
        with self.enter_dataset as dataset:
            for index in indexes:
                song_start_index, song_end_index = dataset.song_index_ranges[index]
                binary_encoded_arrows = dataset.binary_encoded_arrows[song_start_index:song_end_index]
                if labels is None:
                    labels = binary_encoded_arrows
                else:
                    labels = np.concatenate((labels, binary_encoded_arrows), axis=0)

        class_counts = [labels[:, class_index].sum() for class_index in range(labels.shape[1])]

        class_weights = dict(zip(
            list(range(len(class_counts))),
            list((len(labels) / class_count) / len(class_counts) for class_count in class_counts)
        ))

        return dict(enumerate(class_weights))

    def get_init_bias_correction(self):
        # Best practices mentioned in
        # https://www.tensorflow.org/tutorials/structured_data/imbalanced_data#optional_set_the_correct_initial_bias
        # Not completely correct but works for now
        num_all = self.num_train_samples
        num_pos = 0
        with self.enter_dataset as dataset:
            for index in self.train_indexes:
                song_start_index, song_end_index = dataset.song_index_ranges[index]
                num_pos += dataset.labels[song_start_index:song_end_index].sum()
        num_neg = num_all - num_pos
        return np.log(num_pos / num_neg)

    def get_train_scalers(self):
        training_scalers = None
        with self.enter_dataset as dataset:
            for index in self.train_indexes:
                song_start_index, song_end_index = dataset.song_index_ranges[index]
                features = dataset.features[range(song_start_index, song_end_index)]
                training_scalers = get_channel_scalers(features, existing_scalers=training_scalers, n_jobs=-1)
        return training_scalers

    def get_num_samples(self, indexes):
        num_all = 0
        with self.enter_dataset as dataset:
            for index in indexes:
                song_start_index, song_end_index = dataset.song_index_ranges[index]
                num_all += song_end_index - song_start_index
        return num_all

    @property
    def enter_dataset(self):
        return self.dataset_type(self.dataset_path, difficulty=self.difficulty).__enter__()
