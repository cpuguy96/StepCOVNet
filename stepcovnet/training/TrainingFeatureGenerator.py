import numpy as np

from stepcovnet.common.utils import apply_scalers


class TrainingFeatureGenerator(object):
    def __init__(self, dataset, batch_size, indexes, num_samples, scalers=None):
        self.dataset = dataset
        self.train_indexes = indexes
        self.num_samples = num_samples
        self.scalers = scalers
        self.num_batches = int(np.ceil(len(self.num_samples) / batch_size))
        self.batch_index = 0
        self.song_index = 0

    def __len__(self):
        return self.num_batches

    def __call__(self, *args, **kwargs):
        with self.dataset as dataset:
            while True:
                for song_index in self.train_indexes:
                    song_start_index, song_end_index = dataset.song_index_ranges[song_index]

                if self.batch_index >= len(self.indexes_batch):
                    self.batch_index = 0
                batch = self.indexes_batch[self.batch_index]
                self.batch_index += 1
                features_batch = dataset.features[batch]
                x_batch = apply_scalers(features=features_batch, scalers=self.scalers)
                y_batch = dataset.binary_encoded_arrows[batch]
                sample_weight_batch = dataset.sample_weights[batch]
                yield x_batch, y_batch, sample_weight_batch
