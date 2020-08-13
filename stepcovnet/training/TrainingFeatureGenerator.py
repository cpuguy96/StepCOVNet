from collections import defaultdict

import numpy as np

from stepcovnet.common.utils import apply_timeseries_scalers
from stepcovnet.common.utils import get_ngram


class TrainingFeatureGenerator(object):
    def __init__(self, dataset_path, dataset_type, batch_size, indexes, num_samples, lookback=1, scalers=None,
                 difficulty="challenge", warmup=False):
        self.dataset_path = dataset_path
        self.dataset_type = dataset_type
        self.train_indexes = indexes
        self.num_samples = num_samples
        self.scalers = scalers
        self.lookback = lookback
        self.batch_size = batch_size
        self.difficulty = difficulty

        # self.song_index = self.train_indexes[0]
        # self.song_start_index = None
        # The Tensorflow calls the generator three times before starting a training job. We will "warmup" the data
        # yielding by returning the same data for the three calls. This way the indexing is aligned correctly.
        self.warmup_countdown = 3 if warmup else 0
        # add shuffling

    def __len__(self):
        return int(np.ceil(self.num_samples / self.batch_size))

    def __call__(self):
        with self.dataset_type(self.dataset_path) as dataset:
            dataset.set_difficulty(self.difficulty)
            self.song_index = 0
            self.song_start_index = None
            while True:
                features = defaultdict(lambda: np.array([]))
                y_batch = None
                sample_weights_batch = None
                if self.song_index >= len(self.train_indexes):
                    self.song_index = 0
                song_start_index, song_end_index = dataset.song_index_ranges[self.train_indexes[self.song_index]]
                if self.song_start_index is None or self.song_start_index >= song_end_index or \
                        self.song_start_index < song_start_index or self.warmup_countdown > 0:
                    self.song_start_index = song_start_index
                    self.warmup_countdown -= 1
                # We only return partial batches when at the end of the training data
                # Just to make it easy, let's allow partial batches for now
                while y_batch is None or self.song_index < len(self.train_indexes):
                    start_index = self.song_start_index
                    y_batch_len = 0 if y_batch is None else len(y_batch)
                    end_index = min(start_index + self.batch_size - y_batch_len, song_end_index)
                    arrow_features, arrow_mask = self.get_samples_ngram_with_mask(dataset.label_encoded_arrows,
                                                                                  start_index, end_index,
                                                                                  reshape=True)
                    audio_features, _ = self.get_samples_ngram_with_mask(dataset.features,
                                                                         start_index, end_index, squeeze=False)
                    # Lookback data from ngram returns empty value in index 0. Also, arrow features should only
                    # contain previously seen features. Therefore, removing last element and last lookback from
                    # arrows features and first element from audio features.
                    if "arrow_features" in features:
                        features["arrow_features"] = np.concatenate((features["arrow_features"],
                                                                     arrow_features[:-1, 1:]), axis=0)
                    else:
                        features["arrow_features"] = arrow_features[:-1, 1:]
                    if "arrow_mask" in features:
                        features["arrow_mask"] = np.concatenate((features["arrow_mask"], arrow_mask[:-1, 1:]),
                                                                axis=0)
                    else:
                        features["arrow_mask"] = arrow_mask[:-1, 1:]
                    if "audio_features" in features:
                        features["audio_features"] = np.concatenate(
                            (features["audio_features"], audio_features[1:]),
                            axis=0)
                    else:
                        features["audio_features"] = audio_features[1:]
                    if y_batch is not None:
                        y_batch = np.concatenate((y_batch, dataset.binary_encoded_arrows[start_index: end_index]),
                                                 axis=0)
                    else:
                        y_batch = dataset.binary_encoded_arrows[start_index: end_index]
                    if sample_weights_batch is not None:
                        sample_weights_batch = np.concatenate((sample_weights_batch,
                                                               dataset.sample_weights[start_index: end_index]),
                                                              axis=0)
                    else:
                        sample_weights_batch = dataset.sample_weights[start_index: end_index]

                    self.song_start_index = end_index
                    if len(y_batch) >= self.batch_size or self.song_index + 1 >= len(self.train_indexes):
                        break
                    else:
                        self.song_index += 1
                        song_start_index, song_end_index = dataset.song_index_ranges[
                            self.train_indexes[self.song_index]]
                        self.song_start_index = song_start_index

                if self.song_start_index >= song_end_index:
                    self.song_index += 1

                if y_batch is not None:
                    scaled_audio_features = apply_timeseries_scalers(features=features["audio_features"],
                                                                     scalers=self.scalers)
                    x_batch = {"arrow_input": features["arrow_features"].astype(np.int32),
                               "arrow_mask": features["arrow_mask"].astype(np.int32),
                               "audio_input": scaled_audio_features.astype(np.float64)}
                    yield x_batch, y_batch, sample_weights_batch

    def get_samples_ngram_with_mask(self, dataset, start_index, end_index, squeeze=True, reshape=False):
        samples = dataset[start_index:end_index]

        if reshape:
            ngram_samples = get_ngram(samples.reshape(-1, 1), self.lookback)
        else:
            ngram_samples = get_ngram(samples, self.lookback)
        mask = np.zeros((samples.shape[0], 1), dtype=int)
        ngram_mask = get_ngram(mask, self.lookback, padding_value=1)

        if squeeze:
            ngram_samples = np.squeeze(ngram_samples)
        ngram_mask = np.squeeze(ngram_mask)

        return ngram_samples, ngram_mask
