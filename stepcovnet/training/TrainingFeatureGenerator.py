from collections import defaultdict

import numpy as np

from stepcovnet.common.utils import apply_timeseries_scalers
from stepcovnet.common.utils import get_samples_ngram_with_mask


class TrainingFeatureGenerator(object):
    def __init__(self, dataset_path, dataset_type, batch_size, indexes, num_samples, lookback=1, scalers=None,
                 difficulty="challenge", warmup=False, shuffle=True, tokenizer=None):
        self.dataset_path = dataset_path
        self.dataset_type = dataset_type
        self.train_indexes = indexes
        self.num_samples = num_samples
        self.scalers = scalers
        self.lookback = lookback
        self.batch_size = batch_size
        self.difficulty = difficulty
        self.shuffle = shuffle
        # Removing use of tokenizer until we can supply varied input feature sizes
        self.tokenizer = None  # tokenizer

        # The Tensorflow calls the generator three times before starting a training job. We will "warmup" the data
        # yielding by returning the same data for the three calls. This way the indexing is aligned correctly.
        self.warmup_countdown = 3 if warmup else 0
        self.rng = np.random.default_rng(42)
        # add shuffling

    def __len__(self):
        return int(np.ceil(self.num_samples / self.batch_size))

    def __call__(self):
        with self.dataset_type(self.dataset_path) as dataset:
            dataset.set_difficulty(self.difficulty)
            self.song_index = 0
            self.song_start_index = None
            new_song = True
            if self.shuffle:
                self.rng.shuffle(self.train_indexes)
            while True:
                features = defaultdict(lambda: np.array([]))
                if self.song_index >= len(self.train_indexes):
                    self.song_index = 0
                song_start_index, song_end_index = dataset.song_index_ranges[self.train_indexes[self.song_index]]
                if self.song_start_index is None or self.song_start_index >= song_end_index or \
                        self.song_start_index < song_start_index or self.warmup_countdown > 0:
                    self.song_start_index = song_start_index
                    self.warmup_countdown = max(self.warmup_countdown - 1, 0)
                # We only return partial batches when at the end of the training data. Otherwise, use start of next song
                # to append data to the batch.
                while len(features["y_batch"]) == 0 or self.song_index < len(self.train_indexes):
                    start_index = self.song_start_index
                    y_batch_len = len(features["y_batch"])
                    end_index = min(start_index + self.batch_size - y_batch_len, song_end_index)

                    # Lookback data from ngram returns empty value in index 0. Also, arrow features should only
                    # contain previously seen features. Therefore, removing last element and last lookback from
                    # arrows features and first element from audio features.
                    mask_padding_value = 0 if new_song else 1
                    if self.tokenizer is not None:
                        string_arrows = dataset.string_arrows[start_index:end_index]
                        arrow_features, arrow_mask = get_samples_ngram_with_mask(string_arrows, self.lookback,
                                                                                 reshape=True,
                                                                                 sample_padding_value='0000',
                                                                                 mask_padding_value=mask_padding_value)
                        decoded_arrows = [" ".join(line) for line in arrow_features]
                        arrow_features = [self.tokenizer(line, return_tensors='tf',
                                                         add_prefix_space=True)['input_ids']
                                              .numpy()[0][1:].astype(np.int32)
                                          for line in decoded_arrows]
                        arrow_features = arrow_features[:-1]
                        arrow_mask = list(arrow_mask[:-1, 1:].astype(np.int32))
                        for i in range(len(arrow_mask)):
                            len_diff = len(arrow_features[i]) - len(arrow_mask[i])
                            if len_diff > 0:
                                arrow_mask[i] = np.concatenate((arrow_mask[i], np.zeros((len_diff,))), axis=0)
                            elif len_diff < 0:
                                arrow_mask[i] = arrow_mask[i][:-len_diff]
                    else:
                        encoded_arrows = dataset.label_encoded_arrows[start_index:end_index]
                        arrow_features, arrow_mask = get_samples_ngram_with_mask(encoded_arrows, self.lookback,
                                                                                 reshape=True,
                                                                                 mask_padding_value=mask_padding_value)
                        arrow_features = arrow_features[:-1, 1:].astype(np.int32)
                        arrow_mask = arrow_mask[:-1, 1:].astype(np.int32)
                    audio_features, _ = get_samples_ngram_with_mask(dataset.features[start_index:end_index],
                                                                    self.lookback, squeeze=False)
                    audio_features = audio_features[1:]
                    arrows = dataset.binary_encoded_arrows[start_index: end_index]
                    sample_weights = dataset.sample_weights[start_index: end_index]

                    features = self.append_existing_data(features=features, arrow_features=arrow_features,
                                                         arrow_mask=arrow_mask, audio_features=audio_features,
                                                         arrows=arrows, sample_weights=sample_weights)
                    self.song_start_index = end_index
                    # Break if collected enough data for a batch or end of song list.
                    # Otherwise, change to next song to collect more.
                    if len(features["y_batch"]) >= self.batch_size or self.song_index + 1 >= len(self.train_indexes):
                        new_song = False
                        break
                    else:
                        self.song_index += 1
                        song_start_index, song_end_index = \
                            dataset.song_index_ranges[self.train_indexes[self.song_index]]
                        self.song_start_index = song_start_index
                        new_song = True

                if self.song_start_index >= song_end_index:
                    self.song_index += 1

                if len(features["y_batch"]) > 0:
                    scaled_audio_features = apply_timeseries_scalers(features=features["audio_features"],
                                                                     scalers=self.scalers)
                    x_batch = {"arrow_input": features["arrow_features"],
                               "arrow_mask": features["arrow_mask"],
                               "audio_input": scaled_audio_features.astype(np.float64)}
                    yield x_batch, features["y_batch"], features["sample_weights_batch"]

    @staticmethod
    def append_existing_data(features, arrow_features, arrow_mask, audio_features, arrows,
                             sample_weights):
        # Append or set features/labels/sample weights based on if existing data is present
        if not features or any(len(value) == 0 for value in features.values()):
            features["arrow_features"] = arrow_features
            features["arrow_mask"] = arrow_mask
            features["audio_features"] = audio_features
            features["y_batch"] = arrows
            features["sample_weights_batch"] = sample_weights
        else:
            if isinstance(features["arrow_features"], list) or isinstance(features["arrow_mask"], list):
                features["arrow_features"].append(arrow_features)
                features["arrow_mask"].append(arrow_mask)
            else:
                features["arrow_features"] = np.concatenate((features["arrow_features"], arrow_features), axis=0)
                features["arrow_mask"] = np.concatenate((features["arrow_mask"], arrow_mask), axis=0)
            features["audio_features"] = np.concatenate((features["audio_features"], audio_features), axis=0)
            features["y_batch"] = np.concatenate((features["y_batch"], arrows), axis=0)
            features["sample_weights_batch"] = np.concatenate((features["sample_weights_batch"], sample_weights),
                                                              axis=0)

        return features
