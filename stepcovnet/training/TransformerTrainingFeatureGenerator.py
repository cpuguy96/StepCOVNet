from collections import defaultdict

import numpy as np

from stepcovnet.common.utils import apply_timeseries_scalers
from stepcovnet.data.Tokenizers import Tokenizers
from stepcovnet.training.TrainingFeatureGenerator import TrainingFeatureGenerator


class TransformerTrainingFeatureGenerator(TrainingFeatureGenerator):
    def __init__(self, dataset_path, dataset_type, batch_size, indexes, num_samples, lookback=1, scalers=None,
                 difficulty="challenge", warmup=False, shuffle=True, tokenizer_name=None):
        super(TransformerTrainingFeatureGenerator, self).__init__(
            dataset_path, dataset_type, batch_size, indexes, num_samples, lookback, scalers,
            difficulty, warmup, shuffle, tokenizer_name)
        self.tokenizer = None if tokenizer_name is None else Tokenizers[tokenizer_name].value

    def get_x_batch(self, features, audio_features) -> dict:
        return {"arrow_input": features["arrow_features"],
                "arrow_mask": features["arrow_mask"],
                "audio_input": audio_features}

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
                    lookback_index_padding_start = max(start_index - self.lookback, song_start_index)
                    lookback_padding_added = start_index - lookback_index_padding_start
                    if self.tokenizer is not None:
                        arrows = dataset.string_arrows[lookback_index_padding_start:end_index]
                        arrow_features, arrow_mask = self.get_tokenized_arrow_features(arrows, mask_padding_value,
                                                                                       lookback_padding_added)
                    else:
                        arrows = dataset.label_encoded_arrows[lookback_index_padding_start:end_index]
                        arrow_features, arrow_mask = self.get_arrow_features(arrows, mask_padding_value,
                                                                             lookback_padding_added)

                    audio_data = dataset.features[lookback_index_padding_start:end_index]
                    audio_features = self.get_audio_features(audio_data, lookback_padding_added)

                    arrows = dataset.onehot_encoded_arrows[start_index:end_index]
                    sample_weights = dataset.sample_weights[start_index:end_index]

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
                    new_song = True
                    self.song_index += 1

                if len(features["y_batch"]) > 0:
                    scaled_audio_features = apply_timeseries_scalers(features=features["audio_features"],
                                                                     scalers=self.scalers)
                    x_batch = {"arrow_input": features["arrow_features"],
                               "arrow_mask": features["arrow_mask"],
                               "audio_input": scaled_audio_features}
                    yield x_batch, features["y_batch"], features["sample_weights_batch"]
