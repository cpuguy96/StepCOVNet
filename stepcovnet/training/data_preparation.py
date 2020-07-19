import pickle

import numpy as np
from sklearn.model_selection import train_test_split

from stepcovnet.common.utils import pre_process
from stepcovnet.dataset.ModelDataset import ModelDataset
from stepcovnet.training.parameters import BATCH_SIZE


class FeatureGenerator(object):
    def __init__(self, dataset_path, indexes, multi, difficulty, scaler=None, shuffle=True):
        self.dataset_path = dataset_path
        self.scaler = scaler
        self.multi = multi
        self.difficulty = difficulty
        self.shuffle = shuffle
        self.num_batches = int(np.ceil(len(indexes) / BATCH_SIZE))
        self.indexes_batch = [indexes[i * BATCH_SIZE:(i + 1) * BATCH_SIZE] for i in range(self.num_batches)]
        if self.shuffle:
            np.random.shuffle(self.indexes_batch)
        self.batch_index = 0

    def __len__(self):
        return self.num_batches

    def __call__(self, *args, **kwargs):
        with ModelDataset(self.dataset_path) as dataset:
            dataset.set_difficulty(self.difficulty)
            while True:
                if self.batch_index >= len(self.indexes_batch):
                    self.batch_index = 0
                batch = self.indexes_batch[self.batch_index]
                self.batch_index += 1
                feature_batch, label_batch, sample_weight_batch, arrows_batch, = dataset[batch]
                x_batch, y_batch = pre_process(features=feature_batch, labels=label_batch, multi=self.multi,
                                               scalers=self.scaler)
                yield x_batch, y_batch, sample_weight_batch


def get_split_indexes(labels, timeseries=False, limit=-1):
    indices_all = list(np.where(np.array(labels) >= 0)[0])
    if limit > 0:
        indices_all = indices_all[:limit]
    if True or timeseries:  # removing stratify splits until we can get it working
        indices_train, indices_validation, _, _ = \
            train_test_split(indices_all,
                             indices_all,
                             test_size=0.2,
                             shuffle=False,
                             random_state=42)
    else:
        indices_train, indices_validation, _, _ = \
            train_test_split(indices_all,
                             indices_all,
                             test_size=0.2,
                             stratify=labels,
                             shuffle=True,
                             random_state=42)

    return indices_all, indices_train, indices_validation


def load_data(filename_scalers,
              filename_pretrained_model):
    scaler = []
    for filename_scaler in filename_scalers:
        try:
            with open(filename_scaler, 'rb') as f:
                scaler.append(pickle.load(f))
        except Exception as ex:
            print("Error occured while loading scalers. Defaulting to not using. Exception: %r" % ex)
            scaler = None
            break
    if scaler is not None and len(scaler) == 1:
        scaler = [sca for sca in scaler[0]]  # flatten if there is only 1 channel

    if filename_pretrained_model is not None:
        from tensorflow.keras.models import load_model
        pretrained_model = load_model(filename_pretrained_model, compile=False)
    else:
        pretrained_model = None

    return scaler, pretrained_model
