import joblib
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils import compute_class_weight

from stepcovnet.common.utils import pre_process
from stepcovnet.training.parameters import BATCH_SIZE


class FeatureGenerator(tf.keras.utils.Sequence):
    def __init__(self, x, y, sample_weights=None, multi=False, scaler=None, extra_features=None, shuffle=True):
        self.x, self.y, self.sample_weight, self.extra_features = x, y, sample_weights, extra_features
        self.scaler = scaler
        self.multi = multi
        self.shuffle = shuffle
        self.indexes = range(len(self.y))
        self.num_batches = int(np.ceil(len(self.y) / BATCH_SIZE))
        self.indexes_batch = [self.indexes[i * BATCH_SIZE:(i + 1) * BATCH_SIZE] for i in range(self.num_batches)]
        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(len(self.indexes_batch))

    def __getitem__(self, index):
        """Generate one batch of data"""
        indexes = self.indexes_batch[index]
        extra_features = None if self.extra_features is None else self.extra_features[indexes]
        sample_weight = None if self.sample_weight is None else self.sample_weight[indexes]
        x, y = pre_process(self.x[indexes], multi=self.multi, labels=self.y[indexes], scalers=self.scaler,
                           extra_features=extra_features)
        return x, y, sample_weight

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes_batch)


def get_init_expected_loss(num_pos, num_all):
    p0 = num_pos / num_all
    return -p0 * np.log(p0) - (1 - p0) * np.log(1 - p0)


def get_init_bias_correction(num_pos, num_all):
    num_neg = num_all - num_pos
    return np.log(num_pos / num_neg)


def get_split_indexes(labels, multi):
    indices_all = range(len(labels))
    if multi:
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


def load_data(filename_features,
              filename_extra_features,
              filename_labels,
              filename_sample_weights,
              filename_scalers,
              filename_pretrained_model):
    # load training and validation data
    with open(filename_features, 'rb') as f:
        features = joblib.load(f)

    if filename_extra_features is not None:
        with open(filename_extra_features, 'rb') as f:
            extra_features = joblib.load(f)
    else:
        extra_features = None

    with open(filename_labels, 'rb') as f:
        labels = joblib.load(f)

    with open(filename_sample_weights, 'rb') as f:
        sample_weights = joblib.load(f)

    scaler = []
    for filename_scaler in filename_scalers:
        with open(filename_scaler, 'rb') as f:
            scaler.append(joblib.load(f))
    scaler = np.array(scaler)

    if filename_pretrained_model is not None:
        from tensorflow.keras.models import load_model
        pretrained_model = load_model(filename_pretrained_model, compile=False)
    else:
        pretrained_model = None

    class_weights = compute_class_weight('balanced', [0, 1], labels)

    class_weights = {0: class_weights[0] / 2, 1: class_weights[1] / 2}

    return features, extra_features, labels, sample_weights, class_weights, scaler, pretrained_model
