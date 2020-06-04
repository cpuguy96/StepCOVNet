import os

import h5py
import numpy as np


class ModelDataset(object):
    def __init__(self, dataset_path, overwrite=False, mode='a'):
        self.dataset_path = dataset_path
        self.overwrite = overwrite
        self.mode = mode
        if self.overwrite and self.mode not in {"a", "r+", "w", "w+", "x", "w-"}:
            raise ValueError("Mode must be a, r+, w, or w+ while in overwrite mode!")
        if self.overwrite and os.path.isfile(self.dataset_path):
            os.remove(self.dataset_path)
        if not self.overwrite:
            self.mode = "r"
        self.h5py_file = h5py.File(self.dataset_path, self.mode)
        self.dataset_names = ["features", "labels", "sample_weights", "extra_features", "arrows"]
        self.dataset_attr = {"num_samples", "num_valid_samples", "pos_samples", "neg_samples"}
        self.difficulties = {"challenge", "hard", "medium", "easy", "beginner"}
        self.difficulty = "challenge"

    def __getitem__(self, item):
        data = [self.features[item], self.labels[item], self.sample_weights[item], self.arrows[item]]
        try:
            data.append(self.extra_features[item])
        except Exception as ex:
            data.append(None)
        return data

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def create_dataset(self, data, dataset_name):
        if len(data.shape) > 1:
            data_shape = (None,) + data.shape[1:]
        else:
            data_shape = (None,)
        self.h5py_file.create_dataset(dataset_name, data=data, chunks=True, compression="lzf",
                                      maxshape=data_shape)

    def extend_dataset(self, data, dataset_name):
        # TODO: Remove appending and split by file. Will need to change indexing to allow this.
        self.h5py_file[dataset_name].resize((self.h5py_file[dataset_name].shape[0] + data.shape[0]), axis=0)
        self.h5py_file[dataset_name][-data.shape[0]:] = data

    def dump_difficulty_dataset(self, dataset_name, difficulty, value):
        difficulty_dataset_name = dataset_name + "_" + difficulty
        if not self.h5py_file.get(difficulty_dataset_name):
            self.create_dataset(value, difficulty_dataset_name)
            if dataset_name == "labels":
                for dataset_attr in self.dataset_attr:
                    self.h5py_file[difficulty_dataset_name].attrs[dataset_attr] = 0
        else:
            self.extend_dataset(value, difficulty_dataset_name)
        if dataset_name == "labels":
            self.h5py_file[difficulty_dataset_name].attrs["num_samples"] += len(value)
        if dataset_name == "labels" and value[0] >= 0:
            self.h5py_file[difficulty_dataset_name].attrs["num_valid_samples"] += len(value)
            self.h5py_file[difficulty_dataset_name].attrs["pos_samples"] += value.sum()
            self.h5py_file[difficulty_dataset_name].attrs["neg_samples"] += len(value) - value.sum()

    def dump(self, features, labels, sample_weights, extra_features, arrows):
        try:
            num_labels = len(labels[list(labels.keys())[0]])
            all_data = [features, labels, sample_weights, extra_features, arrows]
            for dataset_name, data in zip(self.dataset_names, all_data):
                if data is None:
                    continue
                if dataset_name in {"labels", "sample_weights", "arrows"}:
                    diff_copy = self.difficulties.copy()
                    for difficulty, value in data.items():
                        if difficulty in diff_copy:
                            diff_copy.remove(difficulty)
                        self.dump_difficulty_dataset(dataset_name, difficulty, value)
                    null_values = np.zeros((num_labels,))
                    null_values.fill(-1)
                    for remaining_diff in diff_copy:
                        self.dump_difficulty_dataset(dataset_name, remaining_diff, null_values)
                elif not self.h5py_file.get(dataset_name):
                    self.create_dataset(data, dataset_name)
                else:
                    self.extend_dataset(data, dataset_name)
            self.h5py_file.flush()
        except Exception as ex:
            self.close()
            raise ex

    def close(self):
        self.h5py_file.flush()
        self.h5py_file.close()

    def set_difficulty(self, difficulty):
        if difficulty not in self.difficulties:
            raise ValueError(
                "%s is not a vaild difficulty! Choose a valid difficulty: %s" % (difficulty, self.difficulties))
        self.difficulty = difficulty

    @property
    def num_samples(self):
        return self.h5py_file["labels_" + self.difficulty].attrs["num_samples"]

    @property
    def num_valid_samples(self):
        return self.h5py_file["labels_" + self.difficulty].attrs["num_valid_samples"]

    @property
    def pos_samples(self):
        return self.h5py_file["labels_" + self.difficulty].attrs["pos_samples"]

    @property
    def neg_samples(self):
        return self.h5py_file["labels_" + self.difficulty].attrs["neg_samples"]

    @property
    def labels(self):
        return self.h5py_file["labels_" + self.difficulty]

    @property
    def sample_weights(self):
        return self.h5py_file["sample_weights_" + self.difficulty]

    @property
    def arrows(self):
        return self.h5py_file["arrows_" + self.difficulty]

    @property
    def extra_features(self):
        if self.h5py_file.get("extra_features"):
            return self.h5py_file["extra_features"]
        else:
            return None

    @property
    def features(self):
        return self.h5py_file["features"]
