import os

import h5py
import numpy as np


class ModelDataset(object):
    def __init__(self, dataset_name, overwrite=False, mode='a'):
        self.dataset_name = dataset_name
        self.dataset_path = self.append_file_type(self.dataset_name)
        self.overwrite = overwrite
        self.mode = mode
        if self.overwrite and self.mode not in {"a", "r+", "w", "w+", "x", "w-"}:
            raise ValueError("Mode must be a, r+, w, or w+ while in overwrite mode!")
        if self.overwrite and os.path.isfile(self.dataset_path):
            os.remove(self.dataset_path)
        if not self.overwrite:
            self.mode = "r"
        self.dataset_names = ["features", "labels", "sample_weights", "arrows", "encoded_arrows"]
        self.dataset_attr = {"labels": {"num_samples", "num_valid_samples", "pos_samples", "neg_samples"}}
        self.difficulties = {"challenge", "hard", "medium", "easy", "beginner"}
        self.difficulty = "challenge"
        self.h5py_file = None

    def __getitem__(self, item):
        data = [self.features[item], self.labels[item], self.sample_weights[item], self.arrows[item],
                self.encoded_arrows[item]]
        return data

    def __enter__(self):
        if self.h5py_file is None:
            self.h5py_file = h5py.File(self.dataset_path, self.mode)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def create_dataset(self, data, dataset_name):
        if len(data.shape) > 1:
            data_shape = (None,) + data.shape[1:]
        else:
            data_shape = (None,)
        self.h5py_file.create_dataset(dataset_name, data=data, chunks=True, compression="lzf", maxshape=data_shape)

    def extend_dataset(self, data, dataset_name):
        # TODO: Remove appending and split by file. Will need to change indexing to allow this.
        self.h5py_file[dataset_name].resize((self.h5py_file[dataset_name].shape[0] + data.shape[0]), axis=0)
        self.h5py_file[dataset_name][-data.shape[0]:] = data

    def dump_difficulty_dataset(self, dataset_name, difficulty, value):
        difficulty_dataset_name = self.append_difficulty(dataset_name=dataset_name, difficulty=difficulty)
        if not self.h5py_file.get(difficulty_dataset_name):
            self.create_dataset(value, difficulty_dataset_name)
        else:
            self.extend_dataset(value, difficulty_dataset_name)
        saved_attributes = self.save_attributes(self.h5py_file, difficulty_dataset_name)
        self.set_dataset_attrs(self.h5py_file, difficulty_dataset_name, saved_attributes)
        self.update_dataset_attrs(self.h5py_file, difficulty_dataset_name, value)

    def set_dataset_attrs(self, h5py_file, dataset_name, saved_attributes=None):
        if "labels" in dataset_name:
            for dataset_attr in self.dataset_attr["labels"]:
                if saved_attributes is not None and dataset_attr in saved_attributes:
                    h5py_file[dataset_name].attrs[dataset_attr] = saved_attributes[dataset_attr]
                else:
                    h5py_file[dataset_name].attrs[dataset_attr] = 0

    def dump(self, features, labels, sample_weights, arrows, encoded_arrows, **kwargs):
        try:
            all_data = [features, labels, sample_weights, arrows, encoded_arrows]
            for dataset_name, data in zip(self.dataset_names, all_data):
                if data is None:
                    continue
                if dataset_name in {"labels", "sample_weights", "arrows", "encoded_arrows"}:
                    diff_copy = self.difficulties.copy()
                    for difficulty, value in data.items():
                        if difficulty in diff_copy:
                            diff_copy.remove(difficulty)
                        self.dump_difficulty_dataset(dataset_name, difficulty, value)
                    data_shape = data[next(iter(data))].shape
                    null_values = np.full(data_shape, fill_value=-1)
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

    @staticmethod
    def save_attributes(h5py_file, dataset_name):
        saved_attributes = {}
        if dataset_name in h5py_file:
            for attr_name in h5py_file[dataset_name].attrs:
                saved_attributes[attr_name] = h5py_file[dataset_name].attrs[attr_name]
        return saved_attributes

    @staticmethod
    def update_dataset_attrs(h5py_file, dataset_name, attr_value):
        if "labels" in dataset_name:
            h5py_file[dataset_name].attrs["num_samples"] += len(attr_value)
            if not any(attr_value < 0):
                h5py_file[dataset_name].attrs["num_valid_samples"] += len(attr_value)
            h5py_file[dataset_name].attrs["pos_samples"] += attr_value.sum()
            h5py_file[dataset_name].attrs["neg_samples"] += len(attr_value) - attr_value.sum()

    @staticmethod
    def get_read_only_dataset(dataset_path):
        return ModelDataset(dataset_name=dataset_path, overwrite=False)

    @staticmethod
    def append_file_type(path):
        return path + '.hdf5'

    @staticmethod
    def append_difficulty(dataset_name, difficulty):
        return "%s_%s" % (dataset_name, difficulty)

    @property
    def num_samples(self):
        return self.h5py_file[self.append_difficulty("labels", self.difficulty)].attrs["num_samples"]

    @property
    def num_valid_samples(self):
        return self.h5py_file[self.append_difficulty("labels", self.difficulty)].attrs["num_valid_samples"]

    @property
    def pos_samples(self):
        return self.h5py_file[self.append_difficulty("labels", self.difficulty)].attrs["pos_samples"]

    @property
    def neg_samples(self):
        return self.h5py_file[self.append_difficulty("labels", self.difficulty)].attrs["neg_samples"]

    @property
    def labels(self):
        return self.h5py_file[self.append_difficulty("labels", self.difficulty)]

    @property
    def sample_weights(self):
        return self.h5py_file[self.append_difficulty("sample_weights", self.difficulty)]

    @property
    def arrows(self):
        return self.h5py_file[self.append_difficulty("arrows", self.difficulty)]

    @property
    def encoded_arrows(self):
        return self.h5py_file[self.append_difficulty("encoded_arrows", self.difficulty)]

    @property
    def features(self):
        return self.h5py_file["features"]
