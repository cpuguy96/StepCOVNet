from __future__ import annotations

import os
from collections import defaultdict

import h5py
import numpy as np


class ModelDataset:
    def __init__(
        self,
        dataset_name: str,
        overwrite: bool = False,
        mode: str = "a",
        difficulty: str = "challenge",
    ):
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
        # ensure these dataset names are somewhat unique
        self.dataset_names = [
            "features",
            "labels",
            "sample_weights",
            "arrows",
            "label_encoded_arrows",
            "binary_encoded_arrows",
            "string_arrows",
            "onehot_encoded_arrows",
            "file_names",
            "song_index_ranges",
        ]
        self.difficulty_dataset_names = [
            "labels",
            "sample_weights",
            "arrows",
            "label_encoded_arrows",
            "binary_encoded_arrows",
            "string_arrows",
            "onehot_encoded_arrows",
        ]
        self.scaler_dataset_names = ["file_names"]
        self.dataset_attr = {
            "labels": {"num_valid_samples", "pos_samples", "neg_samples"},
            "features": {"num_samples"},
        }
        self.difficulties = {"challenge", "hard", "medium", "easy", "beginner"}
        self.difficulty = difficulty
        self.h5py_file: h5py.File | None = None

    def __getitem__(self, item) -> list:
        data = [
            self.features[item],
            self.labels[item],
            self.sample_weights[item],
            self.arrows[item],
            self.label_encoded_arrows[item],
            self.binary_encoded_arrows[item],
            self.string_arrows[item],
            self.onehot_encoded_arrows[item],
        ]
        return data

    def __len__(self) -> int:
        try:
            return self.num_samples
        except KeyError:
            return 0

    def __enter__(self) -> ModelDataset:
        self.reset_h5py_file()
        self.set_difficulty(difficulty=self.difficulty)
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        self.h5py_file.flush()
        self.h5py_file.close()

    def reset_h5py_file(self):
        if self.h5py_file is not None:
            try:
                self.h5py_file.close()
            except IOError:
                pass
        self.h5py_file: h5py.File = h5py.File(
            self.dataset_path, self.mode, libver="latest"
        )

    def create_dataset(self, data: np.ndarray, dataset_name: str):
        if dataset_name in self.scaler_dataset_names:
            self.h5py_file.create_dataset(
                dataset_name, data=data, compression="lzf", dtype="S1024"
            )
        else:
            if len(data.shape) > 1:
                data_shape = (None,) + data.shape[1:]
            else:
                data_shape = (None,)
            self.h5py_file.create_dataset(
                dataset_name,
                data=data,
                chunks=True,
                compression="lzf",
                maxshape=data_shape,
            )

    def extend_dataset(self, data: np.ndarray, dataset_name: str):
        if dataset_name in self.scaler_dataset_names:
            saved_dataset = self.h5py_file[dataset_name][:]
            del self.h5py_file[dataset_name]
            self.create_dataset(
                data=np.append(saved_dataset, [data]), dataset_name=dataset_name
            )
        else:
            self.h5py_file[dataset_name].resize(
                (self.h5py_file[dataset_name].shape[0] + data.shape[0]), axis=0
            )
            self.h5py_file[dataset_name][-data.shape[0] :] = data

    def dump_difficulty_dataset(
        self, dataset_name: str, difficulty: str, value: np.ndarray
    ):
        difficulty_dataset_name = self.append_difficulty(
            dataset_name=dataset_name, difficulty=difficulty
        )
        if not self.h5py_file.get(difficulty_dataset_name):
            self.create_dataset(value, difficulty_dataset_name)
        else:
            self.extend_dataset(value, difficulty_dataset_name)
        saved_attributes = self.save_attributes(self.h5py_file, difficulty_dataset_name)
        self.set_dataset_attrs(
            self.h5py_file, difficulty_dataset_name, saved_attributes
        )
        self.update_dataset_attrs(self.h5py_file, difficulty_dataset_name, value)

    def dump(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        sample_weights: np.ndarray,
        arrows: np.ndarray,
        label_encoded_arrows: np.ndarray,
        binary_encoded_arrows: np.ndarray,
        file_names: np.ndarray,
        string_arrows: np.ndarray,
        onehot_encoded_arrows: np.ndarray,
    ):
        try:
            all_data = self.get_dataset_name_to_data_map(
                features=features,
                labels=labels,
                sample_weights=sample_weights,
                arrows=arrows,
                label_encoded_arrows=label_encoded_arrows,
                binary_encoded_arrows=binary_encoded_arrows,
                string_arrows=string_arrows,
                onehot_encoded_arrows=onehot_encoded_arrows,
                file_names=file_names,
                song_index_ranges=[[len(self), len(self) + len(features)]],
            )
            for dataset_name, data in all_data.items():
                if data is None:
                    continue
                if dataset_name in self.difficulty_dataset_names and isinstance(
                    data, (dict, defaultdict)
                ):
                    diff_copy = self.difficulties.copy()
                    for difficulty, value in data.items():
                        if difficulty in diff_copy:
                            diff_copy.remove(difficulty)
                        self.dump_difficulty_dataset(dataset_name, difficulty, value)
                    data_shape = data[next(iter(data))].shape
                    dtype = data[next(iter(data))].dtype
                    if dtype == np.dtype("S4"):
                        null_values = np.chararray(data_shape, itemsize=4)
                        null_values[:] = "0000"
                    else:
                        null_values = np.full(data_shape, fill_value=-1)
                    for remaining_diff in diff_copy:
                        self.dump_difficulty_dataset(
                            dataset_name, remaining_diff, null_values
                        )
                    continue
                else:
                    if not self.h5py_file.get(dataset_name):
                        self.create_dataset(data, dataset_name)
                    else:
                        self.extend_dataset(data, dataset_name)
                    saved_attributes = self.save_attributes(
                        self.h5py_file, dataset_name
                    )
                    self.set_dataset_attrs(
                        self.h5py_file, dataset_name, saved_attributes
                    )
                    self.update_dataset_attrs(self.h5py_file, dataset_name, data)
            self.h5py_file.flush()
        except Exception as ex:
            self.close()
            raise ex

    def set_difficulty(self, difficulty: str):
        if difficulty not in self.difficulties:
            raise ValueError(
                "%s is not a valid difficulty! Choose a valid difficulty: %s"
                % (difficulty, self.difficulties)
            )
        self.difficulty = difficulty

    def get_dataset_name_to_data_map(self, **data_dict) -> dict:
        dataset_name_to_data_map = {}
        for dataset_name in self.dataset_names:
            if dataset_name in data_dict:
                data = data_dict[dataset_name]
                if isinstance(data, str):
                    data = np.array([data], dtype="S1024")
                elif isinstance(data, list):
                    data = np.array(data)
                dataset_name_to_data_map[dataset_name] = data
            else:
                dataset_name_to_data_map[dataset_name] = None
        return dataset_name_to_data_map

    def set_dataset_attrs(
        self,
        h5py_file: h5py.File,
        dataset_name: str,
        saved_attributes: dict | None = None,
    ):
        if "labels" in dataset_name:
            for dataset_attr in self.dataset_attr["labels"]:
                if saved_attributes is not None and dataset_attr in saved_attributes:
                    h5py_file[dataset_name].attrs[dataset_attr] = saved_attributes[
                        dataset_attr
                    ]
                else:
                    h5py_file[dataset_name].attrs[dataset_attr] = 0
        if "features" in dataset_name:
            for dataset_attr in self.dataset_attr["features"]:
                if saved_attributes is not None and dataset_attr in saved_attributes:
                    h5py_file[dataset_name].attrs[dataset_attr] = saved_attributes[
                        dataset_attr
                    ]
                else:
                    h5py_file[dataset_name].attrs[dataset_attr] = 0

    @staticmethod
    def update_dataset_attrs(
        h5py_file: h5py.File, dataset_name: str, attr_value: np.ndarray
    ):
        if "labels" in dataset_name:
            if not any(attr_value < 0):
                h5py_file[dataset_name].attrs["num_valid_samples"] += len(attr_value)
                h5py_file[dataset_name].attrs["pos_samples"] += attr_value.sum()
                h5py_file[dataset_name].attrs["neg_samples"] += (
                    len(attr_value) - attr_value.sum()
                )
        elif "features" in dataset_name:
            h5py_file[dataset_name].attrs["num_samples"] += len(attr_value)

    @staticmethod
    def save_attributes(h5py_file: h5py.File, dataset_name: str) -> dict:
        saved_attributes = {}
        if dataset_name in h5py_file:
            for attr_name in h5py_file[dataset_name].attrs:
                saved_attributes[attr_name] = h5py_file[dataset_name].attrs[attr_name]
        return saved_attributes

    @staticmethod
    def append_file_type(path: str) -> str:
        return path + ".hdf5"

    @staticmethod
    def append_difficulty(dataset_name: str, difficulty: str) -> str:
        return "%s_%s" % (dataset_name, difficulty)

    @property
    def num_samples(self) -> int:
        return self.h5py_file["features"].attrs["num_samples"]

    @property
    def num_valid_samples(self) -> int:
        return self.h5py_file[self.append_difficulty("labels", self.difficulty)].attrs[
            "num_valid_samples"
        ]

    @property
    def pos_samples(self) -> int:
        return self.h5py_file[self.append_difficulty("labels", self.difficulty)].attrs[
            "pos_samples"
        ]

    @property
    def neg_samples(self) -> int:
        return self.h5py_file[self.append_difficulty("labels", self.difficulty)].attrs[
            "neg_samples"
        ]

    @property
    def labels(self) -> np.ndarray:
        return self.h5py_file[self.append_difficulty("labels", self.difficulty)]

    @property
    def sample_weights(self) -> np.ndarray:
        return self.h5py_file[self.append_difficulty("sample_weights", self.difficulty)]

    @property
    def arrows(self) -> np.ndarray:
        return self.h5py_file[self.append_difficulty("arrows", self.difficulty)]

    @property
    def label_encoded_arrows(self) -> np.ndarray:
        return self.h5py_file[
            self.append_difficulty("label_encoded_arrows", self.difficulty)
        ]

    @property
    def binary_encoded_arrows(self) -> np.ndarray:
        return self.h5py_file[
            self.append_difficulty("binary_encoded_arrows", self.difficulty)
        ]

    @property
    def string_arrows(self) -> np.ndarray:
        return self.h5py_file[self.append_difficulty("string_arrows", self.difficulty)]

    @property
    def onehot_encoded_arrows(self) -> np.ndarray:
        return self.h5py_file[
            self.append_difficulty("onehot_encoded_arrows", self.difficulty)
        ]

    @property
    def file_names(self) -> list[str]:
        return [file_name.decode("ascii") for file_name in self.h5py_file["file_names"]]

    @property
    def song_index_ranges(self) -> tuple[int, int]:
        return self.h5py_file["song_index_ranges"]

    @property
    def features(self) -> np.ndarray:
        return self.h5py_file["features"]


class DistributedModelDataset(ModelDataset):
    def __init__(self, *args, **kwargs):
        super(DistributedModelDataset, self).__init__(*args, **kwargs)

    def dump(self, *args, **kwargs):
        sub_dataset_name = self.format_sub_dataset_name(kwargs["file_names"])
        sub_dataset_path = self.append_file_type(sub_dataset_name)
        if os.path.isfile(sub_dataset_path):
            os.remove(sub_dataset_path)
        self.h5py_file = h5py.File(sub_dataset_path, self.mode, libver="latest")
        super(DistributedModelDataset, self).dump(*args, **kwargs)
        try:
            sub_dataset_names = self.file_names + [sub_dataset_name]
        except KeyError:
            sub_dataset_names = [sub_dataset_name]
        self.build_dataset(sub_dataset_names, self.h5py_file)
        self.reset_h5py_file()

    def format_sub_dataset_name(self, file_name: str) -> str:
        return "%s_%s" % (self.dataset_name, file_name)

    def build_dataset(self, sub_dataset_names: list[str], h5py_file: h5py.File):
        if not sub_dataset_names:
            raise ValueError("Cannot build dataset until data is dumped")
        virtual_dataset = h5py.File(self.dataset_path, self.mode, libver="latest")
        for dataset_name in self.dataset_names:
            if dataset_name in self.difficulty_dataset_names:
                for difficulty in self.difficulties:
                    difficulty_dataset_name = self.append_difficulty(
                        dataset_name, difficulty
                    )
                    self.build_virtual_dataset(
                        data=h5py_file[difficulty_dataset_name][:],
                        dataset_name=difficulty_dataset_name,
                        sub_dataset_names=sub_dataset_names,
                        virtual_dataset=virtual_dataset,
                    )
            else:
                self.build_virtual_dataset(
                    data=h5py_file[dataset_name][:],
                    dataset_name=dataset_name,
                    sub_dataset_names=sub_dataset_names,
                    virtual_dataset=virtual_dataset,
                )

    def build_virtual_dataset(
        self,
        data: np.ndarray,
        dataset_name: str,
        sub_dataset_names: list[str],
        virtual_dataset: h5py.File,
    ):
        (
            virtual_sources,
            virtual_source_shape,
            virtual_dtype,
        ) = self.build_virtual_sources(dataset_name, sub_dataset_names)
        virtual_layout = self.build_virtual_layout(
            virtual_sources, virtual_source_shape, virtual_dtype
        )
        saved_attributes = self.save_attributes(virtual_dataset, dataset_name)
        if dataset_name in virtual_dataset:
            del virtual_dataset[dataset_name]
        virtual_dataset.create_virtual_dataset(dataset_name, virtual_layout)
        self.set_dataset_attrs(virtual_dataset, dataset_name, saved_attributes)
        self.update_dataset_attrs(virtual_dataset, dataset_name, data)

    def build_virtual_sources(
        self, dataset_name: str, sub_dataset_names: list[str]
    ) -> tuple[list[h5py.VirtualSource], tuple[int, ...], np.dtype | None]:
        sources = []
        dtype: np.dtype | None = None
        source_shape = None
        for sub_dataset_name in sub_dataset_names:
            with h5py.File(self.append_file_type(sub_dataset_name), "r") as sub_dataset:
                vsource = h5py.VirtualSource(sub_dataset[dataset_name])
                if source_shape is None:
                    source_shape = vsource.shape
                else:
                    source_shape = (source_shape[0] + vsource.shape[0],)
                    if len(vsource.shape) > 1:
                        source_shape = source_shape + vsource.shape[1:]
                dtype = sub_dataset[dataset_name].dtype if dtype is None else dtype
                sources.append(vsource)

        return sources, source_shape, dtype

    @staticmethod
    def build_virtual_layout(
        sources: list[h5py.VirtualSource],
        source_shapes: tuple[int, ...],
        dtype: np.dtype,
    ) -> h5py.VirtualLayout:
        virtual_layout = h5py.VirtualLayout(shape=source_shapes, dtype=dtype)
        offset = 0
        for source in sources:
            length = source.shape[0]
            virtual_layout[offset : offset + length] = source
            offset += length

        return virtual_layout

    @property
    def file_names(self) -> list[str]:
        return [
            self.format_sub_dataset_name(file_name.decode("ascii"))
            for file_name in self.h5py_file["file_names"]
        ]
