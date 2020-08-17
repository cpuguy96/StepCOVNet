import os

import h5py

from stepcovnet.dataset.AbstractModelDataset import AbstractModelDataset


class DistributedModelDataset(AbstractModelDataset):
    def __init__(self, *args, **kwargs):
        super(DistributedModelDataset, self).__init__(*args, **kwargs)

    def dump(self, *args, **kwargs):
        sub_dataset_name = self.format_sub_dataset_name(kwargs["file_names"])
        sub_dataset_path = self.append_file_type(sub_dataset_name)
        if os.path.isfile(sub_dataset_path):
            os.remove(sub_dataset_path)
        self.h5py_file = h5py.File(sub_dataset_path, self.mode, libver='latest')
        super(DistributedModelDataset, self).dump(*args, **kwargs)
        try:
            sub_dataset_names = self.file_names + [sub_dataset_name]
        except KeyError:
            sub_dataset_names = [sub_dataset_name]
        self.build_dataset(sub_dataset_names, self.h5py_file)
        self.reset_h5py_file()

    def format_sub_dataset_name(self, file_name):
        return '%s_%s' % (self.dataset_name, file_name)

    def build_dataset(self, sub_dataset_names, h5py_file):
        if not sub_dataset_names:
            raise ValueError('Cannot build dataset until data is dumped')
        virtual_dataset = h5py.File(self.dataset_path, self.mode, libver='latest')
        for dataset_name in self.dataset_names:
            if dataset_name in self.difficulty_dataset_names:
                for difficulty in self.difficulties:
                    difficulty_dataset_name = self.append_difficulty(dataset_name, difficulty)
                    self.build_virtual_dataset(data=h5py_file[difficulty_dataset_name][:],
                                               dataset_name=difficulty_dataset_name,
                                               sub_dataset_names=sub_dataset_names, virtual_dataset=virtual_dataset)
            else:
                self.build_virtual_dataset(data=h5py_file[dataset_name][:], dataset_name=dataset_name,
                                           sub_dataset_names=sub_dataset_names, virtual_dataset=virtual_dataset)

    def build_virtual_dataset(self, data, dataset_name, sub_dataset_names, virtual_dataset):
        virtual_sources, virtual_source_shape, virtual_dtype = self.build_virtual_sources(dataset_name,
                                                                                          sub_dataset_names)
        virtual_layout = self.build_virtual_layout(virtual_sources, virtual_source_shape, virtual_dtype)
        saved_attributes = self.save_attributes(virtual_dataset, dataset_name)
        if dataset_name in virtual_dataset:
            del virtual_dataset[dataset_name]
        virtual_dataset.create_virtual_dataset(dataset_name, virtual_layout)
        self.set_dataset_attrs(virtual_dataset, dataset_name, saved_attributes)
        self.update_dataset_attrs(virtual_dataset, dataset_name, data)

    def build_virtual_sources(self, dataset_name, sub_dataset_names):
        sources = []
        dtype = None
        source_shape = None
        for sub_dataset_name in sub_dataset_names:
            with h5py.File(self.append_file_type(sub_dataset_name), 'r') as sub_dataset:
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
    def build_virtual_layout(sources, source_shapes, dtype):
        virtual_layout = h5py.VirtualLayout(shape=source_shapes, dtype=dtype)
        offset = 0
        for source in sources:
            length = source.shape[0]
            virtual_layout[offset: offset + length] = source
            offset += length

        return virtual_layout

    @property
    def file_names(self):
        return [self.format_sub_dataset_name(file_name.decode('ascii')) for file_name in self.h5py_file["file_names"]]
