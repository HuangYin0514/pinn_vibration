# encoding: utf-8

from .dynamic_data import DynamicData
from .dynamic_dataset import DynamicDataset
from .dynamic_rnn_data import DynamicRnnData

__dataset_factory = {
    "DynamicData": DynamicData,
    "DynamicRnnData": DynamicRnnData,
}


def get_dataset(dataset_name, config, logger, *args, **kwargs):
    if dataset_name not in __dataset_factory.keys():
        raise ValueError("Dataset '{}' is not implemented".format(dataset_name))
    dataset = __dataset_factory[dataset_name](config, logger, *args, **kwargs)
    return dataset.data
