# encoding: utf-8

from .burgers_equation import DataBurgers


__dataset_factory = {
    "DataBurgers": DataBurgers,
}


def get_dataset(dataset_name, config, logger, *args, **kwargs):
    if dataset_name not in __dataset_factory.keys():
        raise ValueError("Dataset '{}' is not implemented".format(dataset_name))
    dataset_class = __dataset_factory[dataset_name](config, logger, *args, **kwargs)
    dataset = dataset_class.get_data()
    return dataset
