import torch

from .datasets import get_dataset


def get_data(config, logger, *args, **kwargs):
    logger.info("=================>")
    logger.info("Start get dataset...")
    logger.info("Loading dataset from path: {}".format(config.dataset_path))

    train_data, val_data = get_dataset(config.data_name, config, logger, *args, **kwargs)

    return train_data, val_data
