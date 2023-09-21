import torch

from .collate_batch import train_collate_fn, val_collate_fn
from .datasets import DynamicDataset, get_dataset
from .transforms import build_transforms


def get_data(config, logger, *args, **kwargs):
    logger.info("=================>")
    logger.info("Start get dataset...")
    logger.info("Loading dataset from path: {}".format(config.dataset_path))

    train_data, val_data = get_dataset(config.data_name, config, logger, *args, **kwargs)

    # Data transformations
    train_transforms = build_transforms(is_train=True, config=config, **kwargs)
    val_transforms = build_transforms(is_train=False, config=config, **kwargs)

    # Dataset creation
    train_dataset = DynamicDataset(train_data, train_transforms)
    val_dataset = DynamicDataset(val_data, val_transforms)

    # DataLoaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=len(train_dataset),
        shuffle=True,
        collate_fn=train_collate_fn,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=len(val_dataset),
        shuffle=False,
        collate_fn=val_collate_fn,
    )

    return train_loader, val_loader
