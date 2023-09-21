# encoding: utf-8

import os

from .generator import DataGenerator


def generate_dataset(config, logger, dataset_path, *args, **kwargs):
    """
    Generates a dataset using the provided configuration and logger.

    Args:
        config (Config): Configuration object containing dataset generation settings.
        logger (Logger): Logger object for logging messages.
        *args: Variable-length positional arguments.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        None
    """
    #############################################
    logger.info("Start generating dataset...")

    # Create a GenerateData instance with the given configuration and logger
    dataset = DataGenerator(config=config, logger=logger)

    # Generate dataset with specified name based on data_num and t1 values in the configuration
    dataset_name = os.path.join(
        dataset_path, "dataset_num{}_t{}.npy".format(config.data_num, int(config.t1))
    )
    dataset.generate_data(dataset_name=dataset_name, *args, **kwargs)

    #############################################
    # Log the configuration and dataset generation completion
    logger.info("Configuration: {}".format(config))
    logger.info("Dataset generated at: {}".format(dataset_name))
    logger.info("Congratulations, the dataset is generated!")

    # Logs all the attributes and their values present in the given config object.
    keys_values_pairs = []  # List to store attribute-name and attribute-value pairs
    for attr_name in dir(config):
        if not attr_name.startswith("__"):  # Exclude private attributes
            attr_value = getattr(config, attr_name)  # Get the attribute value
            keys_values_pairs.append("{}: {}".format(attr_name, attr_value))
    full_output = "\n".join(keys_values_pairs)
    logger.info("Config values:\n%s", full_output)
