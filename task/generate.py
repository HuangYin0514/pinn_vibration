import argparse
import os
import shutil
import sys
import traceback

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(".")
sys.path.append(PARENT_DIR)

from gendata import generate_dataset
from utils import Logger, read_config_file, set_random_seed

def main():
    try:
        # Parse command line arguments
        parser = argparse.ArgumentParser(description="Generate dataset based on config.")
        parser.add_argument("--config_file", type=str, help="Path to the config.py file")
        args = parser.parse_args()
        config_file_path = args.config_file

        # Read configuration from file
        config = read_config_file(config_file_path)

        # Create output directories for logs and dataset
        logs_dir = os.path.join(config.outputs_dir, config.taskname)
        dataset_path = config.dataset_path

        if os.path.exists(logs_dir):
            shutil.rmtree(logs_dir)
        os.makedirs(logs_dir)

        if os.path.exists(dataset_path):
            shutil.rmtree(dataset_path)
        os.makedirs(dataset_path)

        # Initialize logger
        logger = Logger(logs_dir)
        logger.info("#" * 100)
        logger.info(config.taskname)

        # Set random seed and log device and dtype
        set_random_seed(config.seed)
        logger.info("Using the device: {}".format(config.device))
        logger.info("Using the dtype: {}".format(config.dtype))

        # Generate dataset
        dataset_args = {
            "config": config,
            "logger": logger,
            "dataset_path": dataset_path
        }
        generate_dataset(**dataset_args)

    except Exception as e:
        # Log and print any exceptions that occur
        logger.error(traceback.format_exc())
        print("An error occurred: {}".format(e))

if __name__ == "__main__":
    main()