import argparse
import os
import sys
import traceback
import shutil


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(".")
sys.path.append(PARENT_DIR)

import learner as ln
from utils import Logger, read_config_file, set_random_seed


def train_model_with_config(config, logger, result_dir):
    """Train the model using provided data and model configurations."""
    # Get data using provided data configuration
    data = ln.data.get_data(config=config, logger=logger)

    # Create the neural network model using provided model configuration
    neural_net = ln.nn.get_model(config=config, logger=logger)

    # Initialize and run the training process
    training_args = {
        "data": data,
        "model": neural_net,
        "result_dir": result_dir,
        "config": config,
        "logger": logger,
    }
    training_manager = ln.TrainingManager(**training_args)
    training_manager.run_training()
    training_manager.restore()
    training_manager.output_results(info=training_args)

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument("--config_file", type=str, help="Path to the config.py file")
    args = parser.parse_args()

    # Read the configuration from the provided file
    config_file_path = args.config_file
    config = read_config_file(config_file_path)

    # Set up the output directory
    result_dir = os.path.join(config.outputs_dir, config.taskname)
    if os.path.exists(result_dir):
        shutil.rmtree(result_dir)
    os.makedirs(result_dir)

    # Initialize a logger for logging messages
    logger = Logger(result_dir)
    logger.info("#" * 100)
    logger.info(f"Task: {config.taskname}")

    # Set random seed for reproducibility
    set_random_seed(config.seed)
    logger.info(f"Using device: {config.device}")
    logger.info(f"Using data type: {config.dtype}")

    try:
        train_model_with_config(config, logger,  result_dir)
    except Exception as e:
        logger.error(traceback.format_exc())
        print("An error occurred: {}".format(e))


if __name__ == "__main__":
    main()