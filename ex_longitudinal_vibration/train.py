import argparse
import os
import shutil
import sys
import time
import traceback
import numpy as np

import torch


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(".")
sys.path.append(PARENT_DIR)
from data import get_data
from network import DAE_NN

from utils import Logger, read_config_file, set_random_seed, save_config, to_pickle


def brain(config, logger, dataset_path, outputs_path):
    data = get_data(config, logger, dataset_path)
    y = data["y"].detach().requires_grad_(True).to(config.device).to(config.dtype)
    yt = data["yt"].detach().requires_grad_(True).to(config.device).to(config.dtype)

    model = DAE_NN(config, logger).to(config.device)
    logger.debug(model)

    optim = torch.optim.Adam(model.parameters(), config.learning_rate, weight_decay=1e-5)
    # logger.info(list(model.parameters()))

    # vanilla ae train loop
    stats = {"iter": [], "train_loss": [], "test_loss": []}
    for step in range(config.iterations):
        stats["iter"].append(step)

        train_loss = model.criterion(None, y, yt)
        optim.zero_grad()
        train_loss.backward()
        optim.step()
        stats["train_loss"].append(train_loss.item())

        if step > 100 and train_loss.item() < min(stats["train_loss"][:-1]):
            logger.debug("best step is: {}, the loss is: {:.4e}".format(step, train_loss.item()))
            model_path = os.path.join(outputs_path, "model.tar")
            torch.save(model.state_dict(), model_path)

        if step % config.print_every == 0:
            logger.info("step {}, train_loss {:.4e}, test_loss {:.4e}".format(step, train_loss.item(), train_loss.item()))

    # Save loss history (for txt and png)
    filename = f"loss.pkl"
    path = os.path.join(outputs_path, filename)
    to_pickle(stats, path)


if __name__ == "__main__":
    #  Parse command-line arguments
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument("--config_file", type=str, help="Path to the config.py file")
    args = parser.parse_args()

    # Read the configuration from the provided file
    config_file_path = args.config_file
    config = read_config_file(config_file_path)

    # Set up the output directory
    dataset_path = config.dataset_path
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)

    outputs_path = config.outputs_path
    if os.path.exists(outputs_path):
        shutil.rmtree(outputs_path)
    os.makedirs(outputs_path)

    # Initialize a logger for logging messages
    logger = Logger(outputs_path)
    logger.info("#" * 100)
    logger.info(f"Task: {config.taskname}")

    # Set random seed for reproducibility
    set_random_seed(config.seed)
    logger.info(f"Using device: {config.device}")
    logger.info(f"Using data type: {config.dtype}")

    try:
        start_time = time.time()
        brain(config, logger, dataset_path, outputs_path)
        end_time = time.time()
        execution_time = end_time - start_time
        logger.info(f"The running time of training: {execution_time} s")

    except Exception as e:
        logger.error(traceback.format_exc())
        print("An error occurred: {}".format(e))

    # Logs all the attributes and their values present in the given config object.
    save_config(config, logger)
