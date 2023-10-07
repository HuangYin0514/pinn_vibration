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
from network import PINN

from utils import timing, Logger, read_config_file, set_random_seed, save_config, to_pickle, count_parameters


def train_fn(data, model, optim, scheduler):
    train_loss = model.criterion(data)
    optim.zero_grad()
    train_loss.backward()
    optim.step()
    scheduler.step()
    return train_loss


def train_LBFGS_fn(data, model, optim):
    train_loss_list = []

    def closure():
        train_loss = model.criterion(data)
        optim.zero_grad()
        train_loss.backward()
        train_loss_list.append(train_loss)
        return train_loss

    optim.step(closure)
    train_loss = train_loss_list[-1]
    return train_loss


def eval_fn(data, model):
    u_pred, error = model.eval(data)
    return error


@timing
def brain(config, logger):
    # dataset --------------------------------------------------------------------
    logger.info("#" * 50)
    logger.info("Starting generate dataset...")
    train_data, eval_data = get_data(config, logger)

    # setting the train environment ----------------------------------------------
    logger.info("#" * 50)
    logger.info("Starting set the train environment...")
    model = PINN(config, logger).to(config.device, config.dtype)
    logger.debug(model)
    logger.info("the parameters of model is {}".format(count_parameters(model)))
    optim = torch.optim.Adam(model.parameters(), config.learning_rate, weight_decay=1e-5)
    optimi_LBFGS = torch.optim.LBFGS(model.parameters(), lr=1.0)
    logger.info("the learning rate is {:.3e}".format(config.learning_rate))
    logger.info("the optimizer is {}".format(optim.__class__.__name__))
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=int(config.LBFGS_iterations // 3), gamma=0.99)
    logger.info("the scheduler is {}".format(scheduler.__class__.__name__))

    # training ------------------------------------------------------------------
    logger.info("#" * 50)
    logger.info("Starting train neural network ...")
    stats = {"iter": [], "train_loss": [], "val_loss": [], "learning_rate": []}
    for step in range(config.iterations + 1):
        stats["iter"].append(step)

        # train one step
        if step < config.LBFGS_iterations:
            train_loss = train_fn(data=train_data, model=model, optim=optim, scheduler=scheduler)
            stats["learning_rate"].append(optim.param_groups[0]["lr"])
        else:
            train_loss = train_LBFGS_fn(data=train_data, model=model, optim=optimi_LBFGS)
            stats["learning_rate"].append(optimi_LBFGS.param_groups[0]["lr"])
        stats["train_loss"].append(train_loss.item())

        # validation
        if step % config.print_every == 0:
            val_loss = eval_fn(data=eval_data, model=model)
            stats["val_loss"].append(val_loss.item())

        # check training loss
        if torch.any(torch.isnan(train_loss)):
            raise RuntimeError("Encountering nan, stop training")

        # save best model
        if step > 100 and train_loss.item() < min(stats["train_loss"][:-1]):
            logger.debug("best step is: {}, the loss is: {:.4e}".format(step, train_loss.item()))
            model_path = os.path.join(config.outputs_path, "model.tar")
            torch.save(model.state_dict(), model_path)

        # log current infomatiion
        if step % config.print_every == 0:
            logger.info("step {}, train_loss {:.4e}, val_loss {:.4e}".format(step, train_loss.item(), val_loss.item()))

    # Save train information ----------------------------------------------
    # Save loss history
    filename = f"loss.pkl"
    path = os.path.join(config.outputs_path, filename)
    to_pickle(stats, path)

    # Find the smallest loss
    check_loss_list = np.array(stats["train_loss"])
    best_index = check_loss_list.argmin()
    iteration = int(stats["iter"][best_index])
    train_loss = stats["train_loss"][best_index]
    val_loss = stats["val_loss"][best_index // config.print_every + 1]
    contents = (
        "\n"
        + "Train completion time: "
        + time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime(time.time()))
        + "\n"
        + "Task name: {}".format(config.taskname)
        + "\n"
        + "Model name: {}".format(model.__class__.__name__)
        + "\n"
        + "Best model at iteration: {}".format(iteration)
        + "\n"
        + "Train loss: {:.3e}".format(train_loss)
        + "\n"
        + "Val loss: {:.3e}".format(val_loss)
    )
    logger.info(contents)


if __name__ == "__main__":
    ######################################################################
    #
    # config environment
    #
    ######################################################################
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
    logger.info("#" * 50)
    logger.info(f"Task: {config.taskname}")

    # Set random seed for reproducibility
    set_random_seed(config.seed)
    logger.info(f"Using device: {config.device}")
    logger.info(f"Using data type: {config.dtype}")

    ######################################################################
    #
    # traing
    #
    ######################################################################
    try:
        start_time = time.time()
        brain(config, logger)
        end_time = time.time()
        execution_time = end_time - start_time
        logger.info("The running time of training: {:.5e} s".format(execution_time))

    except Exception as e:
        logger.error(traceback.format_exc())
        print("An error occurred: {}".format(e))

    # Logs all the attributes and their values present in the given config object.
    save_config(config, logger)
