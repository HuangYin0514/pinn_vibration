import argparse
import os
import shutil
import sys
import time
import traceback

import torch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(".")
sys.path.append(PARENT_DIR)

from dynamic_single_pendulum_DAE import DynamicSinglePendulumDAE

from integrator import ODEIntegrate
from utils import from_pickle, to_pickle


def generate_dataset(dynamics, config, logger):
    logger.info("Start generating dataset...")

    start_time = time.time()
    y0 = torch.tensor(config.y0, device=config.device, dtype=config.dtype)
    y0 = y0.repeat(config.data_num, 1)

    # t (time_len)
    # sol (num_trajectories, time_len, states)
    t, sol = ODEIntegrate(
        func=dynamics,
        t0=config.t0,
        t1=config.t1,
        dt=config.dt,
        y0=y0,
        method=config.ode_solver,
        device=config.device,
        dtype=config.dtype,
        dof=config.dof,
    )

    end_time = time.time()
    execution_time = end_time - start_time
    logger.info(f"The running time of ODE solver: {execution_time} s")

    return t, sol


def get_data(config, logger, path):
    path = os.path.join(path, "single-pendulum-dataset.pkl")

    if os.path.exists(path):
        data = from_pickle(path)
        logger.info("Dataset loaded at: {}".format(path))
        return data

    
    # --------------------------------------------------
    # data = {
    #     "y0": y0,
    #     "t": t,
    #     "y": y,
    #     "yt": yt,
    # }
    # data["meta"] = None

    # save the dataset
    to_pickle(data, path)
    logger.info("Dataset generated at: {}".format(path))

    # Log the configuration and dataset generation completion
    logger.info("Congratulations, the dataset is generated!")

    return data
