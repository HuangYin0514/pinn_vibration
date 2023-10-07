import argparse
import os
import shutil
import sys
import time
import traceback

import numpy as np
import scipy.io
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from pyDOE import lhs
from scipy.interpolate import griddata
from torch import nn

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(".")
sys.path.append(PARENT_DIR)

from utils import from_pickle, timing, to_pickle


################################################################
#
# dataset information
# (t, x, usol)
# t->(100,) x->(256,) usol->(256, 100)
#
################################################################
def generate_train_dataset(config, logger):
    N_u = 100
    N_f = 10000

    # read data from file --------------------------------
    path = "/home/lbu/project/pinn_vibration/ex_1D_lateral_vibration/dataset/data_1D_lateral_vibration.pkl"
    data = from_pickle(path)

    t = data["t"].flatten()[:, None]
    x = data["x"].flatten()[:, None]
    Exact = np.real(data["usol"])

    T, X = np.meshgrid(t, x)  # (256, 100)
    X_star = np.hstack((T.flatten()[:, None], X.flatten()[:, None]))
    u_star = Exact.flatten()[:, None]

    # data --------------------------------
    # bc and ic
    ic_x = np.hstack((T[:, 0:1], X[:, 0:1]))
    ic_value = Exact[:, 0:1]
    bc_low_x = np.hstack((T[0:1, :].T, X[0:1, :].T))
    bc_low_value = Exact[0:1, :].T
    bc_up_x = np.hstack((T[-1:, :].T, X[-1:, :].T))
    bc_up_value = Exact[-1:, :].T

    X_data = np.vstack([ic_x, bc_low_x, bc_up_x])
    U_data = np.vstack([ic_value, bc_low_value, bc_up_value])

    # inner point
    inner_X_data = X_star[::5, :]
    inner_U_data = u_star[::5, :]
    X_data = np.vstack((inner_X_data, X_data))
    U_data = np.vstack((inner_U_data, U_data))

    # physics --------------------------------
    lb = X_star.min(0)
    ub = X_star.max(0)
    X_physics = lb + (ub - lb) * lhs(2, N_f)
    X_physics = np.vstack((X_physics, X_data))
    U_physics = np.zeros((X_physics.shape[0], 1))

    # to tensor ------------------------------
    X_data = torch.from_numpy(X_data).to(config.device, config.dtype).requires_grad_(True)
    U_data = torch.from_numpy(U_data).to(config.device, config.dtype)
    X_physics = torch.from_numpy(X_physics).to(config.device, config.dtype).requires_grad_(True)
    U_physics = torch.from_numpy(U_physics).to(config.device, config.dtype)

    return X_data, U_data, X_physics, U_physics


def generate_eval_dataset(config, logger):
    # read data from file --------------------------------
    path = "/home/lbu/project/pinn_vibration/ex_1D_lateral_vibration/dataset/data_1D_lateral_vibration.pkl"
    data = from_pickle(path)

    t = data["t"].flatten()[:, None]
    x = data["x"].flatten()[:, None]
    Exact = np.real(data["usol"])
    T, X = np.meshgrid(t, x)  

    X_star = np.hstack((T.flatten()[:, None], X.flatten()[:, None]))
    u_star = Exact.flatten()[:, None]

    # to tensor ------------------------------
    X_star = torch.from_numpy(X_star).to(config.device, config.dtype).requires_grad_(True)
    u_star = torch.from_numpy(u_star).to(config.device, config.dtype)

    return X_star, u_star


@timing
def get_data(config, logger):
    train_data = None

    path = config.dataset_path

    train_data = generate_train_dataset(config, logger)
    eval_data = generate_eval_dataset(config, logger)

    logger.info("Dataset generated at: {}".format(path))

    # Log the configuration and dataset generation completion
    logger.info("Congratulations, the dataset is generated!")

    return train_data, eval_data
