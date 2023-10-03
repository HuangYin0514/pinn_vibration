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


def generate_train_dataset(config, logger):
    nu = 0.01 / np.pi
    noise = 0.0

    N_u = 100
    N_f = 10000

    # read data from file --------------------------------
    path = "/home/lbu/project/pinn_vibration/ex_burgers/dataset/burgers_shock.mat"
    data = scipy.io.loadmat(path)

    t = data["t"].flatten()[:, None]
    x = data["x"].flatten()[:, None]
    Exact = np.real(data["usol"]).T
    X, T = np.meshgrid(x, t)

    X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
    u_star = Exact.flatten()[:, None]

    # data --------------------------------
    # Doman bounds
    lb = X_star.min(0)
    ub = X_star.max(0)

    ic_x = np.hstack((X[0:1, :].T, T[0:1, :].T))
    ic_value = Exact[0:1, :].T
    bc_low_x = np.hstack((X[:, 0:1], T[:, 0:1]))
    bc_low_value = Exact[:, 0:1]
    bc_up_x = np.hstack((X[:, -1:], T[:, -1:]))
    bc_up_value = Exact[:, -1:]

    X_data = np.vstack([ic_x, bc_low_x, bc_up_x])
    U_data = np.vstack([ic_value, bc_low_value, bc_up_value])

    idx = np.random.choice(X_data.shape[0], N_u, replace=False)
    X_data = X_data[idx, :]
    U_data = U_data[idx, :]

    # inner points
    # TODO: add data points to train

    # physics --------------------------------
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
    path = "/home/lbu/project/pinn_vibration/ex_burgers/dataset/burgers_shock.mat"
    data = scipy.io.loadmat(path)

    t = data["t"].flatten()[:, None]
    x = data["x"].flatten()[:, None]
    Exact = np.real(data["usol"]).T
    X, T = np.meshgrid(x, t)

    X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
    u_star = Exact.flatten()[:, None]

    # to tensor ------------------------------
    X_star = torch.from_numpy(X_star).to(config.device, config.dtype).requires_grad_(True)
    u_star = torch.from_numpy(u_star).to(config.device, config.dtype)

    return X_star, u_star


@timing
def get_data(config, logger):
    train_data = None

    path = os.path.join(config.dataset_path, "burgers-dataset.pkl")

    # if os.path.exists(path):
    #     train_data = from_pickle(path)
    #     logger.info("Dataset loaded at: {}".format(path))
    #     return train_data

    train_data = generate_train_dataset(config, logger)
    eval_data = generate_eval_dataset(config, logger)
    # to_pickle(train_data, path)
    logger.info("Dataset generated at: {}".format(path))

    # Log the configuration and dataset generation completion
    logger.info("Congratulations, the dataset is generated!")

    return train_data, eval_data
