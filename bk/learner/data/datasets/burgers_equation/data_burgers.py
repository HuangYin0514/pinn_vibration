# encoding: utf-8

import os

import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch import nn
from pyDOE import lhs

import scipy.io
from scipy.interpolate import griddata


class DataBurgers:
    data_path = ""

    def __init__(self, config, logger):
        super(DataBurgers, self).__init__()
        self.config = config
        self.logger = logger

    def load_data(self):
        dataset_path = self.config.dataset_path
        data = scipy.io.loadmat(dataset_path)
        return data

    def get_data(self):
        train_data = self.get_train_data()
        val_data = self.get_val_data()
        return train_data, val_data

    def get_train_data(self):
        nu = 0.01 / np.pi
        noise = 0.0

        N_u = 100
        N_f = 10000

        data = self.load_data()
        t = data["t"].flatten()[:, None]
        x = data["x"].flatten()[:, None]
        Exact = np.real(data["usol"]).T
        X, T = np.meshgrid(x, t)

        X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
        u_star = Exact.flatten()[:, None]

        # Doman bounds
        lb = X_star.min(0)
        ub = X_star.max(0)

        xx1 = np.hstack((X[0:1, :].T, T[0:1, :].T))
        uu1 = Exact[0:1, :].T
        xx2 = np.hstack((X[:, 0:1], T[:, 0:1]))
        uu2 = Exact[:, 0:1]
        xx3 = np.hstack((X[:, -1:], T[:, -1:]))
        uu3 = Exact[:, -1:]

        X_u_train = np.vstack([xx1, xx2, xx3])
        X_f_train = lb + (ub - lb) * lhs(2, N_f)
        X_f_train = np.vstack((X_f_train, X_u_train))
        u_train = np.vstack([uu1, uu2, uu3])

        idx = np.random.choice(X_u_train.shape[0], N_u, replace=False)
        X_u_train = X_u_train[idx, :]
        u_train = u_train[idx, :]

        return X_u_train, u_train, X_f_train

    def get_val_data(self):
        nu = 0.01 / np.pi
        noise = 0.0

        N_u = 100
        N_f = 10000

        data = self.load_data()
        t = data["t"].flatten()[:, None]
        x = data["x"].flatten()[:, None]
        Exact = np.real(data["usol"]).T
        X, T = np.meshgrid(x, t)

        X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
        u_star = Exact.flatten()[:, None]

        # Doman bounds
        lb = X_star.min(0)
        ub = X_star.max(0)

        xx1 = np.hstack((X[0:1, :].T, T[0:1, :].T))
        uu1 = Exact[0:1, :].T
        xx2 = np.hstack((X[:, 0:1], T[:, 0:1]))
        uu2 = Exact[:, 0:1]
        xx3 = np.hstack((X[:, -1:], T[:, -1:]))
        uu3 = Exact[:, -1:]

        X_u_train = np.vstack([xx1, xx2, xx3])
        X_f_train = lb + (ub - lb) * lhs(2, N_f)
        X_f_train = np.vstack((X_f_train, X_u_train))
        u_train = np.vstack([uu1, uu2, uu3])

        idx = np.random.choice(X_u_train.shape[0], N_u, replace=False)
        X_u_train = X_u_train[idx, :]
        u_train = u_train[idx, :]

        return X_u_train, u_train, X_f_train
