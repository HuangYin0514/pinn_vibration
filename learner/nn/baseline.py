# encoding: utf-8
"""
@author: Yin Huang
@contact: hy1071324110@gmail.com
@time: 2023/7/7 3:35 PM
@desc:
"""
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from configs.config_plot import *
from utils import initialize_class, tensors_to_numpy

from .utils_nn import MLPBlock


class BackboneNet(nn.Module):
    """
    Custom backbone neural network module.

    This module contains multiple MLPBlocks with a specified number of layers.

    Args:
        input_dim (int): Dimension of the input features.
        hidden_dim (int): Dimension of the hidden layers.
        output_dim (int): Dimension of the output.
        layers_num (int): Number of hidden layers in the network.
    """

    def __init__(self, config):
        super(BackboneNet, self).__init__()

        input_dim = config.BackboneNet_input_dim
        hidden_dim = config.BackboneNet_hidden_dim
        output_dim = config.BackboneNet_output_dim
        layers_num = config.BackboneNet_layers_num

        # activation = SinActivation()
        activation = nn.ReLU()

        # Input layer
        input_layer = MLPBlock(input_dim, hidden_dim, activation)

        # Hidden layers
        hidden_layers = nn.ModuleList(
            [MLPBlock(hidden_dim, hidden_dim, activation) for _ in range(layers_num)]
        )

        # Output layer
        output_layer = nn.Linear(hidden_dim, output_dim, bias=False)

        layers = []
        layers.extend([input_layer])
        layers.extend(hidden_layers)
        layers.extend([output_layer])

        # Create the sequential model
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass of the BackboneNet.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, input_dim).

        Returns:
            torch.Tensor: The output tensor of shape (batch_size, output_dim).
        """
        out = self.net(x)
        return out


class Baseline(nn.Module):
    def __init__(self, config, logger, *args, **kwargs):
        super(Baseline, self).__init__()

        self.config = config
        self.logger = logger

        self.device = config.device
        self.dtype = config.dtype

        self.backboneNet = BackboneNet(config)

        try:
            class_name = self.config.dynamic_class
            kwargs = {"config": config, "logger": logger}
            self.right_term_net = initialize_class("dynamics", class_name, **kwargs)
        except ValueError as e:
            raise RuntimeError("class '{}' is not available".format(class_name))

    def forward(self, t):
        t = t.reshape(-1, 1)
        out = self.backboneNet(t)
        return out

    def get_q_qt_qtt(self, t):
        out = self(t)
        q_hat, qt_hat, qtt_hat = torch.tensor_split(
            out, (self.config.dof, self.config.dof * 2), dim=-1
        )
        return q_hat, qt_hat, qtt_hat

    def criterion(self, data, *args, **kwargs):
        y0, train_y, train_yt, train_t, physics_t = data
        q, qt = torch.chunk(train_y, 2, dim=-1)
        qt, qtt = torch.chunk(train_yt, 2, dim=-1)

        ################################
        # Loss for initial condition (y0)
        t0 = torch.zeros(1, 1, device=self.device, dtype=self.dtype).requires_grad_(
            True
        )
        q_hat, qt_hat, qtt_hat = self.get_q_qt_qtt(t0)
        y0_hat = torch.cat([q_hat, qt_hat], dim=-1)
        loss_y0 = F.mse_loss(y0_hat, y0)
        ################################

        ################################
        # Loss for training data
        q_hat, qt_hat, qtt_hat = self.get_q_qt_qtt(train_t)
        loss_data = 0
        loss_data += F.mse_loss(q_hat, q)
        loss_data += F.mse_loss(qt_hat, qt)
        loss_data += F.mse_loss(qtt_hat, qtt)
        ################################

        ################################
        # Loss for physics data
        # q_left_hat, qt_left_hat, qtt_left_hat = self.get_q_qt_qtt(physics_t)
        # yt_right_hat = self.right_term_net(physics_t, torch.cat([q_left_hat, qt_left_hat], dim=-1))
        # qt_right_hat, qtt_right_hat, lambdas = torch.tensor_split(yt_right_hat, (self.config.dof, self.config.dof * 2), dim=-1)
        # loss_res = torch.mean((qtt_left_hat - qtt_right_hat)**2)
        loss_res = 0
        ################################

        return (
            self.config.loss_y0_weight * loss_y0
            + self.config.loss_data_weight * loss_data
            + self.config.loss_physic_weight * loss_res
        )

    def evaluate(self, data, output_dir="", current_iterations=None, *args, **kwargs):
        ################################
        # Unpack data
        y0, y, yt, data_t, physics_t = data

        # Split tensors
        q, qt = torch.chunk(y, 2, dim=-1)
        qt, qtt = torch.chunk(yt, 2, dim=-1)

        ################################
        # Predict using the physics model
        q_hat, qt_hat, qtt_hat = self.get_q_qt_qtt(data_t)

        ################################
        # Calculate mean squared error
        mse_error = F.mse_loss(q_hat, q) + F.mse_loss(qt_hat, qt)

        ################################
        # Convert tensors to numpy arrays
        data_t = tensors_to_numpy(data_t)
        q, qt, qtt = tensors_to_numpy(q, qt, qtt)
        q_hat, qt_hat, qtt_hat = tensors_to_numpy(q_hat, qt_hat, qtt_hat)

        ################################
        # Calculate energy and other terms using the physics model
        energy = self.right_term_net.calculator.energy(q_hat, qt_hat)
        phi = self.right_term_net.calculator.phi(q_hat, qt_hat, qtt_hat)
        phi_t = self.right_term_net.calculator.phi_t(q_hat, qt_hat, qtt_hat)
        phi_tt = self.right_term_net.calculator.phi_tt(q_hat, qt_hat, qtt_hat)

        # Calculate errors
        gt_energy = self.right_term_net.calculator.energy(q[:2], qt[:2])[0]
        mean_energy_error = np.mean((energy - gt_energy) ** 2)
        energy_error = np.max(energy - gt_energy)
        phi_error = np.max(phi)
        phi_t_error = np.max(phi_t)
        phi_tt_error = np.max(phi_tt)

        # Generate error output strings
        iteration_output = f"iteration: {current_iterations}"
        MSE_y_output = f"MSE_y: {mse_error:.4e}"
        mean_energy_output = f"Mean energy error: {mean_energy_error:.4e}"
        energy_output = f"Energy error: {energy_error:.4e}"
        phi_output = f"Phi error: {phi_error:.4e}"
        phi_t_output = f"Phi_t error: {phi_t_error:.4e}"
        phi_tt_output = f"Phi_tt error: {phi_tt_error:.4e}"
        full_output = " | ".join(
            [
                iteration_output,
                MSE_y_output,
                mean_energy_output,
                energy_output,
                phi_output,
                phi_t_output,
                phi_tt_output,
            ]
        )
        self.logger.debug(full_output)

        ################################
        # Concatenate states
        all_states = np.concatenate([q, qt, qtt], axis=-1)
        all_states_hat = np.concatenate([q_hat, qt_hat, qtt_hat], axis=-1)

        # Plot and save error figures.
        fig_num = self.config.dof * 3
        line_num = fig_num // self.config.dof * 2
        row_num = self.config.dof

        fig, axs = plt.subplots(
            line_num, row_num, figsize=(4 * row_num, 3 * line_num), dpi=DPI
        )

        for check_dim in range(fig_num):
            index_1 = check_dim // row_num
            index_2 = check_dim % row_num

            subfig = axs[index_1, index_2]
            subfig.set_xlabel("$t$ ($-$)")
            subfig.set_ylabel("$q$ ($-$)")
            subfig.plot(data_t, all_states[:, check_dim], label="GT")
            subfig.plot(data_t, all_states_hat[:, check_dim], c="y", label="net")
            subfig.legend()

            index_1 = (check_dim + fig_num) // row_num
            index_2 = (check_dim + fig_num) % row_num
            subfig = axs[index_1, index_2]
            subfig.set_xlabel("$t$ ($-$)")
            subfig.set_ylabel("$err$ ($-$)")
            subfig.plot(
                data_t, np.abs(all_states - all_states_hat)[:, check_dim], label="error"
            )
            subfig.legend()

        plt.tight_layout()

        save_path = os.path.join(output_dir, f"iter_{current_iterations}.png")
        plt.savefig(save_path)
        plt.close()

        return mse_error
