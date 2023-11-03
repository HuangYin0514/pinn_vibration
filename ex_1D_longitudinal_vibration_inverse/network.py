import numpy as np
import torch
from torch import nn

from utils_nn import MLPBlock, SinActivation


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

    def __init__(self, config, logger):
        super(BackboneNet, self).__init__()

        input_dim = config.BackboneNet_input_dim
        hidden_dim = config.BackboneNet_hidden_dim
        output_dim = config.BackboneNet_output_dim
        layers_num = config.BackboneNet_layers_num

        activation = nn.Tanh()

        # Input layer
        input_layer = MLPBlock(input_dim, hidden_dim, activation)

        # Hidden layers
        hidden_layers = nn.ModuleList([MLPBlock(hidden_dim, hidden_dim, activation) for _ in range(layers_num)])

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


#######################################################################
#
# network function
#
#######################################################################
class PINN(nn.Module):
    def __init__(self, config, logger):
        super(PINN, self).__init__()
        self.config = config
        self.logger = logger

        self.backboneNet = BackboneNet(config, logger)

        self.alpha = torch.nn.Parameter(
            0.01
            * torch.randn(
                1,
            )
        )

    def forward(self, t, x):
        out = self.backboneNet(torch.cat([t, x], dim=-1))
        return out

    def net_f(self, t, x, alpha=1.0):
        u = self(t, x)

        u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
        u_tt = torch.autograd.grad(u_t, t, grad_outputs=torch.ones_like(u_t), retain_graph=True, create_graph=True)[0]
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), retain_graph=True, create_graph=True)[0]

        f = u_tt - self.alpha**2 * u_xx
        return f

    def criterion(self, data):
        X_data, U_data, X_physics, U_physics = data
        t = X_data[:, 0:1]
        x = X_data[:, 1:2]
        U_data_hat = self(t, x)
        loss_U_data = torch.mean((U_data_hat - U_data) ** 2)

        t = X_physics[:, 0:1]
        x = X_physics[:, 1:2]
        f_data_hat = self.net_f(t, x)
        loss_f_data = torch.mean((f_data_hat - U_physics) ** 2)

        # self.logger.info("loss1: {:.3e}, loss2: {:.3e}".format(loss_U_data.item(),loss_f_data.item()))
        return 30*loss_U_data + loss_f_data

    def eval(self, data):
        X_data, U_data = data
        t = X_data[:, 0:1]
        x = X_data[:, 1:2]
        U_data_hat = self(t, x)

        error = torch.mean((U_data_hat - U_data) ** 2)
        self.logger.info("alpha is {}".format(self.alpha.item()))

        return U_data_hat, error
