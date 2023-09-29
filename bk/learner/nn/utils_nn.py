# encoding: utf-8

import torch
from torch import nn, Tensor


def weights_init_xavier_normal(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:  # find the linear layer class
        nn.init.xavier_normal_(m.weight)
        # if m.bias is not None:
        #     nn.init.constant_(m.bias, 0.0)


def weights_init_orthogonal_normal(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        nn.init.orthogonal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)


class IdentityActivation(nn.Module):

    def __init__(self):
        super(IdentityActivation, self).__init__()

    def forward(self, input: Tensor) -> Tensor:
        return input


class SinActivation(nn.Module):

    def forward(self, input: Tensor) -> Tensor:
        return torch.sin(input)


class MLPBlock(nn.Module):
    """
    MLPBlock represents a single block of the MLP (Multi-Layer Perceptron) model.

    It consists of a linear layer followed by an activation function.

    Args:
        input_dim (int): The number of input features to the block.
        output_dim (int): The number of output features from the block.
        activation (torch.nn.Module): The activation function to be applied after the linear layer.
    """

    def __init__(self, input_dim, output_dim, activation):
        super(MLPBlock, self).__init__()
        self.linear_layer = nn.Linear(input_dim, output_dim)
        self.activation = activation

    def forward(self, x):
        """
        Forward pass of the MLPBlock.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, input_dim).

        Returns:
            torch.Tensor: The output tensor of shape (batch_size, output_dim).
        """
        return self.activation(self.linear_layer(x))