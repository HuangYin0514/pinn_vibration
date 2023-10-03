import torch
from torch import nn

from dynamic_single_pendulum_DAE import DynamicSinglePendulumDAE


#######################################################################
#
# force F function
#
#######################################################################
class F_Net(nn.Module):
    def __init__(self, config, logger):
        super(F_Net, self).__init__()
        self.device = config.device
        self.dtype = config.dtype

        self.F1 = torch.nn.Parameter(
            0.1
            * torch.randn(
                1,
            )
        )

    # Compute the force F(t, coords)
    def forward(self, t, coords):
        bs, num_states = coords.shape
        e2 = torch.zeros(bs, 2, device=self.device, dtype=self.dtype)
        e2[:, 1] += -self.F1
        F = e2
        return F


#######################################################################
#
# mass M function
#
#######################################################################
class M_Net(nn.Module):
    def __init__(self, config, logger):
        super(M_Net, self).__init__()
        self.device = config.device
        self.dtype = config.dtype

        self.mass1 = torch.nn.Parameter(
            0.1
            * torch.randn(
                1,
            )
        )

    # Compute the mass matrix M(q)
    def forward(self, t, coords):
        bs, num_states = coords.shape
        m1 = self.mass1
        e22 = torch.zeros(bs, 2, 2, device=self.device, dtype=self.dtype)
        e22[:, 0, 0] += self.mass1
        e22[:, 1, 1] += self.mass1
        M = e22
        return M


#######################################################################
#
# network function
#
#######################################################################
class DAE_NN(nn.Module):
    def __init__(self, config, logger):
        super(DAE_NN, self).__init__()
        self.config = config
        self.logger = logger

        m_net = M_Net(config, logger)
        f_net = F_Net(config, logger)

        self.dynamics = DynamicSinglePendulumDAE(config, logger)
        self.dynamics.f_net = f_net
        self.dynamics.m_net = m_net

    def forward(self, t, coords):
        d_coords = self.dynamics(t, coords)

        q, qt, lambdas = torch.tensor_split(d_coords, (self.config.dof, self.config.dof * 2), dim=-1)
        d_coords = torch.cat([q, qt], dim=-1)

        return d_coords

    def criterion(self, t, coords, labels):
        pred = self(t, coords)
        loss_res = torch.mean((pred - labels) ** 2)
        return loss_res
