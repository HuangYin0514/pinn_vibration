import numpy as np
import torch
from sympy import Function, Matrix, lambdify, symbols
from torch import nn

from utils import tensors_to_numpy


class Phi_Net(nn.Module):

    def __init__(self, config, logger):
        super(Phi_Net, self).__init__()

        self.l = config.l

        self.device = config.device
        self.dtype = config.dtype

    def get_phi(self, q):
        phi = torch.stack(
            [q[:,0]**2 + q[:,1]**2 - self.l[0]**2, (q[:,0] - q[:,2])**2 + (q[:,1] - q[:,3])**2 - self.l[1]**2],
            dim=-1,
        )
        return phi

    def get_phi_q(self, q):
        e0 = torch.zeros_like(q[:, 0])
        phi_q = torch.stack(
            [
                torch.stack([2 * q[:, 0], 2 * q[:, 1], e0, e0], dim=-1),
                torch.stack(
                    [
                        2 * q[:, 0] - 2 * q[:, 2],
                        2 * q[:, 1] - 2 * q[:, 3],
                        -2 * q[:, 0] + 2 * q[:, 2],
                        -2 * q[:, 1] + 2 * q[:, 3],
                    ],
                    dim=-1,
                ),
            ],
            dim=-2,
        )
        return phi_q

    def get_phi_q_qt_q_qt(self, q, qt):
        phi_q_qt_q_qt = torch.stack(
            [
                2 * qt[:, 0]**2 + 2 * qt[:, 1]**2, 2 * qt[:, 0] * (qt[:, 0] - qt[:, 2]) + 2 * qt[:, 1] * (qt[:, 1] - qt[:, 3]) - 2 * qt[:, 2] *
                (qt[:, 0] - qt[:, 2]) - 2 * qt[:, 3] * (qt[:, 1] - qt[:, 3])
            ],
            dim=-1,
        )
        return phi_q_qt_q_qt


class F_Net(nn.Module):

    def __init__(self, config, logger):
        super(F_Net, self).__init__()
        self.m = config.m
        self.g = config.g

        self.device = config.device
        self.dtype = config.dtype

    def forward(self, t, coords):
        bs, num_states = coords.shape
        t = t.reshape(-1, 1)
        F = torch.tensor(
            [0, -self.m[0] * self.g, 0, -self.m[1] * self.g],
            device=self.device,
            dtype=self.dtype,
        )
        F = F.reshape(1, -1).repeat(bs, 1)
        return F


class M_Net(nn.Module):

    def __init__(self, config, logger):
        super(M_Net, self).__init__()
        self.m = config.m
        self.l = config.l

        self.device = config.device
        self.dtype = config.dtype

    def forward(self, q):
        bs, states = q.shape
        m_values = torch.tensor(
            [self.m[0], self.m[0], self.m[1], self.m[1]],
            dtype=self.dtype,
            device=self.device,
        )
        diag_tensor = torch.diag(m_values)
        M = diag_tensor.repeat(bs, 1, 1)
        return M


# Class defining the dynamic model for a single pendulum using Differential Algebraic Equations (DAE)
class DynamicDoublePendulumDAE2ODE(nn.Module):

    def __init__(self, config, logger):
        super(DynamicDoublePendulumDAE2ODE, self).__init__()
        self.device = config.device
        self.dtype = config.dtype
        self.config = config
        self.phi_net = Phi_Net(config, logger)
        self.m_net = M_Net(config, logger)
        self.f_net = F_Net(config, logger)
        self.calculator = DynamicSinglePendulumCalculator(config=config, logger=logger)

    def forward(self, t, coords):
        """Compute the forward dynamics of a single pendulum.

        Args:
            t (torch.Tensor): Time tensor.
            coords (torch.Tensor): Coordinates tensor.

        Returns:
            torch.Tensor: Computed derivatives of coordinates.
        """
        q, qt = torch.chunk(coords, 2, dim=-1)

        # Compute phi_q
        phi_q = self.phi_net.get_phi_q(q)

        # Compute M and its inverse Minv
        M = self.m_net(q)
        Minv = torch.linalg.inv(M)

        # Compute F
        F = self.f_net(t, torch.cat([q, qt], dim=-1))

        # Compute phi_q_qt_q_qt
        phi_q_qt_q_qt = -self.phi_net.get_phi_q_qt_q_qt(q, qt)

        # Solve for lam
        phi_q_Minv = torch.matmul(phi_q, Minv)
        L = torch.matmul(phi_q_Minv, phi_q.permute(0, 2, 1))
        R = torch.matmul(phi_q_Minv, F.unsqueeze(-1)) - phi_q_qt_q_qt.unsqueeze(-1)
        lam = torch.linalg.solve(L, R)  # Using torch.linalg.solve for better stability

        # Solve for qtt
        qtt_R = F.unsqueeze(-1) - torch.matmul(phi_q.permute(0, 2, 1), lam)
        qtt = torch.matmul(Minv, qtt_R).squeeze(-1)

        # Combine qt and qtt
        qtqtt = torch.cat([qt, qtt], dim=-1)

        return qtqtt


# Class for calculating kinematic quantities for the pendulum
class KinematicsCalculator:

    def __init__(self, config):
        l = config.l
        dof = config.dof
        self.t = symbols('t')
        # Define symbols for q, qt, and qtt
        q = [Function(f'q{i}')(self.t) for i in range(dof)]
        qt = [q_var.diff(self.t) for q_var in q]
        qtt = [q_var.diff(self.t, self.t) for q_var in q]
        # Define phi expressions
        phi_expr = Matrix([q[0]**2 + q[1]**2 - l[0]**2, (q[0] - q[2])**2 + (q[1] - q[3])**2 - l[1]**2])

        self.expr_phi = phi_expr
        self.expr_phi_fn = lambdify([q, qt, qtt], self.expr_phi, 'numpy')
        self.expr_phi_t = phi_expr.diff(self.t)
        self.expr_phi_t_fn = lambdify([q, qt, qtt], self.expr_phi_t, 'numpy')
        self.expr_phi_tt = phi_expr.diff(self.t, self.t)
        self.expr_phi_tt_fn = lambdify([q, qt, qtt], self.expr_phi_tt, 'numpy')

    # Calculate the value of the phi function
    def get_phi(self, q_values, qt_values, qtt_values):
        phi_val = self.expr_phi_fn(q_values, qt_values, qtt_values)
        return phi_val

    # Calculate the value of the time derivative of phi function
    def get_phi_t(self, q_values, qt_values, qtt_values):
        phi_t_val = self.expr_phi_t_fn(q_values, qt_values, qtt_values)
        return phi_t_val

    # Calculate the value of the second time derivative of phi function
    def get_phi_tt(self, q_values, qt_values, qtt_values):
        phi_tt_val = self.expr_phi_tt_fn(q_values, qt_values, qtt_values)
        return phi_tt_val


# Class for calculating dynamics-related quantities for the pendulum
class DynamicSinglePendulumCalculator:

    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.m_net = M_Net(config, logger)
        self.calculator = KinematicsCalculator(config)

    # Calculate the kinetic energy of the system
    def kinetic(self, q, qt):
        T = 0.
        M = tensors_to_numpy(self.m_net(q))
        T = 0.5 * np.einsum('bi,bii,bi->b', qt, M, qt)
        return T

    # Calculate the potential energy of the system
    def potential(self, q, qd):
        m = self.config.m
        g = self.config.g
        obj = self.config.obj
        U = 0.
        for i in range(obj):
            y = q[:, i * 2 + 1]
            U += m[i] * g * y
        return U

    # Calculate the total energy of the system
    def energy(self, q, qt):
        energy = self.kinetic(q, qt) + self.potential(q, qt)
        return energy

    # Calculate the value of the phi function
    def phi(self, q, qt, qtt):
        res = self.calculator.get_phi(q.T, qt.T, qtt.T)
        return np.array(res, dtype=np.double).squeeze().T

    # Calculate the value of the time derivative of phi function
    def phi_t(self, q, qt, qtt):
        res = self.calculator.get_phi_t(q.T, qt.T, qtt.T)
        return np.array(res, dtype=np.double).squeeze().T

    # Calculate the value of the second time derivative of phi function
    def phi_tt(self, q, qt, qtt):
        res = self.calculator.get_phi_tt(q.T, qt.T, qtt.T)
        return np.array(res, dtype=np.double).squeeze().T