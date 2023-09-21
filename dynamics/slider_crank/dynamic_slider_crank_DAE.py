import numpy as np
import torch
from sympy import Function, Matrix, cos, lambdify, sin, symbols
from torch import nn

from utils import tensors_to_numpy


#######################################################################
#
# constrain Phi function
#
#######################################################################
class Phi_Net(nn.Module):
    def __init__(self, config, logger):
        super(Phi_Net, self).__init__()
        self.l = config.l
        self.device = config.device
        self.dtype = config.dtype

    # Compute the Phi(q) function
    def get_phi_q(self, q):
        e0 = torch.zeros_like(q[:, 0])
        e1 = torch.ones_like(q[:, 0])
        # shape(bs, constrain len, m)
        phi_q = torch.stack(
            [
                torch.stack(
                    [e1, e0, self.l[0] * torch.sin(q[:, 2]) / 2, e0, e0, e0, e0], dim=-1
                ),
                torch.stack(
                    [e0, e1, -self.l[0] * torch.cos(q[:, 2]) / 2, e0, e0, e0, e0],
                    dim=-1,
                ),
                torch.stack(
                    [
                        e0,
                        e0,
                        self.l[0] * torch.sin(q[:, 2]),
                        e1,
                        e0,
                        self.l[1] * torch.sin(q[:, 5]) / 2,
                        e0,
                    ],
                    dim=-1,
                ),
                torch.stack(
                    [
                        e0,
                        e0,
                        -self.l[0] * torch.cos(q[:, 2]),
                        e0,
                        e1,
                        -self.l[1] * torch.cos(q[:, 5]) / 2,
                        e0,
                    ],
                    dim=-1,
                ),
                torch.stack(
                    [
                        e0,
                        e0,
                        self.l[0] * torch.sin(q[:, 2]),
                        e0,
                        e0,
                        self.l[1] * torch.sin(q[:, 5]),
                        e1,
                    ],
                    dim=-1,
                ),
                torch.stack(
                    [
                        e0,
                        e0,
                        self.l[0] * torch.cos(q[:, 2]),
                        e0,
                        e0,
                        self.l[1] * torch.cos(q[:, 5]),
                        e0,
                    ],
                    dim=-1,
                ),
            ],
            dim=-2,
        )

        return phi_q

    # Compute the Phi(q, qt, qtt) function
    # shape(bs, constrain len)
    def get_phi_q_qt_q_qt(self, q, qt):
        phi_q_qt_q_qt = torch.stack(
            [
                qt[:, 2] ** 2 * self.l[0] * torch.cos(q[:, 2]) / 2,
                qt[:, 2] ** 2 * self.l[0] * torch.sin(q[:, 2]) / 2,
                qt[:, 2] ** 2 * self.l[0] * torch.cos(q[:, 2])
                + qt[:, 5] ** 2 * self.l[1] * torch.cos(q[:, 5]) / 2,
                qt[:, 2] ** 2 * self.l[0] * torch.sin(q[:, 2])
                + qt[:, 5] ** 2 * self.l[1] * torch.sin(q[:, 5]) / 2,
                qt[:, 2] ** 2 * self.l[0] * torch.cos(q[:, 2])
                + qt[:, 5] ** 2 * self.l[1] * torch.cos(q[:, 5]),
                -qt[:, 2] ** 2 * self.l[0] * torch.sin(q[:, 2])
                - qt[:, 5] ** 2 * self.l[1] * torch.sin(q[:, 5]),
            ],
            dim=-1,
        )
        return phi_q_qt_q_qt


#######################################################################
#
# force F function
#
#######################################################################
class F_Net(nn.Module):
    def __init__(self, config, logger):
        super(F_Net, self).__init__()
        self.m = config.m
        self.g = config.g
        self.device = config.device
        self.dtype = config.dtype

    # Compute the force F(t, coords)
    def forward(self, t, coords):
        bs, num_states = coords.shape
        t = t.reshape(-1, 1)
        F = torch.tensor(
            [0, -self.m[0] * self.g, 0, 0, -self.m[1] * self.g, 0, 0],
            device=self.device,
            dtype=self.dtype,
        )
        F = F.reshape(1, -1).repeat(bs, 1)
        return F


#######################################################################
#
# mass M function
#
#######################################################################
class M_Net(nn.Module):
    def __init__(self, config, logger):
        super(M_Net, self).__init__()
        self.m = config.m
        self.l = config.l
        self.device = config.device
        self.dtype = config.dtype

    # Compute the mass matrix M(q)
    def forward(self, q):
        bs, states = q.shape
        m_values = torch.tensor(
            [
                self.m[0],
                self.m[0],
                1 / 12 * self.m[0] * self.l[0] ** 2,
                self.m[1],
                self.m[1],
                1 / 12 * self.m[1] * self.l[1] ** 2,
                self.m[2],
            ],
            device=self.device,
            dtype=self.dtype,
        )
        diag_tensor = torch.diag(m_values)
        M = diag_tensor.repeat(bs, 1, 1)
        return M


#######################################################################
#
# Differential Algebraic Equations (DAE)
#
#######################################################################
class DynamicSliderCrankDAE(nn.Module):
    def __init__(self, config, logger):
        super(DynamicSliderCrankDAE, self).__init__()
        self.device = config.device
        self.dtype = config.dtype
        self.config = config
        self.phi_net = Phi_Net(config, logger)
        self.m_net = M_Net(config, logger)
        self.f_net = F_Net(config, logger)
        self.calculator = DynamicSliderCrankCalculator(config=config, logger=logger)

    # Compute the dynamics of the system
    def forward(self, t, coords):
        q, qt, lambdas = torch.tensor_split(
            coords, (self.config.dof, self.config.dof * 2), dim=-1
        )

        # Compute phi_q
        phi_q = self.phi_net.get_phi_q(q)

        # Compute M and its inverse Minv
        M = self.m_net(q)

        # Compute F
        F = self.f_net(t, torch.cat([q, qt], dim=-1))

        # Compute phi_q_qt_q_qt
        phi_q_qt_q_qt = -self.phi_net.get_phi_q_qt_q_qt(q, qt)

        # Construct L
        bs = coords.shape[0]
        e0 = torch.zeros(
            bs, phi_q.shape[1], phi_q.shape[1], dtype=self.dtype, device=self.device
        )
        L1 = torch.cat([M, phi_q.permute(0, 2, 1)], dim=-1)
        L2 = torch.cat([phi_q, e0], dim=-1)
        L = torch.cat([L1, L2], dim=1)

        # Construct R
        R1 = F
        R2 = phi_q_qt_q_qt
        R = torch.cat([R1, R2], dim=-1)

        # Compute qtt qt lam
        solution = torch.linalg.solve(L, R)

        qtt = solution[:, : self.config.dof]
        lambdas = solution[:, self.config.dof :]

        augmented_solution = torch.cat([qt, qtt, lambdas], dim=-1)
        return augmented_solution


#######################################################################
#
# calculating kinematic quantities by using sympy
# phi, phi_t, phi_tt
#
#######################################################################
class KinematicsCalculator:
    def __init__(self, config):
        l = config.l
        dof = config.dof
        self.t = symbols("t")
        # Define symbols for q, qt, and qtt
        q = [Function(f"q{i}")(self.t) for i in range(dof)]
        qt = [q_var.diff(self.t) for q_var in q]
        qtt = [q_var.diff(self.t, self.t) for q_var in q]
        # Define phi expressions
        # phi_expr = Matrix([
        #     q[0] - l[0] / 2 * cos(q[2]), q[1] - l[0] / 2 * sin(q[2]), q[3] - l[0] / 2 * cos(q[2]) - q[0] - l[1] / 2 * cos(q[5]),
        #     q[4] - l[0] / 2 * sin(q[2]) - q[1] - l[1] / 2 * sin(q[5]), q[6] - l[0] * cos(q[2]) - l[1] * cos(q[5])
        # ])
        phi_expr = Matrix(
            [
                q[0] - l[0] / 2 * cos(q[2]),
                q[1] - l[0] / 2 * sin(q[2]),
                q[3] - l[0] * cos(q[2]) - l[1] / 2 * cos(q[5]),
                q[4] - l[0] * sin(q[2]) - l[1] / 2 * sin(q[5]),
                q[6] - l[0] * cos(q[2]) - l[1] * cos(q[5]),
                l[0] * sin(q[2]) + l[1] * sin(q[5]),
            ]
        )
        self.expr_phi = phi_expr
        self.expr_phi_fn = lambdify([q, qt, qtt], self.expr_phi, "numpy")
        self.expr_phi_t = phi_expr.diff(self.t)
        self.expr_phi_t_fn = lambdify([q, qt, qtt], self.expr_phi_t, "numpy")
        self.expr_phi_tt = phi_expr.diff(self.t, self.t)
        self.expr_phi_tt_fn = lambdify([q, qt, qtt], self.expr_phi_tt, "numpy")

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


#######################################################################
#
# calculating dynamics-related quantities
# energy, kinetic, potential, phi, phi_t, phi_tt
#
#######################################################################
class DynamicSliderCrankCalculator:
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.m_net = M_Net(config, logger)
        self.calculator = KinematicsCalculator(config)

    # Calculate the kinetic energy of the system
    def kinetic(self, q, qt):
        T = 0.0
        M = tensors_to_numpy(self.m_net(q))
        T = 0.5 * np.einsum("bi,bii,bi->b", qt, M, qt)
        return T

    # Calculate the potential energy of the system
    def potential(self, q, qd):
        m = self.config.m
        g = self.config.g

        U = 0.0
        y = 0.0

        y = q[:, 1]
        U += m[0] * g * y

        y = q[:, 4]
        U += m[1] * g * y

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
