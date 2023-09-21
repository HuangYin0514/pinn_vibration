# -*- coding: utf-8 -*-
import torch
from .base import SolverBase  # 基于 SolverBase 类的继承


class Euler(SolverBase):

    def __init__(self, *args, **kwargs):
        super(Euler, self).__init__(*args, **kwargs)

    def step(self, func, t0, y0, dt, *args, **kwargs):
        """
        Execute a single time step of numerical integration using the Euler method.

        Args:
            func: The right-hand side function of the differential equation, i.e., dy/dt = func(t, y).
            t0: Current time.
            y0: Current value of state variables.
            dt: Time step.
            *args, **kwargs: Other possible parameters.

        Returns:
            The value of state variables at the next time step.
        """
        # Calculate k1 = func(t0, y0)
        k1 = func(t0, y0)

        # Update the value of state variables using the Euler method
        y1 = y0 + dt * k1

        return y1


class ImplicitEuler(SolverBase):

    def __init__(self, *args, **kwargs):
        super(ImplicitEuler, self).__init__(*args, **kwargs)

    def step(self, func, t0, y0, dt, dof, *args, **kwargs):
        # Split the input state tensor into position, velocity, and lambda parts
        qi_0, qti_0, lam = torch.tensor_split(y0, (dof, dof * 2), dim=-1)
        
        # Update the state 
        new = torch.cat([qi_0, qti_0, lam], dim=-1)
        dy = func(t0, new)
        qi_1, qtti_1, lam_1 = torch.tensor_split(dy, (dof, dof * 2), dim=-1)
        K1 = qtti_1

        # Update the state 
        new = torch.cat([qi_0 + dt * qti_0, qti_0 + dt * K1, lam], dim=-1)
        dy = func(t0, new)
        qi_2, qtti_2, lam_2 = torch.tensor_split(dy, (dof, dof * 2), dim=-1)
        K2 = qtti_2

        # Update the position and velocity components using the implicit Euler formula
        qi = qi_0 + dt / 2 * (qti_0 + qti_0 + dt * K1)
        qti = qti_0 + dt / 2 * (K1 + K2)

        # Combine the updated position, velocity, and lambda tensors
        y1 = torch.cat([qi, qti, lam], dim=-1)

        return y1
