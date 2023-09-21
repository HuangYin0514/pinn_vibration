# -*- coding: utf-8 -*-
import numpy as np
import torch
from tqdm import tqdm

from .base import SolverBase


class RK4(SolverBase):
    def __init__(self, *args, **kwargs):
        super(RK4, self).__init__(*args, **kwargs)

    def step(self, func, t0, y0, dt, *args, **kwargs):
        k1 = dt * func(t0, y0)
        k2 = dt * func(t0 + 0.5 * dt, y0 + 0.5 * k1)
        k3 = dt * func(t0 + 0.5 * dt, y0 + 0.5 * k2)
        k4 = dt * func(t0 + dt, y0 + k3)
        y1 = y0 + (k1 + k2 * 2 + k3 * 2 + k4) / 6
        return y1


class RK4_high_order(SolverBase):
    """
    仅限于输出为q_ori, dq_ori, lam的情况
    """

    def __init__(self, *args, **kwargs):
        super(RK4_high_order, self).__init__(*args, **kwargs)

    def solve(self, func, t0, t1, dt, y0, device=None, dtype=None, *args, **kwargs):
        """
        Solves an ODE using a numerical integration method.

        Args:
            func (callable): The ODE function to be solved.
            t0 (float): The initial time.
            t1 (float): The final time.
            dt (float): The time step for integration.
            y0 (torch.Tensor): The initial state of the system.
            device (torch.device, optional): The device to use for computation (default: None).
            dtype (torch.dtype, optional): The data type to use for computation (default: None).
            *args: Additional positional arguments to be passed to the ODE function.
            **kwargs: Additional keyword arguments to be passed to the ODE function.

        Returns:
            torch.Tensor: The time values.
            torch.Tensor: The solution trajectory.

        Raises:
            Exception: If NaN values are encountered during integration.
        """

        # Generate a time array from initial_time to final_time with time_step
        t = np.arange(t0, t1 + dt, dt)
        t = torch.tensor(t, device=device, dtype=dtype)

        # Initialize the solution tensor with the initial state
        sol = torch.empty(len(t), *y0.shape, device=device, dtype=dtype)
        sol[0] = y0

        # Create a progress bar
        pbar = tqdm(range(len(t) - 1), desc="solving ODE...")
        for idx in pbar:
            ti = t[idx]
            yi = sol[idx]

            # Calculate the next state using the provided ODE function
            yi_next = self.step(func, ti, yi, dt, *args, **kwargs).clone().detach()

            # Check for NaN values in the next state
            stop_condition_1 = torch.any(torch.isnan(yi_next))
            # 杆16与杆52的夹角
            stop_condition_2 = sum((yi_next[0, 0:3] - yi_next[0, 15:18]) * (yi_next[0, 12:15] - yi_next[0, 3:6])) < 0
            if stop_condition_1 or stop_condition_2:
                print("stop time is ", ti.item())
                sol = sol[: idx + 1]
                t = t[: idx + 1]
                break
            else:
                sol[idx + 1] = yi_next

        # Permute the solution tensor to have the dimensions in the desired order
        # desired order(num, time, state)
        sol = sol.permute(1, 0, 2)

        return t, sol

    def step(self, func, t0, y0, dt, dof, *args, **kwargs):
        q_ori, dq_ori, lam = torch.tensor_split(y0, (dof, dof * 2), dim=-1)

        k1 = q_ori
        dk1 = dq_ori
        dqddqlam = func(t0, torch.cat([k1, dk1, lam], dim=-1))
        dq, ddk1, lam = torch.tensor_split(dqddqlam, (dof, dof * 2), dim=-1)

        k2 = k1 + 0.5 * dt * dk1
        dk2 = dk1 + 0.5 * dt * ddk1
        dqddqlam = func(t0, torch.cat([k2, dk2, lam], dim=-1))
        dq, ddk2, lam = torch.tensor_split(dqddqlam, (dof, dof * 2), dim=-1)

        k3 = k1 + 0.5 * dt * dk2
        dk3 = dk1 + 0.5 * dt * ddk2
        dqddqlam = func(t0, torch.cat([k3, dk3, lam], dim=-1))
        dq, ddk3, lam = torch.tensor_split(dqddqlam, (dof, dof * 2), dim=-1)

        k4 = k1 + dt * dk3
        dk4 = dk1 + dt * ddk3
        dqddqlam = func(t0, torch.cat([k4, dk4, lam], dim=-1))
        dq, ddk4, lam = torch.tensor_split(dqddqlam, (dof, dof * 2), dim=-1)

        q_next = k1 + dt * dk1 + (dt**2) / 6 * (ddk1 + ddk2 + ddk3)
        dq_next = dk1 + dt / 6 * (ddk1 + 2 * ddk2 + 2 * ddk3 + ddk4)

        return torch.cat([q_next, dq_next, lam], dim=-1)
