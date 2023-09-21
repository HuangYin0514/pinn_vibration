# encoding: utf-8

import numpy as np
import torch
from tqdm import tqdm


class SolverBase:
    """
    A base class for solving Ordinary Differential Equations (ODEs).
    """

    def __init__(self, *args, **kwargs):
        pass

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
            if torch.any(torch.isnan(yi_next)):
                print("Stop time is ", self.t[idx].item())
                sol = sol[: idx + 1]
                t = t[: idx + 1]
                break
            else:
                sol[idx + 1] = yi_next

        # Permute the solution tensor to have the dimensions in the desired order
        # desired order(num, time, state)
        sol = sol.permute(1, 0, 2)

        return t, sol

    def step(self, func, ti, yi, dt, *args, **kwargs):
        """
        Computes the next state of the ODE system using a specific integration method.

        Args:
            func (callable): The ODE function to be solved.
            ti (float): The current time.
            yi (torch.Tensor): The current state of the system.
            dt (float): The time step for integration.
            *args: Additional positional arguments to be passed to the ODE function.
            **kwargs: Additional keyword arguments to be passed to the ODE function.

        Returns:
            torch.Tensor: The next state of the system.
        """
        # Implement the integration method (e.g., Euler's method or other numerical methods)
        # Replace this with the specific integration method you want to use
        # For example:
        # next_state = current_state + time_step * ode_function(current_time, current_state, *args, **kwargs)
        pass
