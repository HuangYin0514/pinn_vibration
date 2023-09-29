import os
from matplotlib import pyplot as plt
import numpy as np
from utils import tensors_to_numpy
from configs.config_plot import *


def calculate_dynamics_metrics(calculator, pred_data, gt_data):
    """
    Calculate dynamic metrics.

    Args:
        pred_data (tuple): A tuple containing q_hat, qt_hat, qtt_hat.
        gt_data (tuple): A tuple containing q, qt, qtt.

    Returns:
        tuple: A tuple containing metric values, metric error values, and a full error output string.
    """
    q_hat, qt_hat, qtt_hat = pred_data
    q, qt, qtt = gt_data

    energy = calculator.energy(q_hat, qt_hat)
    kinetic = calculator.kinetic(q_hat, qt_hat)
    potential = calculator.potential(q_hat, qt_hat)
    gt_energy = calculator.energy(q[:2], qt[:2])[0]
    phi = calculator.phi(q_hat, qt_hat, qtt_hat)
    phi_t = calculator.phi_t(q_hat, qt_hat, qtt_hat)
    phi_tt = calculator.phi_tt(q_hat, qt_hat, qtt_hat)

    # Calculate errors
    energy_error = energy - gt_energy
    mean_energy_error = np.mean((energy_error) ** 2)
    max_energy_error = np.max(energy_error)
    phi_error = np.max(phi)
    phi_t_error = np.max(phi_t)
    phi_tt_error = np.max(phi_tt)

    metric_value = [energy, kinetic, potential, phi, phi_t, phi_tt]
    metric_error_value = [
        mean_energy_error,
        max_energy_error,
        phi_error,
        phi_t_error,
        phi_tt_error,
    ]

    # Generate error output strings
    mean_energy_output = f"Mean energy error: {mean_energy_error:.4e}"
    energy_output = f"Max energy error: {max_energy_error:.4e}"
    phi_output = f"Phi error: {phi_error:.4e}"
    phi_t_output = f"Phi_t error: {phi_t_error:.4e}"
    phi_tt_output = f"Phi_tt error: {phi_tt_error:.4e}"
    output_log_list = [
        mean_energy_output,
        energy_output,
        phi_output,
        phi_t_output,
        phi_tt_output,
    ]

    return metric_value, metric_error_value, output_log_list


def plot_dynamics_metrics(config, pred_data, gt_data, t):
    """
    Plot dynamic metrics.

    Args:
        config (Config): An object containing configuration settings.
        predicted_states (tuple): A tuple containing predicted states (q_hat, qt_hat, qtt_hat).
        ground_truth_states (tuple): A tuple containing ground truth states (q, qt, qtt).
        time_points (array-like): Time points for the data.

    Returns:
        plt.Figure: A matplotlib Figure object containing the plots.
    """
    ################################
    q_hat, qt_hat, qtt_hat = pred_data
    q, qt, qtt = gt_data

    # Concatenate states
    all_states = np.concatenate([q, qt, qtt], axis=-1)
    all_states_hat = np.concatenate([q_hat, qt_hat, qtt_hat], axis=-1)

    # Calculate the number of subplots
    fig_num = config.dof * 3
    line_num = fig_num // config.dof * 2
    row_num = config.dof

    # Create the figure and subplots
    fig, axs = plt.subplots(
        line_num, row_num, figsize=(4 * row_num, 3 * line_num), dpi=DPI
    )

    for check_dim in range(fig_num):
        index_1 = check_dim // row_num
        index_2 = check_dim % row_num

        subfig = axs[index_1, index_2]
        subfig.set_xlabel("$t$ ($-$)")
        subfig.set_ylabel("$q$ ($-$)")
        subfig.plot(t, all_states[:, check_dim], label="GT")
        subfig.plot(t, all_states_hat[:, check_dim], c="y", label="net")
        subfig.legend()

        index_1 = (check_dim + fig_num) // row_num
        index_2 = (check_dim + fig_num) % row_num
        subfig = axs[index_1, index_2]
        subfig.set_xlabel("$t$ ($-$)")
        subfig.set_ylabel("$err$ ($-$)")
        subfig.plot(t, np.abs(all_states - all_states_hat)[:, check_dim], label="error")
        subfig.legend()

    plt.tight_layout()
