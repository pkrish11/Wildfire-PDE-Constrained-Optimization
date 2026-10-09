"""
Plot results produced by adjoint_gradient_refinement.py.
"""

import numpy as np
import matplotlib.pyplot as plt


RESULTS_FILE = "adjoint_gradient_convergence.npz"


def plot_one_study(
    refinement_values,
    gradient_norms,
    gradients,
    x_label,
    title_prefix
):
    """
    Plot:

    1. Total gradient L2 norm.
    2. Each signed derivative value.
    3. Absolute error in the total gradient norm relative to the
       finest result.
    4. Absolute error in each derivative relative to the finest result.
    """
    derivative_names = [
        "dJ/dx1",
        "dJ/dy1",
        "dJ/dtheta1",
        "dJ/dt1",
        "dJ/dx2",
        "dJ/dy2",
        "dJ/dtheta2",
        "dJ/dt2"
    ]

    reference_gradient = gradients[-1, :]
    reference_gradient_norm = gradient_norms[-1]

    norm_error = np.abs(gradient_norms - reference_gradient_norm)
    derivative_error = np.abs(gradients - reference_gradient)

    # --------------------------------------------------------------
    # Raw gradient values
    # --------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    axes[0].plot(
        refinement_values,
        gradient_norms,
        "o-",
        linewidth=2
    )
    axes[0].set_xlabel(x_label)
    axes[0].set_ylabel("Total gradient L2 norm")
    axes[0].set_title(title_prefix + ": total gradient norm")
    axes[0].grid(True)

    for j in range(8):
        axes[1].plot(
            refinement_values,
            gradients[:, j],
            "o-",
            label=derivative_names[j]
        )

    axes[1].set_xlabel(x_label)
    axes[1].set_ylabel("Gradient component value")
    axes[1].set_title(title_prefix + ": individual derivatives")
    axes[1].grid(True)
    axes[1].legend()

    plt.tight_layout()
    plt.show()

    # --------------------------------------------------------------
    # Errors relative to finest calculation
    # --------------------------------------------------------------
    # The finest point has exactly zero error. Do not put it on a log
    # scale because log(0) is undefined.
    coarse_values = refinement_values[:-1]
    coarse_norm_error = norm_error[:-1]
    coarse_derivative_error = derivative_error[:-1, :]

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    axes[0].loglog(
        coarse_values,
        coarse_norm_error,
        "o-",
        linewidth=2
    )
    axes[0].set_xlabel(x_label)
    axes[0].set_ylabel(
        "| ||gradient||_2 - ||gradient_ref||_2 |"
    )
    axes[0].set_title(
        title_prefix + ": total gradient-norm error"
    )
    axes[0].grid(True, which="both")

    for j in range(8):
        axes[1].loglog(
            coarse_values,
            coarse_derivative_error[:, j],
            "o-",
            label=derivative_names[j]
        )

    axes[1].set_xlabel(x_label)
    axes[1].set_ylabel("|derivative - reference derivative|")
    axes[1].set_title(
        title_prefix + ": individual derivative errors"
    )
    axes[1].grid(True, which="both")
    axes[1].legend()

    plt.tight_layout()
    plt.show()


def plot_convergence_results(results_file=RESULTS_FILE):
    """
    Load the saved results and create spatial and temporal convergence
    plots.

    The finest result in each study is used as the reference solution.
    """
    data = np.load(results_file)

    control_vars = data["control_vars"]

    print("Control variables used in every run:")
    print(control_vars)
    print("")
    print("Gradient order:")
    print("[x1, y1, theta1, t1, x2, y2, theta2, t2]")

    # --------------------------------------------------------------
    # Spatial convergence
    # --------------------------------------------------------------
    spatial_dx = data["spatial_dx"]
    spatial_gradient_norms = data["spatial_gradient_norms"]
    spatial_gradients = data["spatial_gradients"]

    valid_spatial = ~np.isnan(spatial_gradient_norms)

    if np.any(valid_spatial):
        plot_one_study(
            spatial_dx[valid_spatial],
            spatial_gradient_norms[valid_spatial],
            spatial_gradients[valid_spatial, :],
            "Spatial spacing dx",
            "Spatial refinement"
        )

    # --------------------------------------------------------------
    # Time convergence
    # --------------------------------------------------------------
    time_dt = data["time_dt"]
    time_gradient_norms = data["time_gradient_norms"]
    time_gradients = data["time_gradients"]

    valid_time = ~np.isnan(time_gradient_norms)

    if np.any(valid_time):
        plot_one_study(
            time_dt[valid_time],
            time_gradient_norms[valid_time],
            time_gradients[valid_time, :],
            "Time step dt",
            "Time refinement"
        )


if __name__ == "__main__":
    plot_convergence_results()