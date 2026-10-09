"""
Convergence test for the adjoint gradient.

This script runs two separate studies:

1. Spatial refinement:
   Nx = Ny is refined from 100 to 800, with Nt fixed.

2. Time refinement:
   Nt is refined from 150 to 300, with Nx = Ny fixed at 800.

The gradient is the physical gradient dJ/d[x1, y1, theta1, t1,
x2, y2, theta2, t2], not the normalized optimizer gradient.
"""

import numpy as np
import wildfire

from wildfire.utils.functions import G
from wildfire.utils import gradient


# ---------------------------------------------------------------------
# User settings
# ---------------------------------------------------------------------

RESULTS_FILE = "adjoint_gradient_convergence.npz"
SPATIAL_CSV_FILE = "adjoint_spatial_convergence.csv"
TIME_CSV_FILE = "adjoint_time_convergence.csv"

# Five spatial resolutions and five time resolutions.
Nx_values = np.asarray([100,250,350,375,400,425,450,600])
Nt_values = np.asarray([150, 188, 225, 263, 300])

# Keep the other discretization fixed in each separate study.
Nt_for_spatial_study = 300
Nx_for_time_study = 250
Ny_for_time_study = 250

# Set one of these to False if you only want to run one study.
RUN_SPATIAL_STUDY = True
RUN_TIME_STUDY = True


# ---------------------------------------------------------------------
# Physical and simulation parameters
# ---------------------------------------------------------------------

x_min = 0.0
x_max = 10.0
y_min = 0.0
y_max = 10.0
t_min = 0.0
t_max = 10.0

# This kap is deliberately small enough for explicit Euler on the
# requested 0--10 domain, including the Nx = Ny = 800 case.
kap = 2.5e-4
eps = 3.0e-1
alp = 1.0e-3
q = 1.0

# Small constant wind, chosen to keep the explicit upwind CFL reasonable.
wind_speed = 8.0e-2


def V(x, y, t):
    v1 = wind_speed * np.cos(np.pi / 4.0) * np.ones_like(x)
    v2 = wind_speed * np.sin(np.pi / 4.0) * np.ones_like(y)
    return v1, v2


# ---------------------------------------------------------------------
# Initial conditions, fuel map, phase-change map, and objective weights
# ---------------------------------------------------------------------

def make_initial_temperature(x, y):
    """
    Ignition near the lower-left side of the domain.
    """
    return 6.0 * G(x - 2.0, y - 2.0, 5.0e-2)


def make_initial_fuel(x, y):
    """
    Mostly homogeneous forest fuel, with slightly lower fuel in the two
    protected asset regions. Boundary values are zero.
    """
    B = 0.9 * np.ones_like(x)

    asset_1 = (
        (x >= 4.8) & (x <= 5.8) &
        (y >= 4.5) & (y <= 5.5)
    )

    asset_2 = (
        (x >= 4.8) & (x <= 5.8) &
        (y >= 6.0) & (y <= 7.0)
    )

    B[asset_1] = 0.7
    B[asset_2] = 0.7

    B[0, :] = 0.0
    B[-1, :] = 0.0
    B[:, 0] = 0.0
    B[:, -1] = 0.0

    return B


def make_phase_change_map(x, y):
    """
    u_pc is 3.5 in the ordinary fuel and 3.0 in each asset area.
    """
    u_pc = 3.5 * np.ones_like(x)

    asset_1 = (
        (x >= 4.8) & (x <= 5.8) &
        (y >= 4.5) & (y <= 5.5)
    )

    asset_2 = (
        (x >= 4.8) & (x <= 5.8) &
        (y >= 6.0) & (y <= 7.0)
    )

    u_pc[asset_1] = 3.0
    u_pc[asset_2] = 3.0

    return u_pc


def make_weights(x, y):
    """
    Terminal objective weights: low baseline weight everywhere and
    high value in the two asset regions.
    """
    weights = 0.5 * np.ones_like(x)

    asset_1 = (
        (x >= 4.8) & (x <= 5.8) &
        (y >= 4.5) & (y <= 5.5)
    )

    asset_2 = (
        (x >= 4.8) & (x <= 5.8) &
        (y >= 6.0) & (y <= 7.0)
    )

    weights[asset_1] = 10.0
    weights[asset_2] = 10.0

    return weights


# ---------------------------------------------------------------------
# Fixed two-drop control vector
# ---------------------------------------------------------------------

def make_random_control_vars():
    """
    Creates one reproducible random pair of drops.

    The time coordinates t1 and t2 are chosen randomly once, then kept
    identical for every mesh and time refinement run.

    Order:
    [x1, y1, theta1, t1, x2, y2, theta2, t2]
    """
    np.random.seed(27)

    control_vars = np.zeros((2, 4))

    control_vars[0, 0] = np.random.uniform(2.4, 3.5)
    control_vars[0, 1] = np.random.uniform(2.4, 3.5)
    control_vars[0, 2] = np.random.uniform(0.4, 1.2)
    control_vars[0, 3] = np.random.uniform(1.0, 2.5)

    control_vars[1, 0] = np.random.uniform(3.5, 4.6)
    control_vars[1, 1] = np.random.uniform(3.5, 4.6)
    control_vars[1, 2] = np.random.uniform(0.4, 1.2)
    control_vars[1, 3] = np.random.uniform(3.5, 5.5)

    return control_vars


def make_control_params(control_vars):
    """
    Same control structure used in your optimization notebook, but
    spatial scales are appropriate for a 0--10 domain.
    """
    control_params = {
        "v": 0.4,
        "sigma_x": 0.30,
        "sigma_y": 0.30,
        "T": 1.5,
        "k": 0.6,
        "delta": 0.2,
        "control_vars": control_vars.copy()
    }

    return control_params


# ---------------------------------------------------------------------
# One forward / adjoint / gradient evaluation
# ---------------------------------------------------------------------

def print_stability_numbers(Nx, Ny, Nt):
    dx = (x_max - x_min) / (Nx - 1)
    dy = (y_max - y_min) / (Ny - 1)
    dt = (t_max - t_min) / Nt

    diffusion_number = kap * dt * (1.0 / dx**2 + 1.0 / dy**2)

    v1 = wind_speed * np.cos(np.pi / 4.0)
    v2 = wind_speed * np.sin(np.pi / 4.0)
    convection_number = dt * (abs(v1) / dx + abs(v2) / dy)

    print("")
    print("Nx =", Nx, "Ny =", Ny, "Nt =", Nt)
    print("dx =", dx, "dy =", dy, "dt =", dt)
    print("Diffusion number =", diffusion_number, "(should be below about 0.5)")
    print("Convection CFL =", convection_number, "(should be below about 1.0)")


def compute_gradient_at_resolution(Nx, Ny, Nt, control_vars):
    """
    Run forward PDE, adjoint PDE, and then use your gradient.gradient
    class directly to calculate the eight physical gradient components.
    """
    print_stability_numbers(Nx, Ny, Nt)

    x = np.linspace(x_min, x_max, Nx)
    y = np.linspace(y_min, y_max, Ny)
    X0, Y0 = np.meshgrid(x, y)

    u_pc = make_phase_change_map(X0, Y0)
    weights = make_weights(X0, Y0)
    control_params = make_control_params(control_vars)

    physical_parameters = {
        "kap": kap,
        "eps": eps,
        "upc": u_pc,
        "alp": alp,
        "q": q,
        "x_lim": (x_min, x_max),
        "y_lim": (y_min, y_max),
        "t_lim": (t_min, t_max)
    }

    fire_object = wildfire.Fire(
        **physical_parameters,
        control_params=control_params
    )

    t, X, Y, U, B = fire_object.solvePDE(
        Nx,
        Ny,
        Nt,
        make_initial_temperature,
        make_initial_fuel,
        V,
        space_method="FD",
        time_method="Euler",
        last=False,
        acc=2,
        sparse=False
    )

    B0 = make_initial_fuel(X, Y)

    rho_u_terminal = np.zeros_like(B0)

    rho_b_terminal = np.zeros_like(B0)
    rho_b_terminal[1:-1, 1:-1] = (
        -weights[1:-1, 1:-1] /
        (B0[1:-1, 1:-1] + 1.0e-4)
    )

    rho_b_terminal[0, :] = 0.0
    rho_b_terminal[-1, :] = 0.0
    rho_b_terminal[:, 0] = 0.0
    rho_b_terminal[:, -1] = 0.0

    _, _, _, rho_u, rho_b = fire_object.solve_adjointPDE(
        Nx,
        Ny,
        Nt,
        rho_u_terminal,
        rho_b_terminal,
        U,
        B,
        V,
        space_method="FD",
        time_method="Euler",
        last=False,
        acc=2,
        sparse=False
    )

    gradient_calculator = gradient.gradient(
        control_vars,
        control_params,
        X,
        Y,
        t
    )

    gradient_vector = gradient_calculator.compute_gradient(
        U,
        B,
        rho_u,
        rho_b
    )

    dx = X[0, 1] - X[0, 0]
    dy = Y[1, 0] - Y[0, 0]

    objective = np.sum(
        weights * (1.0 - B[-1, :, :] / (B0 + 1.0e-4))
    ) * dx * dy

    gradient_norm = np.linalg.norm(gradient_vector)

    print("Objective =", objective)
    print("Gradient =", gradient_vector)
    print("Gradient L2 norm =", gradient_norm)

    return objective, gradient_vector, gradient_norm, dx, dy, t[1] - t[0]


# ---------------------------------------------------------------------
# Saving results
# ---------------------------------------------------------------------

def save_results(
    control_vars,
    spatial_N,
    spatial_Nt,
    spatial_dx,
    spatial_dy,
    spatial_dt,
    spatial_objective,
    spatial_gradients,
    spatial_gradient_norms,
    time_N,
    time_Nt,
    time_dx,
    time_dy,
    time_dt,
    time_objective,
    time_gradients,
    time_gradient_norms
):
    np.savez(
        RESULTS_FILE,
        control_vars=control_vars,

        spatial_N=spatial_N,
        spatial_Nt=spatial_Nt,
        spatial_dx=spatial_dx,
        spatial_dy=spatial_dy,
        spatial_dt=spatial_dt,
        spatial_objective=spatial_objective,
        spatial_gradients=spatial_gradients,
        spatial_gradient_norms=spatial_gradient_norms,

        time_N=time_N,
        time_Nt=time_Nt,
        time_dx=time_dx,
        time_dy=time_dy,
        time_dt=time_dt,
        time_objective=time_objective,
        time_gradients=time_gradients,
        time_gradient_norms=time_gradient_norms
    )

    spatial_table = np.column_stack((
        spatial_N,
        spatial_N,
        spatial_Nt,
        spatial_dx,
        spatial_dy,
        spatial_dt,
        spatial_objective,
        spatial_gradient_norms,
        spatial_gradients
    ))

    time_table = np.column_stack((
        time_N,
        time_N,
        time_Nt,
        time_dx,
        time_dy,
        time_dt,
        time_objective,
        time_gradient_norms,
        time_gradients
    ))

    header = (
        "Nx,Ny,Nt,dx,dy,dt,objective,gradient_l2,"
        "dJ_dx1,dJ_dy1,dJ_dtheta1,dJ_dt1,"
        "dJ_dx2,dJ_dy2,dJ_dtheta2,dJ_dt2"
    )

    np.savetxt(
        SPATIAL_CSV_FILE,
        spatial_table,
        delimiter=",",
        header=header,
        comments=""
    )

    np.savetxt(
        TIME_CSV_FILE,
        time_table,
        delimiter=",",
        header=header,
        comments=""
    )


# ---------------------------------------------------------------------
# Main refinement study
# ---------------------------------------------------------------------

def run_refinement_study():
    control_vars = make_random_control_vars()

    print("Fixed physical control variables:")
    print(control_vars)
    print("")
    print("The gradient order is:")
    print("[x1, y1, theta1, t1, x2, y2, theta2, t2]")

    number_spatial_cases = len(Nx_values)
    number_time_cases = len(Nt_values)

    spatial_N = Nx_values.astype(float)
    spatial_Nt = np.full(number_spatial_cases, Nt_for_spatial_study, dtype=float)
    spatial_dx = np.full(number_spatial_cases, np.nan)
    spatial_dy = np.full(number_spatial_cases, np.nan)
    spatial_dt = np.full(number_spatial_cases, np.nan)
    spatial_objective = np.full(number_spatial_cases, np.nan)
    spatial_gradients = np.full((number_spatial_cases, 8), np.nan)
    spatial_gradient_norms = np.full(number_spatial_cases, np.nan)

    time_N = np.full(number_time_cases, Nx_for_time_study, dtype=float)
    time_Nt = Nt_values.astype(float)
    time_dx = np.full(number_time_cases, np.nan)
    time_dy = np.full(number_time_cases, np.nan)
    time_dt = np.full(number_time_cases, np.nan)
    time_objective = np.full(number_time_cases, np.nan)
    time_gradients = np.full((number_time_cases, 8), np.nan)
    time_gradient_norms = np.full(number_time_cases, np.nan)

    if RUN_SPATIAL_STUDY:
        print("")
        print("Starting spatial refinement study")

        for i in range(number_spatial_cases):
            Nx = int(Nx_values[i])
            Ny = int(Nx_values[i])
            Nt = int(Nt_for_spatial_study)

            objective, grad, grad_norm, dx, dy, dt = (
                compute_gradient_at_resolution(
                    Nx,
                    Ny,
                    Nt,
                    control_vars
                )
            )

            spatial_dx[i] = dx
            spatial_dy[i] = dy
            spatial_dt[i] = dt
            spatial_objective[i] = objective
            spatial_gradients[i, :] = grad
            spatial_gradient_norms[i] = grad_norm

            save_results(
                control_vars,
                spatial_N,
                spatial_Nt,
                spatial_dx,
                spatial_dy,
                spatial_dt,
                spatial_objective,
                spatial_gradients,
                spatial_gradient_norms,
                time_N,
                time_Nt,
                time_dx,
                time_dy,
                time_dt,
                time_objective,
                time_gradients,
                time_gradient_norms
            )

    if RUN_TIME_STUDY:
        print("")
        print("Starting time refinement study")

        for i in range(number_time_cases):
            Nx = int(Nx_for_time_study)
            Ny = int(Ny_for_time_study)
            Nt = int(Nt_values[i])

            objective, grad, grad_norm, dx, dy, dt = (
                compute_gradient_at_resolution(
                    Nx,
                    Ny,
                    Nt,
                    control_vars
                )
            )

            time_dx[i] = dx
            time_dy[i] = dy
            time_dt[i] = dt
            time_objective[i] = objective
            time_gradients[i, :] = grad
            time_gradient_norms[i] = grad_norm

            save_results(
                control_vars,
                spatial_N,
                spatial_Nt,
                spatial_dx,
                spatial_dy,
                spatial_dt,
                spatial_objective,
                spatial_gradients,
                spatial_gradient_norms,
                time_N,
                time_Nt,
                time_dx,
                time_dy,
                time_dt,
                time_objective,
                time_gradients,
                time_gradient_norms
            )

    print("")
    print("Finished.")
    print("Saved NumPy results to:", RESULTS_FILE)
    print("Saved spatial table to:", SPATIAL_CSV_FILE)
    print("Saved time table to:", TIME_CSV_FILE)


if __name__ == "__main__":
    run_refinement_study()