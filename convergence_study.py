"""
Grid / time-step convergence study for the wildfire forward solve.

Runs Fire.solvePDE (FD in space, RK4 in time) for every combination of
Nx = Ny in NX_LIST and Nt in NT_LIST, and records
    - max temperature over all time
    - max temperature at the final time
    - total burned area (fuel-weighted, interior nodes)
    - cost = Nx * Ny * Nt
Runs that diverge are skipped (stored as NaN).
Only the metrics are saved, to data/convergence_study.npz.

Place in the repository root (next to optimizer.py) and run:
    python convergence_study.py
"""
import time
import warnings
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import wildfire
from wildfire.utils.functions import G

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------
# Study settings
# ----------------------------------------------------------------------------
NX_LIST = [150, 200, 250, 300, 350, 400]   # Nx = Ny
NT_LIST = [150, 200, 250, 300, 350, 400]
DATA_DIR = Path(__file__).resolve().parent / "data"

# ----------------------------------------------------------------------------
# Physical setup (as in Optimization_testing2.ipynb, assets by coordinates)
# ----------------------------------------------------------------------------
kap, eps, alp, q = 1, 3e-1, 1e-3, 1
x_min, x_max = -100, 100
y_min, y_max = -100, 100
t_min, t_max = 0, 10

gamma = 7
w1 = lambda x, y, t: gamma * np.cos(np.pi / 4 + x * 0)
w2 = lambda x, y, t: gamma * np.sin(np.pi / 4 + x * 0)
V = lambda x, y, t: (w1(x, y, t), w2(x, y, t))

u0 = lambda x, y: 6 * G(x + 25, y + 25, 100)

# Asset boxes (x_lo, x_hi, y_lo, y_hi): the Nx=252 index-based assets in coordinates
ASSETS = [(14.34, 33.47, 18.33, 37.45),
          (14.34, 33.47, -9.56, 9.56)]


def fuel_and_upc(X, Y):
    B = np.where(X <= 0, 0.9, 0.2).astype(float)   # forest left, grass right
    B[(X >= 0) & (Y <= 0)] = 0.45                  # shrub bottom-right
    upc = np.ones_like(B)
    upc[B == 0.9], upc[B == 0.45], upc[B == 0.2] = 3.5, 3.0, 2.5
    for xl, xh, yl, yh in ASSETS:
        m = (X >= xl) & (X <= xh) & (Y >= yl) & (Y <= yh)
        B[m], upc[m] = 0.7, 3.0
    B[0, :] = B[-1, :] = B[:, 0] = B[:, -1] = 0    # Dirichlet boundary
    upc[0, :] = upc[-1, :] = upc[:, 0] = upc[:, -1] = 1.0
    return B, upc


b0 = lambda x, y: fuel_and_upc(x, y)[0]

control_params = {
    'v': 5, 'sigma_x': 5.5, 'sigma_y': 5.5, 'T': 3.0, 'k': 0.6, 'delta': 0.2,
    'control_vars': np.asarray([[-25, -25, np.pi / 4, 0], [0, 0, 3 * np.pi / 4, 3]]),
}


def run(N, Nt):
    X, Y = np.meshgrid(np.linspace(x_min, x_max, N), np.linspace(y_min, y_max, N))
    _, upc = fuel_and_upc(X, Y)
    physical_parameters = {'kap': kap, 'eps': eps, 'upc': upc, 'q': q, 'alp': alp,
                           'x_lim': (x_min, x_max), 'y_lim': (y_min, y_max),
                           't_lim': (t_min, t_max)}
    fire = wildfire.Fire(**physical_parameters, control_params=dict(control_params))
    try:
        t, X, Y, U, B = fire.solvePDE(N, N, Nt, u0, b0, V, space_method='FD',
                                  time_method='RK4', last=False, acc=2, sparse=False)
    except:
        raise RuntimeError("non-finite solution")                           

    dA = (X[0, 1] - X[0, 0]) * (Y[1, 0] - Y[0, 0])
    B0, BT = B[0], B[-1]
    fuel = np.zeros_like(B0, dtype=bool)
    fuel[1:-1, 1:-1] = B0[1:-1, 1:-1] > 1e-8
    burned = np.sum(1 - BT[fuel] / B0[fuel]) * dA
    return U.max(), U[-1].max(), burned


def main():
    nx, nt = np.array(NX_LIST), np.array(NT_LIST)
    shape = (len(nt), len(nx))                     # rows: Nt, columns: Nx
    umax_all = np.full(shape, np.nan)
    umax_final = np.full(shape, np.nan)
    burned = np.full(shape, np.nan)
    cost = (nx[None, :] * nx[None, :] * nt[:, None]).astype(float)

    print(f"{'Nx':>5} {'Nt':>5} {'cost':>9} {'umax_all':>9} {'umax_T':>8} {'burned':>8} {'time[s]':>8}")
    for i, Nt in enumerate(NT_LIST):
        for j, N in enumerate(NX_LIST):
            t0 = time.time()
            try:
                umax_all[i, j], umax_final[i, j], burned[i, j] = run(N, Nt)
                print(f"{N:5d} {Nt:5d} {cost[i, j]:9.1e} {umax_all[i, j]:9.3f} "
                      f"{umax_final[i, j]:8.3f} {burned[i, j]:8.1f} {time.time() - t0:8.0f}")
            except Exception:
                print(f"{N:5d} {Nt:5d} {cost[i, j]:9.1e}   diverged, skipped")

    DATA_DIR.mkdir(exist_ok=True)
    np.savez(DATA_DIR / "convergence_study.npz", Nx=nx, Nt=nt, cost=cost,
             umax_all=umax_all, umax_final=umax_final, burned_area=burned)

    metrics = [(umax_all, "Max temperature (all t)"),
               (umax_final, f"Max temperature (t = {t_max})"),
               (burned, "Total burned area")]
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(nt)))
    for data, label in metrics:
        plt.figure(figsize=(7, 4.5))
        for i, Nt in enumerate(nt):
            plt.plot(nx, data[i], 'o-', color=colors[i], label=f"Nt = {Nt}")
        plt.xlabel("Nx = Ny")
        plt.ylabel(label)
        plt.title(label)
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
