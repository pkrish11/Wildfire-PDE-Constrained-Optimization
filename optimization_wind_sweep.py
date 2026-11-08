"""
Run the drop optimization for several wind speeds in parallel (one process per wind).

For each wind speed:
  - forward run with NO drops (x = empty array)          -> no_drop_obj, no_drop_U, no_drop_B
  - objective for the initial guess x0                    -> init_obj
  - trust-constr optimization, fields at the last iterate -> drop_obj, drop_U, drop_B
Results per wind -> data/optimization_runs/gamma=<g>.npz, summary table printed at the end.

Run from the repository root:   python optimization_wind_sweep.py
(Must be run as a script, not inside Jupyter: macOS multiprocessing re-imports this file.)
"""
import os
# One BLAS thread per process (set BEFORE numpy is imported, workers inherit it)
for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[var] = "1"

import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

# ----------------------------------------------------------------------------
# Sweep settings
# ----------------------------------------------------------------------------
WINDS = [4, 5, 6, 7, 8, 9, 10]
T_MAX = {4: 20, 5: 20, 6: 20, 7: 20, 8: 15, 9: 14, 10: 13}   # fire stays in the domain
MAX_WORKERS = 4 # min(len(WINDS), max(1, (os.cpu_count() or 2) - 2))
MAXITER = {4: 100, 5: 100, 6: 100, 7: 100, 8: 150, 9: 150, 10: 150}   # more iterations allowed for stronger wind settings
NX = NY = 200
OUT_DIR = Path(__file__).resolve().parent / "data" / "optimization_runs_slsqp"
SAVE_DTYPE = np.float32          # fields saved as float32 (halves the file size); use np.float64 for full precision

X0 = np.asarray([-25, -25, (1/4) * np.pi, 0.0, 0, 0, (3/4) * np.pi, 4])   # initial guess


# ----------------------------------------------------------------------------
# Stable Nt for RK4 + 3rd-order upwind (80 % of the von Neumann limit)
# ----------------------------------------------------------------------------
def stable_Nt(t_lims, x_lims, y_lims, Nx, Ny, v_x, v_y, kap=1.0, safety=0.8, damping=0.0, n_theta=401):
    dx = (x_lims[1] - x_lims[0]) / (Nx - 1)
    dy = (y_lims[1] - y_lims[0]) / (Ny - 1)
    th = np.linspace(-np.pi, np.pi, n_theta)
    TX, TY = np.meshgrid(th, th)

    def sym(t, v, h):
        e = np.exp(1j * t)
        return kap * (e - 2 + 1 / e) / h**2 - abs(v) * (2 * e + 3 - 6 / e + 1 / e**2) / (6 * h)

    lam = sym(TX, v_x, dx) + sym(TY, v_y, dy) - damping
    R = lambda z: 1 + z + z**2 / 2 + z**3 / 6 + z**4 / 24
    lo, hi = 0.0, 1.0
    while np.abs(R(hi * lam)).max() <= 1 + 1e-12:
        hi *= 2
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if np.abs(R(mid * lam)).max() <= 1 + 1e-12 else (lo, mid)
    return int(np.ceil((t_lims[1] - t_lims[0]) / (safety * lo)))


# ----------------------------------------------------------------------------
# Problem setup (same as optimization_runs.py), built inside each worker
# ----------------------------------------------------------------------------
def build_problem(gamma, t_max):
    from wildfire.utils.functions import G

    kap, eps, alp, q = 1, 3e-1, 0.09, 1
    x_min, x_max, y_min, y_max, t_min = -100, 100, -100, 100, 0

    V = lambda x, y, t: (gamma * np.cos(np.pi / 4 + x * 0), gamma * np.sin(np.pi / 4 + x * 0))
    u0 = lambda x, y: 6 * G(x + 25, y + 25, 100)

    # fuel_properties used in this setup 
    # {"forest_beta":0.9, "forest_u":3.5, "shrub_beta":0.45, "shrub_u":3.0, "grass_beta":0.2, "grass_u":2.5}
    def b_test_intermediate(x, y):
        xmid = (x[-1, 0] - x[0, 0]) // 2
        ymid = (y[0, -1] - y[0, 0]) // 2
        B = np.where(x <= xmid, 0.9, 0.2).astype(float)
        B[(x >= xmid) & (y <= ymid)] = 0.45
        B[0, :] = B[-1, :] = B[:, 0] = B[:, -1] = 0
        return B

    B_test = b_test_intermediate(*np.meshgrid(np.linspace(x_min, x_max, NX), np.linspace(y_min, y_max, NY)))
    u_pc = np.ones_like(B_test)
    u_pc[B_test == 0.9], u_pc[B_test == 0.45], u_pc[B_test == 0.2] = 3.5, 3.0, 2.5
    weights = np.ones_like(B_test) * 0.5
    a = 12
    assets = [((2 * NX // 4) + 35, (2 * NY // 4) + 30), ((2 * NX // 4), (2 * NY // 4) + 30)]
    masks = []
    for r, c in assets:
        weights[r - a:r + a, c - a:c + a] = 100
        u_pc[r - a:r + a, c - a:c + a] = 3.0
        m = np.zeros_like(B_test, bool); m[r - a:r + a, c - a:c + a] = True
        masks.append(m)

    def b0_test(x, y):
        B = b_test_intermediate(x, y)
        for r, c in assets:
            B[r - a:r + a, c - a:c + a] = 0.7
        return B

    physical_parameters = {'kap': kap, 'eps': eps, 'upc': u_pc, 'q': q, 'alp': alp,
                           'x_lim': (x_min, x_max), 'y_lim': (y_min, y_max), 't_lim': (t_min, t_max)}
    control_params = {'v': 5, 'sigma_x': 5., 'sigma_y': 5., 'T': 3.5, 'k': 1.0, 'delta': 0.6,
                      'control_vars': X0.reshape(2, 4).copy()}
    vxy = gamma * np.cos(np.pi / 4)
    Nt = stable_Nt((t_min, t_max), (x_min, x_max), (y_min, y_max), NX, NY, vxy, vxy,
                   kap=kap, damping=control_params['k'])
    # drop start times live in [0, t_max - T] so every drop finishes before t_max
    T = control_params['T']
    normalization_ranges = np.array([[x_min, x_max], [y_min, y_max], [0, 2 * np.pi], [0, t_max]] * 2, float)
    return dict(physical_parameters=physical_parameters, control_params=control_params, Nt=Nt,
                u0=u0, b0=b0_test, V=V, weights=weights, ranges=normalization_ranges, masks=masks)


# ----------------------------------------------------------------------------
# One wind speed (runs in its own process)
# ----------------------------------------------------------------------------
def run_wind(gamma):
    import scipy.optimize as opt
    from optimizer import optimizer_object

    t0 = time.time()
    t_max = T_MAX[gamma]
    max_iter = MAXITER[gamma]
    P = build_problem(gamma, t_max)
    rng = P["ranges"]
    t_lims = P["physical_parameters"]["t_lim"]
    opt_obj = optimizer_object(P["physical_parameters"], dict(P["control_params"]), NX, NY, P["Nt"],
                               P["u0"], P["b0"], P["V"], P["weights"], rng)
    masks = P["masks"]
    lost = lambda B: [float(np.mean(1 - B[-1][m] / B[0][m])) for m in masks]

    # --- 1) no drops: forward evaluation with an empty control vector ---
    t, X, Y, U_nd, B_nd, J_nd = opt_obj._evaluate(np.array([]))
    U_nd, B_nd = U_nd.astype(SAVE_DTYPE), B_nd.astype(SAVE_DTYPE)
    dmg_nd = lost(B_nd)

    # --- 2) initial guess ---
    z0 = (X0 - rng[:, 0]) / (rng[:, 1] - rng[:, 0])
    J_init = opt_obj.objective_function(z0)

    # --- 3) optimization ---
    T = P["control_params"]["T"]
    t_range = rng[3, 1] - rng[3, 0]              # = t_max 
    ub_t1 = (t_max - 2 * T) / t_range           # drop 1 must leave room for drop 2
    ub_t2 = (t_max - T) / t_range
    lb_t2 = T / t_range                         # second drop must be atleast after the first drop ends
    lb = np.asarray([0.2, 0.2, -np.inf, 0, 0.2, 0.2, -np.inf, lb_t2])
    ub = np.asarray([0.8, 0.8, np.inf, ub_t1, 0.8, 0.8, np.inf, ub_t2])
    A = np.zeros((1, 8)); A[0, 3], A[0, 7] = -1, 1
    time_constraint = opt.LinearConstraint(A, T / t_range, np.inf)   # t2 - t1 >= T

    """ Callback function to work with trust-constr method only

    history = []
    def callback(intermediate_result):          # SciPy needs exactly this argument name
        r = intermediate_result
        history.append([r.nit, r.fun * J_init, r.optimality, r.constr_violation, *opt_obj.denormalize(r.x)])
        return False
    """

    # Callback function to work with SLSQP
    

    res = opt.minimize(lambda z: opt_obj.objective_function(z) / J_init, x0=z0, method="slsqp",
                       jac=lambda z: opt_obj.gradient(z) / J_init, bounds=opt.Bounds(lb, ub),
                       constraints=time_constraint, options={"maxiter": max_iter},
                       callback=None)
    

    # --- fields and objective at the last iterate ---
    x_opt = opt_obj.denormalize(res.x)
    _, _, _, U_d, B_d, J_d = opt_obj._evaluate(x_opt)
    U_d, B_d = U_d.astype(SAVE_DTYPE), B_d.astype(SAVE_DTYPE)
    dmg_d = lost(B_d)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT_DIR / f"gamma={gamma}.npz",
        gamma=gamma, t_lims=np.array(t_lims, float), Nt=P["Nt"], Nx=NX, Ny=NY,
        t=t, x=X[0], y=Y[:, 0],
        no_drop_obj=J_nd, no_drop_U=U_nd, no_drop_B=B_nd,
        no_drop_A1_lost=dmg_nd[0], no_drop_A2_lost=dmg_nd[1],
        init_obj=J_init, x0=X0,
        drop_obj=J_d, drop_U=U_d, drop_B=B_d, x_opt=x_opt, z_opt=res.x,
        drop_A1_lost=dmg_d[0], drop_A2_lost=dmg_d[1],
        nit=res.nit, status=res.status, success=res.success, # history=None, optimality=res.optimality,
        message=str(res.message), normalization_ranges=rng,
        weights=P["weights"], asset1=masks[0], asset2=masks[1])
    return dict(wind=gamma, t_max=t_max, Nt=P["Nt"], J_none=J_nd, J_init=J_init, J_opt=J_d,
                dmg_none=dmg_nd, dmg_opt=dmg_d, nit=res.nit, # optimality=res.optimality,
                minutes=(time.time() - t0) / 60)


# ----------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"Running winds {WINDS} on {MAX_WORKERS} processes")
    results = []
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(run_wind, g): g for g in WINDS}
        for f in as_completed(futures):
            g = futures[f]
            try:
                r = f.result()
                results.append(r)
                print(f"wind {g} done in {r['minutes']:.1f} min "
                      f"(J opt / J none = {r['J_opt'] / r['J_none']:.3f})", flush=True)
            except Exception as e:
                print(f"wind {g} FAILED: {e!r}", flush=True)

    print(f"\n{'wind':>4} {'t_max':>5} {'Nt':>4} | {'J none':>9} {'J init':>9} {'J opt':>9} | "
          f"{'A1 none':>7} {'A1 opt':>7} | {'A2 none':>7} {'A2 opt':>7} | {'nit':>4} ")
    for r in sorted(results, key=lambda r: r["wind"]):
        print(f"{r['wind']:4d} {r['t_max']:5g} {r['Nt']:4d} | {r['J_none']:9.0f} {r['J_init']:9.0f} {r['J_opt']:9.0f} | "
              f"{r['dmg_none'][0]:7.0%} {r['dmg_opt'][0]:7.0%} | "
              f"{r['dmg_none'][1]:7.0%} {r['dmg_opt'][1]:7.0%} | {r['nit']:4d}")
