import numpy as np
import matplotlib.pyplot as plt
import wildfire
from wildfire.utils.functions import G
from wildfire.utils import plots, gradient
from optimizer import optimizer_object
import time
import cProfile, pstats
import scipy.optimize as opt

# callback helper function to store iteration data
def callback(intermediate_result):
    r = intermediate_result
    iteration_history.append({
        "nit": r.nit,
        "z": r.x.copy(),
        "control_vars": opt_obj.denormalize(r.x).copy(),
        "fun": r.fun,
        "optimality": r.optimality,
        "constr_violation": r.constr_violation,
        "tr_radius": r.tr_radius,
        "nfev": r.nfev,
    })
    return False          

## Physical and Simulation Parameters
compute_discretization = True

# Model parameters #
kap = 1
eps = 3e-1
# upc = 3.
alp = 0.09 # default is 1e-3
q = 1
x_min, x_max = -100, 100
y_min, y_max = -100, 100
t_min, t_max = 0, 20

# Numerical #
# Space
space_method = 'FD'
Nx = 200
Ny = 200
acc = 2
sparse = False

# Time
time_method = 'RK4'
Nt = 175
last = False

# Wind effect
gamma = 4
w1 = lambda x, y, t: gamma * np.cos(np.pi/4 + x * 0)
w2 = lambda x, y, t: gamma * np.sin(np.pi/4 + x * 0)
V = lambda x, y, t: (w1(x, y, t), w2(x, y, t))
def changing_wind(x, y, t):
    if t<5:
        return (w1(x, y, t), w2(x, y, t))
    else:
        return (gamma * np.sin(0 + x * 0), gamma * np.cos(0 + x * 0))
print(f"Wind speed is {gamma}")

### COMPUTING PECLET AND CFL NUMBERS - constant wind setting
if compute_discretization: 
    dx, dy = (x_max - x_min) / Nx, (y_max - y_min) / Ny             
    dt_cfl_adv = 1 / ((gamma * np.cos(np.pi/4))/dx + (gamma * np.sin(np.pi/4))/dy) ## max dt for advection stability (CFL < 1)
    dt_cfl_diff = (dx**2) / 4*kap ## max dt for diffusion stability (CFL < 1)

    dt = np.min([dt_cfl_adv, dt_cfl_diff])
    print(f"dx : {dx} , dy : {dy} , dt : {dt}")
    Nt_min = np.ceil((t_max - t_min) / dt ).astype(int)
    print(f"Minimum Nt for satisfying CFL is {Nt_min}")
print(f"Nx : {Nx}, Nt : {Nt}, ")

# Initial conditions
u0 = lambda x, y: 6 * G(x+25, y+25, 100)

# test case initial fuel 
fuel_properties = {"forest_beta":0.9, "forest_u":3.5, "shrub_beta":0.45, "shrub_u":3.0, "grass_beta":0.2, "grass_u":2.5}
def b_test_intermediate(x, y):
    x_rows, x_cols = x.shape
    y_rows, y_cols = y.shape
    xmid = (x[-1,0] - x[0,0]) // 2
    ymid = (y[0,-1] - y[0,0]) //2
    forest_beta, shrub_beta, grass_beta = 0.9, 0.45, 0.2
    B = np.where(x <= xmid, forest_beta, grass_beta).astype(float)
    shrub_mask = (x >= xmid) & (y <= ymid)   
    B[shrub_mask] = shrub_beta

    # homogenous dirichlet boundary conditions
    B[0,:] = np.zeros(x_cols)
    B[-1,:] = np.zeros(x_cols)
    B[:,0] = np.zeros(y_rows)
    B[:,-1] = np.zeros(y_rows)
    return B

# initial u_pc (ignition temp) conditions
B_test = b_test_intermediate(*np.meshgrid(np.linspace(x_min, x_max, Nx), np.linspace(y_min, y_max, Ny)))
u_pc = np.ones_like(B_test)
u_pc[B_test==0.9] = 3.5 # forest
u_pc[B_test==0.45] = 3.0 # shrub
u_pc[B_test==0.2] = 2.5 # grass

# asset/weight map, 2 assets on the left-hand side directly in line with fire.
weights = np.ones_like(B_test) * 0.5 # 0.5 weight everywhere in general
asset_weight = 100
x_num, y_num = np.shape(B_test)
asset_size = 12
# first asset
x_asset1, y_asset1 = (2*x_num//4)+35, (2*y_num//4)+30
weights[x_asset1-asset_size:x_asset1+asset_size,  y_asset1-asset_size : y_asset1+asset_size] = asset_weight
u_pc[x_asset1-asset_size:x_asset1+asset_size,  y_asset1-asset_size : y_asset1+asset_size] = 3.0
# 2nd asset
x_asset2, y_asset2 = (2*x_num//4), (2*y_num//4)+30
weights[x_asset2-asset_size:x_asset2+asset_size,  y_asset2-asset_size : y_asset2+asset_size] = asset_weight
u_pc[x_asset2-asset_size:x_asset2+asset_size,  y_asset2-asset_size : y_asset2+asset_size] = 3.0

def b0_test(x, y):
    B = b_test_intermediate(x, y)
    x_num, y_num = np.shape(B)
    asset_size = 12
    x_asset1, y_asset1 = (2*x_num//4)+35, (2*y_num//4)+30
    x_asset2, y_asset2 = (2*x_num//4), (2*y_num//4)+30
    B[x_asset1-asset_size:x_asset1+asset_size,  y_asset1-asset_size : y_asset1+asset_size] = 0.7 # 0.7 fuel at assets.
    B[x_asset2-asset_size:x_asset2+asset_size,  y_asset2-asset_size : y_asset2+asset_size] = 0.7
    return B


### PARAMETERS ###

# Physical parameters for the model
physical_parameters = {    
    'kap': kap, # diffusion coefficient
    'eps': eps, # inverse of activation energy
    'upc': u_pc, # u phase change map.
    'q': q, # reaction heat
    'alp': alp, # natural convection,
    # Domain
    'x_lim': (x_min, x_max), # x-axis domain 
    'y_lim': (y_min, y_max), # y-axis domain
    't_lim': (t_min, t_max) # time domain
}

drop_1 = [0, 0, (3/4)*np.pi, 1]
drop_2 = [20, 20, (3/4)*np.pi, 4]
control_vars = np.asarray([drop_1, drop_2])

# CONTROL PARAMETERS
control_params = {
    'v':5,          # velocity of plane
    'sigma_x':5.,  # size of drop
    'sigma_y':5.,  # size of drop
    'T': 3.5 ,
    'k':1.0, # original value was 0.5
    'delta': 0.6, # original value was 0.6
    'control_vars':control_vars
}

### Optimization run
iteration_history = []

normalization_ranges = np.array([[x_min, x_max], [y_min, y_max], [0, 2*np.pi], [0, 10], [x_min, x_max], [y_min, y_max], [0, 2*np.pi], [0, 10]])
opt_obj = optimizer_object(physical_parameters, control_params, Nx, Ny, Nt, u0, b0_test, V, weights, normalization_ranges)

x0 = np.asarray([-25, -25, (1/4)*np.pi, 0, 0, 0, (3/4)*np.pi, 4]) # initial guess
z0 = (x0 - normalization_ranges[:,0])/ (normalization_ranges[:,1] - normalization_ranges[:,0]) 
J_ref = opt_obj.objective_function(z0)

T = control_params["T"]
t_max = physical_parameters["t_lim"][1]
lb = np.asarray([0.2, 0.2, -np.inf, 0, 0.2, 0.2, -np.inf, 0]) # normalized lower bounds. angles are unbounded. spatial lower bound is x,y=-60.
ub = np.asarray([0.8, 0.8, np.inf, 1-(2*T/t_max), 0.8, 0.8, np.inf, 1-(2*T/t_max)]) # normalized upper bounds. spatial is x/y=0.6. time is 1-(2*T) and 1-T.
A = np.zeros((1, 8))
A[0, 3] = -1    # x4
A[0, 7] =  1 
t_range = normalization_ranges[3,1] - normalization_ranges[3,0]
time_constraint = opt.LinearConstraint(A, T/t_range, np.inf) # second drop must be atleast after the first drop finishes.
bounds_z = opt.Bounds(lb=lb, ub=ub)

func_scaled = lambda control_params : opt_obj.objective_function(control_params) / J_ref
jac_scaled = lambda control_params : opt_obj.gradient(control_params) / J_ref


res = opt.minimize(func_scaled, x0=z0, args=(), method="trust-constr", jac=jac_scaled, bounds=bounds_z, constraints=time_constraint,
                     options={"maxiter":100,"verbose":2}, callback=callback)

print(res)