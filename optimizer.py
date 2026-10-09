""""
Optimizer object, that will take in all the simulation params and initialize a wildfire object.
Calls to the "objective function" will be a call to this optimizer, that will return the burnt area evaluated according to weights.
Calls for gradient will also t=be to this object which will return the gradient.
"""
import numpy as np
import matplotlib.pyplot as plt
import wildfire
from wildfire.utils.functions import G
from wildfire.utils import plots, gradient

class optimizer_object:

    def __init__(self, physical_params, control_params, Nx, Ny, Nt, u0, b0, V, weights, normalization_ranges):
        self.physical_params = physical_params
        self.control_params = control_params
        self.Nx, self.Ny, self.Nt = Nx, Ny, Nt
        self.u0, self.b0, self.V = u0, b0, V
        self.weights = weights
        self.dx = (physical_params['x_lim'][1] - physical_params['x_lim'][0]) / (Nx-1)
        self.dy = (physical_params['y_lim'][1] - physical_params['y_lim'][0]) / (Ny-1)
        self.dt = (physical_params['t_lim'][1] - physical_params['t_lim'][0]) / Nt
        self.norm_range = normalization_ranges[:,1] - normalization_ranges[:,0]
        self.x_min = normalization_ranges[:,0]
    
    def denormalize(self, normalized_control_vars):
        """ Normalize all the control vars between 0 and 1
            self.normalization ranges should be a (num vars) x 2 array storing (min and max)
            control_vars should be 1d array over here.
        """
        control_vars = self.x_min + (normalized_control_vars * self.norm_range)
        return control_vars
    
    def normalize_gradient(self, gradient):
        return gradient * self.norm_range

    def forward_eval(self, control_vars, full_out=False):
        """
        control_vars should be a 4*num_drops array of control variables, arranged like
        x_i, y_i, theta_i, time_i --> input will be normalized and process later?
        """
        control_vars_2d = control_vars.reshape(-1, 4)
        self.control_params["control_vars"] = control_vars_2d
        self.fire_object = wildfire.Fire(**self.physical_params, control_params=self.control_params)
        t, X, Y, U, B = self.fire_object.solvePDE(self.Nx, self.Ny, self.Nt, self.u0, self.b0, self.V, space_method='FD', time_method = 'Euler', last=False, acc=2, sparse=False)
        self.X, self.Y, self.t =  X, Y, t

        # set gradient object as well in here. 
        # will ensure correct values get passed to gradient computer.
        if full_out:
            return t, X, Y, U, B
        else:
            return U, B
    
    def objective_function(self, normalized_control_vars): 
        control_vars = self.denormalize(normalized_control_vars)
        *_, objective = self._evaluate(control_vars)
       # print(f"normalized : {normalized_control_vars}")
       # print(f"vars : {control_vars}")
       # print(f"objective : {objective}")
        return objective
    
    def gradient(self, normalized_control_vars): 
        control_vars = self.denormalize(normalized_control_vars)
        t, X, Y, U, B, _ = self._evaluate(control_vars)
        _, rho_u, rho_b = self.backward_eval(U, B)
        control_vars_2d = control_vars.reshape(-1, 4)
        self.control_params["control_vars"] = control_vars_2d
        gradient_calculator = gradient.gradient(control_vars_2d, self.control_params, X, Y, t)
        grad = gradient_calculator.compute_gradient(U, B, rho_u, rho_b)

        normalized_grad = self.normalize_gradient(grad)
     #   print(f"normalized grad : {normalized_grad}")
        return normalized_grad


    def backward_eval(self, U, B):
        if not self.fire_object:
            print("Run Forward Simulation First.")
            return None
       
        B0 = self.b0(self.X ,self.Y)
        rho_u_terminal, rho_b_terminal = self.get_terminal_conditions(B0, self.weights)

        t1, X1, Y1, rho_u, rho_b = self.fire_object.solve_adjointPDE(self.Nx, self.Ny, self.Nt, rho_u_terminal, rho_b_terminal, U, B, self.V, 
            space_method='FD', time_method = 'Euler', last=False, acc=2, sparse=False)
        
        return t1, rho_u, rho_b
    
    def get_terminal_conditions(self, B0, weights):
        rho_u_terminal = np.zeros_like(B0) # # terminal condition is rho_u = 0 across the grid.
        rho_b_terminal = np.zeros_like(B0)
        rho_b_terminal[1:-1, 1:-1] = - weights[1:-1, 1:-1] / (B0[1:-1,1:-1] + 1e-4)
        # set boundary values to 0 for rho_b_terminal.
        rho_b_terminal[-1,:] = 0
        rho_b_terminal[0,:] = 0
        rho_b_terminal[:,0] = 0
        rho_b_terminal[:,-1] = 0

        return rho_u_terminal, rho_b_terminal
    
    def _evaluate(self, control_vars):
        """Run and cache the forward solve for one control vector."""
        control_vars = np.asarray(control_vars, dtype=float).ravel()

        cache_is_valid = (
            hasattr(self, "_cached_control_vars")
            and np.array_equal(control_vars, self._cached_control_vars)
        )

        if not cache_is_valid:
            t, X, Y, U, B = self.forward_eval(control_vars, full_out=True)

            objective = np.sum( self.weights * (1 - B[-1, :, :] / (self.b0(X, Y) + 1e-4))) * self.dx * self.dy
            #objective = np.sum( -1 * B[-1, :, :]) * self.dx * self.dy

            self._cached_control_vars = control_vars.copy()
            self._cached_t = t
            self._cached_X = X
            self._cached_Y = Y
            self._cached_U = U
            self._cached_B = B
            self._cached_objective = objective

        return (
            self._cached_t,
            self._cached_X,
            self._cached_Y,
            self._cached_U,
            self._cached_B,
            self._cached_objective,
        )
        
    # def plot(self, U, B):
        
        



        

