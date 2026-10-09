""" 
Class designed to compute gradient, with input as parameters, u, beta, rho_u and rho_beta fields.
"""
import numpy as np
from .functions import Q_func, M_func, time_control, spray_gaussian

class gradient:

    def __init__(self, control_vars, control_params, X, Y, time_arr):
        """
        Initialize gradient object with control parameters as input.
        control_params is list of constant values like T, sigma_x, etc.
        Control vars should be an object with:
        * num rows = num water drops
        * column wise data entry : 0 = x_i, 1 = y_i, 2 = theta_i, 3 = t_i

        X,Y are mesh grids. T is array of values at which time has been discretized.
        """
        self.num_drops = np.shape(control_vars)[0] # number of water drops to be made.
        self.control_vars = control_vars
        for key, value in control_params.items():
            setattr(self, key, value)
        self.X, self.Y, self.time_arr = X, Y, time_arr[:, None, None]
        self.dx = X[0, 1] - X[0, 0]
        self.dy = Y[1, 0] - Y[0, 0]
        self.dt = time_arr[1] - time_arr[0]
    
    # functions to compute gradient for x,y,
    def compute_gradient(self, u, beta, rho_u, rho_beta):
        grad = np.zeros(self.num_drops * 4) # initialize gradient vector
        x,y,t = self.X, self.Y, self.time_arr
        rho_u = rho_u[::-1, :, :] # switch order of rho_u and rho_beta fields, since they go from T_final to T_initial
        rho_beta = rho_beta[::-1, :, :]
        for i in range(self.num_drops):
            # drop_number = i 
            # compute common terms in both Gradients 
            psi_times_theta = self.Theta_i(x,y,t,i) * self.psi_i(x,y,t,i)
            d_dt = self.d_ti_term(x, y, t, i) * psi_times_theta
            d_dtheta = self.d_psi_d_thetai(x,y,t,i) * psi_times_theta
            d_dx = self.d_psi_d_xi(x,y,t,i) * psi_times_theta
            d_dy = self.d_psi_d_yi(x,y,t,i) * psi_times_theta
            # compute (rho_u * gradQ) + (rho_beta * GradM) values
            W = - (self.k * u * rho_u +  self.delta * beta * rho_beta)
            t_term = W * d_dt
            theta_term = W * d_dtheta 
            x_term = W * d_dx 
            y_term = W * d_dy 
            # compute gradient by summing terms over time and space axes.
            grad[0 + (4*i)] = np.sum(x_term, axis=(0,1,2))      * self.dx * self.dy * self.dt
            grad[1 + (4*i)] = np.sum(y_term, axis=(0,1,2))      * self.dx * self.dy * self.dt
            grad[2 + (4*i)] = np.sum(theta_term, axis=(0,1,2))  * self.dx * self.dy * self.dt
            grad[3 + (4*i)] = np.sum(t_term, axis=(0,1,2))      * self.dx * self.dy * self.dt
        
        return grad
   
    def psi_i(self, x, y, t, drop_number):
        theta_i, t_i =  self.control_vars[drop_number, 2], self.control_vars[drop_number, 3]
        x_i, y_i = self.control_vars[drop_number, 0], self.control_vars[drop_number, 1]
        return spray_gaussian(x, y, t, x_i, y_i, t_i, theta_i, self.v, self.sigma_x, self.sigma_y)
    
    def Theta_i(self, x, y, t, drop_number):
        theta_i, t_i =  self.control_vars[drop_number, 2], self.control_vars[drop_number, 3]
        x_i, y_i = self.control_vars[drop_number, 0], self.control_vars[drop_number, 1]
        return time_control(t, t_i, self.T, h=10)

    def d_ti_term(self, x, y, t, drop_number, h=10):
        """
        return the term (partial Q / partial t_i) except the -ku factor., given i=drop number.
        """
        theta_i, t_i =  self.control_vars[drop_number, 2], self.control_vars[drop_number, 3]
        x_i, y_i = self.control_vars[drop_number, 0], self.control_vars[drop_number, 1]

        # first the (psi * partial Theta / partial t_i term)
        term = h * (1/(1+np.exp(-h*(t-t_i-self.T))) - 1/(1+np.exp(h*(t-t_i)))) -self.v * ((((x-x_i)-self.v*np.cos(theta_i)*(t-t_i))*np.cos(theta_i) / self.sigma_x**2) + (((y-y_i)-self.v*np.sin(theta_i)*(t-t_i))*np.sin(theta_i) / self.sigma_y**2))
        
        return term
    
    def d_psi_d_thetai(self, x,y,t, drop_number):
        """
        return a lambda function for the term (partial psi / partial theta_i), given i=drop number.
        """
        theta_i =  self.control_vars[drop_number, 2]
        x_i, y_i, t_i = self.control_vars[drop_number, 0], self.control_vars[drop_number, 1], self.control_vars[drop_number, 3]

        return self.v*(t-t_i)*(((y-y_i)-self.v*np.sin(theta_i)*(t-t_i))*np.cos(theta_i)/self.sigma_y**2 - ((x-x_i)-self.v*np.cos(theta_i)*(t-t_i))*np.sin(theta_i)/self.sigma_x**2)
    
    def d_psi_d_xi(self, x,y,t, drop_number):
        theta_i =  self.control_vars[drop_number, 2]
        x_i, y_i, t_i = self.control_vars[drop_number, 0], self.control_vars[drop_number, 1], self.control_vars[drop_number, 3]

        return ((x-x_i)-self.v*np.cos(theta_i)*(t-t_i)) / self.sigma_x**2

    
    def d_psi_d_yi(self, x,y,t, drop_number):
        theta_i =  self.control_vars[drop_number, 2]
        x_i, y_i, t_i = self.control_vars[drop_number, 0], self.control_vars[drop_number, 1], self.control_vars[drop_number, 3]

        return ((y-y_i)-self.v*np.sin(theta_i)*(t-t_i)) / self.sigma_y**2


