"""Functions utilities.

Some functions for implementation.

"""
import numpy as np

# Generic #
def G(x, y, s):
    """Gaussian kernel.
    
    .. math::
        G(x, y) = \exp(-(x^2 + y^2) / s)

    Parameters
    ----------
    x : float or array_like
        x value.
    y : float or array_like
        y value.
    s : float
        Gaussian shape parameter.

    Returns
    -------
    float or array_like
        Gaussian function
    """
    return np.exp(-(x ** 2 + y ** 2) / s)

# PDE FUNCTIONS #
def K(u, kap, eps):
    """Compute diffusion function 

    .. math::
        K(u) = \kappa \, (1 + \varepsilon u)^3 + 1

    Parameters
    ----------
    u : array_like
        Temperature variable.
    kap : float
        Diffusion parameter.
    eps : float
        Inverse of activation energy.

    Returns
    -------
    array_like
        Evaluation of K function.
    """
    return kap * (1 + eps * u) ** 3 + 1

def Ku(u, kap, eps):
    """Derivative of K with respect to u.

    .. math:
        \dfrac{\partial K}{\partial u} = K_{u} = 3\,\varepsilon \kappa\, (1 + \varepsilon\, u)^2
    Parameters
    ----------
    u : array_like
        Temperature variable.
    kap : float
        Diffusion parameter.
    eps : float
        Inverse of activation energy.

    Returns
    -------
    array_like
        Ku evaluation.
    """
    return 3 * eps * kap * (1 + eps * u) ** 2

def f(u, b, eps, alp, s):
    """Temperature-fuel reaction function.

    Parameters
    ----------
    u : array_like
        Temperature value.
    b : array_like
        Fuel value.
    eps : float
        Inverse of activation energy parameter.
    alp : float
        Natural convection parameter.
    s : function or lambda
        Step function.

    Returns
    -------
    array_like
        Reaction function.
    """
    return s(u) * b * np.exp(u / (1 + eps * u)) - alp * u

def g(u, b, eps, q, s):
    """RHS of fuel PDE.

    Parameters
    ----------
    u : array_like
        Temperature value
    b : array_like
        Fuel value.
    eps : float
        Inverse of activation energy parameter.
    q : float
        Reaction heat parameter.
    s : function or lambda
        Step function.

    Returns
    -------
    array_like
        Fuel RHS PDE.
    """
    return -s(u) * (eps / q) * b * np.exp(u / (1 + eps * u))

def H(u, upc):
    """2D heaviside funcion

    Parameters
    ----------
    u : array_like
        Temperature value
    upc : float
        Phase change threshold.

    Returns
    -------
    array_like
        Heaviside function evaluation.
    """
    S = np.zeros_like(u)
    S[u >= upc] = 1.0
    return S

def sigmoid(u, k=.5):
    """Sigmoid function.

    Parameters
    ----------
    u : array_like
        Temperature value.
    k : float, optional
        Slope constant factor, by default .5

    Returns
    -------
    array_like
        Sigmoid evaluation.
    """	
    return 1 / (1 + np.exp(-k * scale(u))) #0.5 * (1 + np.tanh(k * self.scale(u)))

def sigmoid_u(u, u_pc, k=10):
    """ New sigmoid function for u, to approximate the step function H"""
    return 1 /  (1 + np.exp(-k * (u-u_pc)))

def scale(u, a=-10, b=10):
    """Scale function.

    Parameters
    ----------
    u : array_like
        Temperature value.
    a : int, optional
        Minimum value, by default -10
    b : int, optional
        Maximum value, by default 10

    Returns
    -------
    array_like
        Scaled value of u.
    """
    return (b - a) * (u - np.min(u)) / (np.max(u) - np.min(u)) + a


## Additional functions for control parameter
def time_sigmoid(t, t_i, h):
    """
    Sigmoid function to turn spray parameter on (if h >0) and off (if h<0).
    """
    return 1/(1 + np.exp(-h*(t-t_i)))

def time_control(t, t_i, T, h=10):
    """
    function that does full on-off feature. T is model parameter-> time duration of water drop.
    """
    return time_sigmoid(t, t_i, h) * time_sigmoid(t, (t_i+T), -h)

def spray_gaussian(x, y, t, x_i, y_i, t_i, theta_i, v, sigma_x, sigma_y):
    """
    moving gaussian, centered at (x_i,y_i) at t=t_i and moving in direction theta_i from positive x-axis
    with velocity v. 
    """
    exp_1 = -((x-x_i) - v*np.cos(theta_i)*(t-t_i))**2 / (2*sigma_x**2)
    exp_2 = -((y-y_i) - v*np.sin(theta_i)*(t-t_i))**2 / (2*sigma_y**2)
    return np.exp(exp_1 + exp_2)

# Helper functions to get control paramaeters (Q and M) as functions of space and time.

def sum_spray_terms(x, y, t, control_params):
    v = control_params["v"]
    sigma_x, sigma_y = control_params["sigma_x"], control_params["sigma_y"]
    T = control_params["T"]

    control_vars = control_params["control_vars"]
    N = np.shape(control_vars)[0] # number of water drops to be made.

    sum = 0
    for i in range(N):
        x_i, y_i = control_vars[i, 0], control_vars[i, 1]
        theta_i = control_vars[i, 2]
        t_i = control_vars[i, 3]
        # add time control * moving gaussian to total term. 
        sum += time_control(t, t_i, T) * spray_gaussian(x, y, t, x_i, y_i, t_i, theta_i, v, sigma_x, sigma_y)
    
    return sum

def Q_func(x, y, t, control_params):

    k = control_params["k"]
    q = -k * sum_spray_terms(x, y, t, control_params) 

    return q

def M_func(x, y, t, control_params):

    delta = control_params["delta"]
    m = - delta * sum_spray_terms(x, y, t, control_params)

    return m


