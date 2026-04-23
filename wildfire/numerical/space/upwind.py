import numpy as np

# First derivative with Finite Differences (non matrix), Upwind Scheme

def grad_u_upwind(U, dx, dy, wind_x, wind_y):
    """ 
    Parameters
    ----------
    wind_velocity : (Nx,Ny,2) array denoting direction of velocity at each mesh point in x and y components.
    """
    Dx = upwind_x(U, dx, wind_x)
    Dy = upwind_y(U, dy, wind_y)

    return Dx, Dy


def upwind_x(U, h, wind_velocity_x):
    """
    Compute first derivative using Finite Difference Matrix (upwind scheme)
    with O(h) accuracy.
    
    Parameters
    ----------
    U : (Nx,Ny) ndarray 
        values for which to compute (partial u / partial x)
    h : float
        Step size.
    wind_velocity_x : (N,N) ndarray 
        denotes wind_velocity in x-direction.
        must be positive at the points where wind velocity is > 0 (so backward difference can be used)
        and     negative at the points where wind velocity is < 0 (so forward difference can be used)
            
    Returns
    -------
    D1 : (N, N) ndarray
        Finite difference dense matrix; or
    """

    D1 = np.zeros_like(U)

    # shifted matrices
    forward = (U[2:,:] - U[1:-1, :]) / h
    backward = (U[1:-1, :] - U[0:-2,:] ) / h

    D1[1:-1, :] = np.where(wind_velocity_x[1:-1, :] > 0, backward, forward)

    # forward and backward differences necessarily at boundary.
    D1[0, :] = (U[1,:] - U[0,:]) / h
    D1[-1,:] = (U[-1,:] - U[-2,:]) / h

    return D1

def upwind_y(U, h, wind_velocity_y):
    """
    Compute first derivative using Finite Difference Matrix (upwind scheme)
    with O(h) accuracy.
    """

    D1 = np.zeros_like(U)

    # shifted matrices
    forward = (U[:,2:] - U[:, 1:-1]) / h
    backward = (U[:, 1:-1] - U[:,0:-2] ) / h

    D1[:, 1:-1] = np.where(wind_velocity_y[:,1:-1] > 0, backward, forward)

    # forward and backward differences necessarily at boundary.
    D1[:,0] = (U[:,1] - U[:,0]) / h
    D1[:,-1] = (U[:,-1] - U[:,-2]) / h

    return D1