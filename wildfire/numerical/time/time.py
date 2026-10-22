"""Implements the following integration methods:
    - Euler method
    - Runge-Kutta of fourth order method (RK4)

Also includes `solve_ivp` from `scipy.integrate`. 

Details in: 
    - Sauer, T. (2018). Numerical Analysis. Pearson. https://www.pearson.com/us/higher-education/program/Sauer-Numerical-Analysis-3rd-Edition/PGM1735484.html
    - Scipy. solve_ivp. https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html
"""
import numpy as np
from scipy.integrate import solve_ivp 

def Euler(t, F, y0, last=True, vdata=False):
    """Euler method implementation.

    Parameters
    ----------
    t : array_like
        Time discrete variable.
    F : function
        RHS function of ODE.
    y0 : array_like
        Initial condition
    last : bool, optional
        Return and keep only last approximation, by default True.
    vdata: bool, optional
        Vector field is np.ndarray, by default False.
        If vdata is True, method uses time iteration as index for vector field data.

    Returns
    -------
    array_like
        Aproximated solution of ODE.

    """
    # Get number time of nodes
    Nt = t.shape[0]

    # Get :math:`\Delta t`
    dt = t[1] - t[0] 

    # Only keep and return last approximation
    if last: 
        y = y0

        if vdata:
            for n in range(Nt - 1):
                yc = np.copy(y)
                y = yc + dt * F(n, yc)
        else:
            for n in range(Nt - 1):
                yc = np.copy(y)
                y = yc + dt * F(t[n], yc)

    # Keep and return array with all approximations
    else:
        y = np.zeros((Nt, y0.shape[0])) 
        y[0] = y0 # Initial condition
        
        if vdata:
            for n in range(Nt - 1):
                y[n+1] = y[n] + dt * F(n, y[n])
        else:
            for n in range(Nt - 1):
                y[n+1] = y[n] + dt * F(t[n], y[n])
        
    return y

def Euler_adjoint(t, F, y0, field_data, last=True, vdata=False):
    """Euler method for the adjoint system with stored forward fields."""
    Nt = t.shape[0]
    dt = t[1] - t[0]

    if last:
        y = y0

        if vdata:
            for n in range(Nt - 1):
                field_vec = field_data[n, :]
                y = y + dt * F(n, y, field_vec)
        else:
            for n in range(Nt - 1):
                field_vec = field_data[n, :]
                y = y + dt * F(t[n], y, field_vec)

    else:
        y = np.zeros((Nt, y0.shape[0]))
        y[0] = y0

        if vdata:
            for n in range(Nt - 1):
                field_vec = field_data[n + 1, :]          ## MADE FIX HERE FOR USING PREVIOUS TIME JACOBIAN, CORRECTING THE
                y[n + 1] = y[n] + dt * F(t[n + 1], y[n], field_vec)     # OFF BY ONE ERROR  
        else:
            for n in range(Nt - 1):
                field_vec = field_data[n + 1, :]          # FIX HERE AS WELL.
                y[n + 1] = y[n] + dt * F(t[n + 1], y[n], field_vec)

    return y

def RK4_adjoint(t, F_adj, F_fwd, y0, field_data, gradient_object):
    """Runge-Kutta of fourth order implementation.

    Currently using constant (not intermediate) values of u and beta fields in RHS computation.

    Parameters
    ----------
    t : array_like
        Time discrete variable.
    F_adj : function
        RHS function of adjoint equations.
    F_fwd : function
        RHS function of forward equations, required for recomputing intermediate forward fields.
    y0 : array_like
        Initial condition
    field_data: array_like
        values of u and beta fields, vectorized, ordered form t_max to t_min in Nt steps.
    
    v_data and last functions are input params to stay consistent, but function is only implemented
    for last=False and v_data=False.

    Returns
    -------
    array_like
        Aproximated solution of "adjoint" ODE.

    """
    # Get number time of nodes
    Nt = t.shape[0]

    # Get  positive Delta t
    dt = np.abs(t[1] - t[0])
    
    # Keep and return all approximations
    y = np.zeros((Nt, y0.shape[0])) # array for storing backward fields.
    y[0] = y0 # Initial condition

    # (dF/dy)^T w from RHS_adjoint
    JT = lambda w, Y, tt: -F_adj(tt, w, Y)      
    grad = np.zeros(4 * gradient_object.num_drops)
    Nx, Ny = gradient_object.X.shape[0], gradient_object.Y.shape[1]
    n_u = Nx * Ny


    for n in range(Nt - 1):
        # Redo forward RK4 solve to get the intermediate fields, Y2,Y3,Y4.
        t0 = t[n + 1]    # forward step goes t[n+1] -> t[n] in given time convention.
        Y1 = field_data[n + 1]
        k1 = F_fwd(t0, Y1)
        Y2 = Y1 + 0.5 * dt * k1
        k2 = F_fwd(t0 + 0.5 * dt, Y2)
        Y3 = Y1 + 0.5 * dt * k2
        k3 = F_fwd(t0 + 0.5 * dt, Y3)
        Y4 = Y1 + dt * k3

        # backward RK4 solve
        # exact discrete adjoint of that step (stages in reverse)
        lam = y[n]                    # current time-step
        w4 = lam / 6;                 th4 = dt * JT(w4, Y4, t0 + dt)
        w3 = lam / 3 + th4;           th3 = dt * JT(w3, Y3, t0 + 0.5 * dt)
        w2 = lam / 3 + 0.5 * th3;     th2 = dt * JT(w2, Y2, t0 + 0.5 * dt)
        w1 = lam / 6 + 0.5 * th2;     th1 = dt * JT(w1, Y1, t0)
        y[n + 1] = lam + th1 + th2 + th3 + th4

        # --- gradient: dJ/dp += h * sum_i w_i^T dF/dp(Y_i, t_i) ---
        for Yi, wi, ti in ((Y1, w1, t0), (Y2, w2, t0 + 0.5*dt),
                        (Y3, w3, t0 + 0.5*dt), (Y4, w4, t0 + dt)):
            U_i  = Yi[:n_u].reshape((Ny, Nx), order='F')
            B_i  = Yi[n_u:].reshape((Ny, Nx), order='F')
            wu_i = wi[:n_u].reshape((Ny, Nx), order='F')
            wb_i = wi[n_u:].reshape((Ny, Nx), order='F')
            grad += dt * gradient_object.stage_gradient(U_i, B_i, wu_i, wb_i, ti)
    return y, grad

def RK4(t, F, y0, last=True, vdata=False):
    """Runge-Kutta of fourth order implementation.

    Parameters
    ----------
    t : array_like
        Time discrete variable.
    F : function
        RHS function of ODE.
    y0 : array_like
        Initial condition
    last : bool, optional
        Return and keep only last approximation, by default True.
    vdata: bool, option
        Vector field is np.ndarray, by default False.
        If vdata is True, method uses time iteration as index for vector field data.

    Returns
    -------
    array_like
        Aproximated solution of ODE.

    """
    # Get number time of nodes
    Nt = t.shape[0]

    # Get \Delta t
    dt = t[1] - t[0] 
    
    if last: # Only keep and return last approximation
        # Initial condition
        y = y0
        
        if vdata:
            for n in range(Nt - 1):
                yc = np.copy(y)
                k1 = F(n, yc)
                k2 = F(n, yc + 0.5 * dt * k1)
                k3 = F(n, yc + 0.5 * dt * k2)
                k4 = F(n, yc + dt * k3)
                y = yc + (1/6) * dt * (k1 + 2 * k2 + 2 * k3 + k4)
        else:
            for n in range(Nt - 1):
                yc = np.copy(y)
                k1 = F(t[n], yc)
                k2 = F(t[n] + 0.5 * dt, yc + 0.5 * dt * k1)
                k3 = F(t[n] + 0.5 * dt, yc + 0.5 * dt * k2)
                k4 = F(t[n] + dt, yc + dt * k3)

                y = yc + (1/6) * dt * (k1 + 2 * k2 + 2 * k3 + k4)

    else: # Keep and return all approximations
        # Array for approximations
        y = np.zeros((Nt, y0.shape[0]))
        y[0] = y0 # Initial condition

        if vdata:
            for n in range(Nt - 1):
                k1 = F(n, y[n])
                k2 = F(n, y[n] + 0.5 * dt * k1)
                k3 = F(n, y[n] + 0.5 * dt * k2)
                k4 = F(n, y[n] + dt * k3)
                y[n + 1] = y[n] + (1/6) * dt * (k1 + 2 * k2 + 2 * k3 + k4)
        else: #### ONLY BRANCH THAT SHOULD BE IMPLEMENTED ( WIND PASSED AS FUNCTION AND LAST=FALSE)
            for n in range(Nt - 1):
                k1 = F(t[n], y[n])
                k2 = F(t[n] + 0.5 * dt, y[n] + 0.5 * dt * k1)
                k3 = F(t[n] + 0.5 * dt, y[n] + 0.5 * dt * k2)
                k4 = F(t[n] + dt, y[n] + dt * k3)
                y[n + 1] = y[n] + (1/6) * dt * (k1 + 2 * k2 + 2 * k3 + k4)
        
    return y

def IVP(t, F, y0, last=True, method='RK45'):
    """Solve IVP wrapper.

    Parameters
    ----------
    t : array_like
        Time discrete variable.
    F : function
        RHS function of ODE.
    y0 : array_like
        Initial condition
    last : bool, optional
        Return and keep only last approximation, by default True.
    method : strin, optional
        Numerical method to solve IVP, default RK45. 
        Also includes 'RK23', 'DOP853', 'Radau', 'BDF', 'LSODA'. More details in `scipy.integrate.solve_ivp` documentation.

    Returns
    -------
    array_like
        Aproximated solution of ODE.

    """
    t_min = t[0]
    t_max = t[-1]
    t_eval = np.array([t_max]) if last else t
    sol = solve_ivp(F, (t_min, t_max), y0, t_eval=t_eval, method=method)  
    y = sol.y if last else sol.y.T  
    return y