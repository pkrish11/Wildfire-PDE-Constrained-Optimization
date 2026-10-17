"""
Third-order upwind-biased advection for the wildfire solver (drop-in for wildfire/numerical/space/upwind.py).

Why: on the 252x252 mesh the cell Peclet number is |v|h/kappa ~ 3.9. First-order upwind adds
numerical diffusion |v|h/2 ~ 2*kappa; this stencil's leading error is dispersive/dissipative at O(h^3)
(dissipation ~ |v| h^3/12 * u_xxxx), so the front is not smeared, and it is linear, so its adjoint
is just the transposed (mirrored) stencil.

Usage in fd.py
--------------
    from .upwind3 import upwind3_derivative

  RHS (forward):            # note: correct axes (axis=1 is x, axis=0 is y)
    Ux = upwind3_derivative(U, self.dx, V1, axis=1)
    Uy = upwind3_derivative(U, self.dy, V2, axis=0)
    convection = Ux * V1 + Uy * V2

  RHS_adjoint (transpose of the forward operator; exact in the interior for spatially constant wind,
  because D_pos^T = -D_neg):
    convection = (V1 * upwind3_derivative(r_u, self.dx, -V1, axis=1)
                + V2 * upwind3_derivative(r_u, self.dy, -V2, axis=0))
    Uf = -(diffusion + convection + r_u*df_du + r_beta*dg_du + r_u*dq_du)   # (+Q, +M signs fixed)

  Time stepping: forward Euler is unstable with this stencil at Nt=225. Use time_method='RK4'
  (already in the repo) or SSP-RK3, with Nt >= 150 for the notebook case.
"""
import numpy as np


def upwind3_derivative(U, h, vel, axis):
    """Third-order upwind-biased first derivative of U along `axis`.

    Arrays are (Ny, Nx) as produced by np.meshgrid: axis=1 is x, axis=0 is y.
    `vel` is the advecting velocity component along that axis (scalar or array);
    its sign at each node selects the stencil:
      vel >= 0:  ( u[i-2] - 6 u[i-1] + 3 u[i] + 2 u[i+1]) / (6h)
      vel <  0:  (-2 u[i-1] - 3 u[i] + 6 u[i+1] - u[i+2]) / (6h)
    Nodes next to the boundary fall back to first-order upwind; the boundary nodes
    themselves are overwritten by the Dirichlet condition anyway.
    """
    W = np.moveaxis(np.asarray(U, float), axis, -1)
    V = np.moveaxis(np.broadcast_to(vel, np.shape(U)), axis, -1)
    D = np.zeros_like(W)
    um2, um1, u0, up1, up2 = W[..., :-4], W[..., 1:-3], W[..., 2:-2], W[..., 3:-1], W[..., 4:]
    dpos = (um2 - 6.0 * um1 + 3.0 * u0 + 2.0 * up1) / (6.0 * h)
    dneg = (-2.0 * um1 - 3.0 * u0 + 6.0 * up1 - up2) / (6.0 * h)
    D[..., 2:-2] = np.where(V[..., 2:-2] >= 0, dpos, dneg)
    for i in (1, W.shape[-1] - 2):
        D[..., i] = np.where(V[..., i] >= 0, W[..., i] - W[..., i - 1], W[..., i + 1] - W[..., i]) / h
    return np.moveaxis(D, -1, axis)


def grad_u_upwind3(U, dx, dy, V1, V2):
    """Same call shape as grad_u_upwind(U, dx, dy, wind_x, wind_y), but with the correct axes.
    V1, V2 can be the velocities or just their signs."""
    return upwind3_derivative(U, dx, V1, axis=1), upwind3_derivative(U, dy, V2, axis=0)
