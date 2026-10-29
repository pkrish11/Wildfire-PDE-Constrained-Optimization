import numpy as np

# terms to enforce water/fuel control with saturation (limited total capacity)
# saturation scales (tuned to optimizer (A2_optimizer values)
u_s, b_s = 1.5, 0.3

# new Q term is Q = - kappa * psi * theta * u_s * tanh(u/u_s). self.Q_func from utils imports everything except u_s * tanh(u/u_s) part.
q  = lambda u: u_s * np.tanh(u / u_s)      

# for adjoint, dq/du = self.Q_func / cosh(u/u_s)**2
dq = lambda u: 1 / np.cosh(u / u_s)**2

# new M term is M = - delta * psi * theta * b_s * tanh(b/b_s). self.M_func from utils imports everything except b_s * tanh(b/b_s) part.
m  = lambda b: b_s * np.tanh(b / b_s)

# for adjoint, dm/dbeta = self.M_func / cosh(b/b_s)**2
dm = lambda b: 1 / np.cosh(b / b_s)**2
