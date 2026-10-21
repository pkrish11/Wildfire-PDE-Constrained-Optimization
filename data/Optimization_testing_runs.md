## Optimization Testing Runs - Record Log

### Run 1
* Sim Time - [0,10], no constraints on spray times. (Nt=150)
* Normalized variables and scaled gradients+objective function.
* Initial Guess : x0 = [-25, -25, (1/4)*np.pi, 0, 0, 0, (3/4)*np.pi, 3] 
* Optimizer Method : Trust Constraint. 
* Output : Does not converge, gets to max iterations (100)
* Final value : [-2.18969226e+01 -1.91017462e+01  6.13863104e-01  1.46738846e-04 -5.62333099e+00 -1.30615703e+01  2.39173766e+00  2.14551118e+00]
* Final Objective Function value : 0.117 * J(x0)
* Gradient at final point : [ 2.09466780e+00 -1.96851808e+00 -8.38802275e+00  3.14569521e+01 3.13345675e-02 -3.02524393e-02  9.35341797e-02  1.82184831e+00], with 2 norm of 32.
* Gradient of Lagrangian : [ 4.087e+01 -3.964e+01 -4.929e-01  9.039e-09  6.904e-01 -6.424e-01  5.554e-02  7.478e-01], with 2 norm of 0.3 and inf norm of 40.8 . 
* 141 Function + Jacobian Evals, over 1909 seconds -> average of 13.5 seconds for 1 forward + backward solve.
* Result saved as screenshot titled "Successful Optimization Run 1".

Notes : getting to sufficiently good objective value with normalization + trust region, maybe just have to figure out how to make convergence possible.
For the time variable, first drop is still made as soon as possible, t=0, and the second drop is made at 2.14 (seperated by less than 3 seconds, the duration of a single drop). Position and angles do change somewhat.
Gradient is still large, with 2 norm value being 32, specifically the components in the time direction are the largest.



