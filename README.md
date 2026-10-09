# Wildfire-PDE-Constrained-Optimization
Repository to keep track of code for research work with PDE constrained optimization for wildfires.

### Structutre
* `wildfire` directory is the main source code for wildfire simulations.
* `gradient` class computes gradients.
* `optimizer` class can be used to access objective function and gradient functionalities, will take input as normalized variables and return normalized gradients.
* `optimization_testing2.ipynb` is the most recent notebook with experiments. 

### Notes on physical / model setup
* The second cell in the jupyter notebook describes the complete setting for the model - time and space domains and discretizations, fuel and fire initial conditions (keep constant), asset map (can be changed by asset_weight variable), wind (use V or changing_wind, and gamma variable for wind speed. Note that if wind-speed is changed, the total time of simulation T and time-discretization Nt might have to be adjusted), and control_params (sigma_x, sigma_y, v, k, delta can all be varied to see effect on fire/optimizer).
* The third cell gives functionality for tracking optimizer behavior, leave as is.
* Fourth cell runs the optimization experiment. Normalization of initial guess (x0) is done in the cell, and additionally auxillary functions are obtained (to allow tracking, plus scale objective function and gradients by the objective function value of initial guess). The cell also outlines the constraint for water drops not to overlap. Vary the opt.minimize() function as necessary, SQSLP method is the only one seen to converge until now.
* Run the last cell to see the actual simulation of the optimal point found (Make sure wind-setting matches the wind-input to optimizer object).