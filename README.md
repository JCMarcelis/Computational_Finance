# Computational Finance
Computational Finance Course at University of Amsterdam 2025

This repository will contain the three lab assignments of the course. Currently, the root directory contains the code for the second lab assignment. The first assignment is in the folder named `Assignment1`.

In order for the code to run smoothly, pull this repository and run `pip install -r requirements.txt` to install all Python packages used.

## Assignment 1 (separate folder): Financial data science, option pricing, and hedging
The main python notebook contains all the plots created by the code from the src folder.
* [volatility.py](src/volatility.py): contains all the code for the volatility part.
* [hedging_sim.py](src/hedging_sim.py): contains all the code for the hedging simulation.

## Assignment 2 (root directory): Asian options and Temperature derivatives
The main notebook contains all plots and data created by the code in the src folder.
* [Euler_vs_Milstein.py](src/Euler_vs_Milstein.py): contains all the code for comparing Euler and Milstein discretization for the Heston model
* [control_variate_MC.py](src/control_variate_MC.py): contains all the code for the plain and control variate Monte Carlo simluations