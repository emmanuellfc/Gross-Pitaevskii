# Bose-Einstein Condensate Simulation

This code implements a simulation framework for a Bose-Einstein Condensate (BEC), combining the Gross-Pitaevskii Equation (GPE) with a Metropolis-like stochastic algorithm.

The main source files for running simulations are located in the `/src` directory. A brief summary of the purposes of each script is included below:
GPESolver.py: Functions used in solving the GPE equation
metropolis.py: Functions used in running the stochastic algorithm
help_functions.py: Accessory functions, including those needed for animations

The scripts `Harmonic_Trap.py` and `Metropolis_Stochastic.py`, also within `/src`, provide an example of how to utilize the GPE and stochastic codes respectively. 

The stochastic algorithm implementation is based on Pjotrs Grišins and Igor E. Mazets' paper: https://www.sciencedirect.com/science/article/pii/S0010465514001015?ref=pdf_download&fr=RR-2&rr=93b18ea3f8b58f84
