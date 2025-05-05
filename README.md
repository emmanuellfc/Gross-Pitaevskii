# Bose-Einstein Condensate Simulation

This code implements a simulation framework for a Bose-Einstein Condensate (BEC), combining the Gross-Pitaevskii Equation (GPE) with a Metropolis-like stochastic algorithm.

The main source files for running simulations are located in the `/src` directory. A brief summary of the purposes of each script is included below: <br/><br/>
`GPESolver.py`: Functions used in solving the GPE equation <br/>
`metropolis.py`: Functions used in running the stochastic algorithm <br/>
`help_functions.py`: Accessory functions, including those needed for animations

The scripts `Harmonic_Trap.py` and `Metropolis_Stochastic.py`, are also in the main folder and they provide an example of how to utilize the GPE and stochastic codes respectively.

The stochastic algorithm implementation is based on Pjotrs Grišins and Igor E. Mazets' paper: https://www.sciencedirect.com/science/article/pii/S0010465514001015?ref=pdf_download&fr=RR-2&rr=93b18ea3f8b58f84
