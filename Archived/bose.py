# Load modules
import numpy as np

from src.QMSolver import QMSolver
from src.helper_functions import *

# Solving the SHO
# Declare Solver Parameters
dt = 0.1
dx = 0.1
t_steps    = 200
resolution = 200
# Declare Gaussian Wave Packet Parameters
sigma = 1.0  # Ground state width
x0 = 0.0      # Initial center position (equilibrium)
k0 = 0.0      # Initial wave number (ground state)
sim = QMSolver(dt=dt, dx=dx, n=resolution, steps=t_steps)
sim.create_grid(-5,5)
sim.gaussian_wave_packet(x0, sigma, k0)
sim.sho_potential()
sim.create_hamiltonian_fd()
sim_solution = sim.solve_finite_difference()

# Animation
time_series_psi_sq = generate_time_series_psi_squared(sim, t_steps)
f_name_psi_sq = 'Harmonic Potential (CN Scheme)'
label_sq = r"$|\psi|^2$"
animate_psi(time_series_psi_sq, t_steps, filename=f_name_psi_sq, label=label_sq, color = "magenta")

time_series_psi_real =generate_time_series_psi_real(sim, t_steps)
f_name_psi_re = 'Harmonic Potential Re Part (CD Scheme)'
label_re = r"Re($\psi$)"
animate_psi(time_series_psi_real, t_steps, filename=f_name_psi_re, label=label_re, color = "red")

time_series_psi_imag =generate_time_series_psi_imag(sim, t_steps)
f_name_psi_im = 'Harmonic Potential Im Part (CN Scheme)'
label_im = r"Im($\psi$)"
animate_psi(time_series_psi_imag, t_steps, filename=f_name_psi_im, label=label_im, color = "blue")
