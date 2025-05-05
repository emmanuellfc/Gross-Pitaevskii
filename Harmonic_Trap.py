# Import main solver and helper functions
from src.GPESolver import GPESolver
from src.helper_functions import *
import matplotlib.pyplot as plt
import numpy as np

# Parameters
N_POINTS = 512       # Number of spatial grid points
X_MIN = -20.0        # Spatial grid min
X_MAX = 20.0         # Spatial grid max
DX = (X_MAX - X_MIN) / (N_POINTS - 1) # Calculate dx based on N and bounds

TIME_STEPS = 500     # Number of time steps
TOTAL_TIME = 10.0    # Total simulation time
DT = TOTAL_TIME / TIME_STEPS # Calculate dt

G_NONLINEAR = -1.0    # Interaction strength (g < 0 attractive, g > 0 repulsive)
HBAR = 1.0
MASS = 1.0

# --- Setup Solver ---
solver = GPESolver(dt=DT, dx=DX, n=N_POINTS, steps=TIME_STEPS, g=G_NONLINEAR, hbar=HBAR, mass=MASS)

# --- Create Grid ---
x_grid = solver.create_grid(X_MIN, X_MAX)

# --- Set Initial Condition ---
# Moving Gaussian Packet
X0 = -5.0  # Initial position
SIGMA0 = 1.0 # Initial width
K0 = 2.0   # Initial momentum
solver.gaussian_wave_packet(x0=X0, sigma=SIGMA0, k0=K0)

# --- Set External Potential ---
# Harmonic trap
OMEGA_TRAP = 1.0
solver.set_potential(solver.potential_sho(omega=OMEGA_TRAP))

# --- Solve ---
print("Starting GPE simulation...")
psi_evolution = solver.solve_gpe_crank_nicolson()
print(f"Simulation finished. Psi evolution shape: {psi_evolution.shape}")

# Make animations
fname = "Test_GPE_Solver"
fname_real = "Test_GPE_Solver_Real"
fname_imag = "Test_GPE_Solver_Imag"
sq_label= r"$|\psi|^2$"
re_label= r"$Re(\psi)$"
im_label= r"$Im(\psi)$"
squared_series = generate_time_series_psi_squared(solver, TIME_STEPS)
real_series = generate_time_series_psi_real(solver, TIME_STEPS)
imag_series = generate_time_series_psi_imag(solver, TIME_STEPS)
animate_psi(squared_series, TIME_STEPS, fname, sq_label, "magenta")
animate_psi(real_series, TIME_STEPS, fname_real, re_label, "blue")
animate_psi(imag_series, TIME_STEPS, fname_imag, im_label, "red")
