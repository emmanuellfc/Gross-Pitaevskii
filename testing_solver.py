from src.GPESolver import GPESolver
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
# Example 1: Ground state Gaussian
# solver.ground_state_gaussian(sigma=1.0)
# Example 2: Moving Gaussian Packet
X0 = -5.0  # Initial position
SIGMA0 = 1.0 # Initial width
K0 = 2.0   # Initial momentum (velocity = hbar*k0/m)
solver.gaussian_wave_packet(x0=X0, sigma=SIGMA0, k0=K0)

# --- Set External Potential ---
# Example 1: Zero potential
solver.set_potential(solver.potential_zero(x_grid))
# Example 2: Harmonic trap
# OMEGA_TRAP = 1.0
# solver.set_potential(solver.potential_sho(omega=OMEGA_TRAP))
# Example 3: Barrier
# V0_BARRIER = 5.0
# BARRIER_LEFT = -0.5
# BARRIER_RIGHT = 0.5
# solver.set_potential(solver.potential_barrier(v0=V0_BARRIER, x_left=BARRIER_LEFT, x_right=BARRIER_RIGHT))

# --- Solve ---
print("Starting GPE simulation...")
psi_evolution = solver.solve_gpe_crank_nicolson()
print(f"Simulation finished. Psi evolution shape: {psi_evolution.shape}") # (steps+1, n)
