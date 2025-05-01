from src.GPESolver import GPESolver
import numpy as np

N_POINTS = 500       # Number of spatial grid points
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
solver.set_potential(solver.potential_zero())
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

# --- Analysis / Visualization (Example using matplotlib) ---
try:
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 8))

    plt.subplot(2, 1, 1)
    plt.plot(x_grid, np.abs(psi_evolution[0])**2, label=f't = 0')
    plt.plot(x_grid, np.abs(psi_evolution[TIME_STEPS // 2])**2, label=f't = {TOTAL_TIME / 2:.1f}')
    plt.plot(x_grid, np.abs(psi_evolution[-1])**2, label=f't = {TOTAL_TIME:.1f}')
    # Plot potential for reference
    if solver.potential_ext is not None:
            # Scale potential for visibility if needed
            pot_scale = np.max(np.abs(psi_evolution[0])**2) / (np.max(np.abs(solver.potential_ext)) + 1e-9) if np.max(np.abs(solver.potential_ext)) > 1e-9 else 1.0
            pot_scale = min(pot_scale, 1.0) # Don't make potential huge
            plt.plot(x_grid, solver.potential_ext.real * pot_scale, 'k--', label=f'V(x) (scaled)', alpha=0.5)

    plt.title(f'GPE Simulation (g={G_NONLINEAR}): Probability Density $|\psi(x,t)|^2$')
    plt.xlabel('Position x')
    plt.ylabel('$|\psi|^2$')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 1, 2)
    density_map = plt.imshow(np.abs(psi_evolution)**2, aspect='auto', origin='lower',
                            extent=[X_MIN, X_MAX, 0, TOTAL_TIME],
                            cmap='viridis') # Use 'viridis' or other suitable colormap
    plt.colorbar(density_map, label='$|\psi(x,t)|^2$')
    plt.xlabel('Position x')
    plt.ylabel('Time t')
    plt.title('Density Evolution')

    plt.tight_layout()
    plt.show()

except ImportError:
    print("Matplotlib not found. Skipping visualization.")
except Exception as e:
    print(f"An error occurred during visualization: {e}")
