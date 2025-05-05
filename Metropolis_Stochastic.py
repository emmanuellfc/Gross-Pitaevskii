from src.metropolis import *
from src.helper_functions import *

#Define Constants
omega=1
hbar=1
mass=1
g_param=-1
k_boltz=1
mu_chem_potential=1
#N_iterations=10
c1_val=1
c2_val=1
c3_val=.001
v_val=1
phase=1
u=1

#Construct grid
x_start=-5
x_end=5
num_steps=1000
grid=np.linspace(x_start,x_end, num_steps)

#Construct potential and initial wave function
psi_0=(1/np.pi**.25)*np.exp(-.5*grid**2)
V=grid**2/2

#Define run parameters
iterations=1000
temperature=10

#run stochastic algorithm for two different temperatures
psi_array, psi_sq_array, entropy_array=loop_stochastic(hbar, mass, g_param, k_boltz, mu_chem_potential, c1_val, c2_val, c3_val, v_val, phase, u, x_end, x_start, num_steps, grid, psi_0, V, iterations, temperature)
psi_array2, psi_sq_array2, entropy_array2=loop_stochastic(hbar, mass, g_param, k_boltz, mu_chem_potential, c1_val, c2_val, c3_val, v_val, phase, u, x_end, x_start, num_steps, grid, psi_0, V, iterations, 20)

fname = "Metropolis"
label= r"$|\psi|^2$"
steps = iterations
time_series = generate_time_series_psi_squared_stochastic(grid, psi_sq_array, steps)
animate_psi(time_series, steps, fname, label, "magenta")
