import numpy as np

class GPESolver:
    """
    A class for solving the 1D time-dependent Gross-Pitaevskii equation
    using a semi-implicit Crank-Nicolson scheme with finite differences.

    Parameters:
        dt (float): Time step.
        dx (float): Spatial step.
        n (int): Number of grid points.
        steps (int): Number of time steps.
        g (float): Nonlinear interaction strength.
        hbar (float): Reduced Planck's constant (default: 1).
        mass (float): Particle mass (default: 1).
    """
    def __init__(self, dt: float, dx: float, n: int, steps: int, g: float, hbar: float = 1.0, mass: float = 1.0):
        """
        Args:
            dt: delta t
            dx: delta x
            n: number of grid points
            steps: number of time steps (time-evolution)
            g: nonlinear interaction strength
            hbar: reduced Planck's constant
            mass: particle mass
        """
        self.dt = dt
        self.dx = dx
        self.n = n
        self.steps = steps
        self.g = g  # Nonlinear coupling strength
        self.hbar = hbar
        self.mass = mass

        self.x = None
        self.psi = None           # Current wave function
        self.psi_total = None     # Store history of wave function
        self.potential_ext = None # External potential V(x)
        self.hamiltonian_linear = None # Stores the linear part (T+V_ext) of H

        # Coefficient for kinetic energy term in Hamiltonian
        self.kinetic_coeff = self.hbar**2 / (2 * self.mass * self.dx**2)

    def create_grid(self, x_min: float, x_max: float):
        """
        Create Space Grid
        Args:
            x_min: Minimum x value
            x_max: Maximum x value
        Returns:
            - Numpy array for x grid
        """
        self.x = np.linspace(x_min, x_max, self.n)
        return self.x

    # --- Initial Wavefunction Methods ---
    def gaussian_wave_packet(self, x0, sigma, k0):
        """Generates a Gaussian wave packet."""
        if self.x is None:
            raise ValueError("Grid must be created before setting initial wavefunction.")
        # Normalization to ensure |psi|^2 dx = 1
        A = (1 / (sigma * np.sqrt(2*np.pi)))**0.5
        gaussian_part = np.exp(-(self.x - x0)**2 / (2 * sigma**2))
        plane_wave_part = np.exp(1j * k0 * self.x)
        self.psi = A * gaussian_part * plane_wave_part
        # Ensure normalization numerically
        self.normalize_psi()
        return self.psi

    def ground_state_gaussian(self, sigma=1.0):
        """Generates a simple centered Gaussian (approx ground state for some potentials)."""
        if self.x is None:
            raise ValueError("Grid must be created before setting initial wavefunction.")
         # Normalization to ensure |psi|^2 dx = 1
        A = (1 / (sigma * np.sqrt(2*np.pi)))**0.5
        self.psi = A * np.exp(-self.x**2 / (2 * sigma**2))
        self.normalize_psi()
        return self.psi

    def normalize_psi(self):
        """Normalizes the current wavefunction self.psi."""
        if self.psi is not None:
            norm = np.sqrt(np.sum(np.abs(self.psi)**2) * self.dx)
            if norm > 1e-12: # Avoid division by zero
                 self.psi = self.psi / norm
            else:
                print("Warning: Wavefunction norm is close to zero.")
        return self.psi

    # --- External Potential Methods ---
    def set_potential(self, potential_array: np.ndarray):
        """Sets the external potential directly from an array."""
        if potential_array.shape != (self.n,):
             raise ValueError(f"Potential array shape {potential_array.shape} must match grid size ({self.n},).")
        self.potential_ext = potential_array.astype(complex) # Ensure complex type

    def potential_zero(self):
        """Sets zero potential."""
        if self.x is None:
            raise ValueError("Grid must be created before setting potential.")
        self.potential_ext = np.zeros_like(self.x, dtype=complex)
        return self.potential_ext

    def potential_sho(self, omega: float = 1.0):
        """Simple Harmonic Oscillator potential V = 0.5 * m * omega^2 * x^2."""
        if self.x is None:
            raise ValueError("Grid must be created before setting potential.")
        self.potential_ext = 0.5 * self.mass * omega**2 * self.x**2
        return self.potential_ext.astype(complex)

    def potential_barrier(self, v0: float, x_left: float, x_right: float):
        """Rectangular potential barrier."""
        if self.x is None:
            raise ValueError("Grid must be created before setting potential.")
        self.potential_ext = np.zeros_like(self.x, dtype=complex)
        mask = (self.x > x_left) & (self.x < x_right)
        self.potential_ext[mask] = v0
        return self.potential_ext

    # --- Hamiltonian ---
    def create_linear_hamiltonian(self):
        """
        Construct the time-independent LINEAR part of the Hamiltonian matrix H = T + V_ext
        using finite differences for the kinetic term T = -hbar^2/(2m) d^2/dx^2.
        Handles boundary conditions implicitly (assuming psi=0 at ends, typical for CN).

        Returns:
            - Square matrix (n x n) of the linear Hamiltonian (T + V_ext).
              Using dense matrix here, consider sparse for large n.
        """
        if self.potential_ext is None:
            raise ValueError("External potential must be set before creating the Hamiltonian.")

        # Kinetic Energy Term Coefficients (using self.kinetic_coeff)
        diag_kin = 2.0 * self.kinetic_coeff
        offdiag_kin = -1.0 * self.kinetic_coeff

        self.hamiltonian_linear = np.zeros((self.n, self.n), dtype=complex)

        # Fill diagonal (Kinetic + External Potential)
        np.fill_diagonal(self.hamiltonian_linear, diag_kin + self.potential_ext)

        # Fill off-diagonals (Kinetic)
        indices = np.arange(self.n - 1)
        self.hamiltonian_linear[indices, indices + 1] = offdiag_kin
        self.hamiltonian_linear[indices + 1, indices] = offdiag_kin

        return self.hamiltonian_linear


    # --- Time Evolution ---
    def solve_gpe_crank_nicolson(self):
        """
        Solve the GPE using the semi-implicit Crank-Nicolson scheme.
        (I + idt/(2*hbar)*H(psi_n)) psi_{n+1} = (I - idt/(2*hbar)*H(psi_n)) psi_n

        Where H(psi_n) = T + V_ext + g * |psi_n|^2 is the effective Hamiltonian
        evaluated using the wavefunction at the current time step n.

        Returns:
            - List containing the wavefunction psi at each time step.
        """
        if self.psi is None:
            raise ValueError("Initial wavefunction psi must be set before solving.")
        if self.potential_ext is None:
            # Default to zero potential if none set
            print("Warning: External potential not explicitly set. Using zero potential.")
            self.potential_zero()

        # 1. Create the time-independent linear part of the Hamiltonian matrix (T + V_ext)
        self.create_linear_hamiltonian()

        # 2. Prepare time evolution
        self.psi_total = [self.psi.copy()] # Store initial state
        identity_matrix = np.identity(self.n, dtype=complex)
        const_factor = 1j * self.dt / (2 * self.hbar)

        # --- Time Stepping Loop ---
        for _ in range(self.steps):
            # 3. Calculate the nonlinear potential term at the current time n
            # V_nonlinear is a vector containing g * |psi_j|^2 at each grid point j
            V_nonlinear = self.g * np.abs(self.psi)**2

            # 4. Construct the effective Hamiltonian for this step H_eff = H_linear + V_nl(psi_n)
            # We add V_nonlinear to the diagonal of the linear Hamiltonian matrix
            H_effective = self.hamiltonian_linear + np.diag(V_nonlinear)

            # 5. Construct the Crank-Nicolson matrices A and B
            # A = I + const_factor * H_effective
            # B = I - const_factor * H_effective
            forward_matrix = identity_matrix + const_factor * H_effective
            backward_matrix = identity_matrix - const_factor * H_effective

            # 6. Define the right-hand side (RHS) vector
            rhs_vector = backward_matrix @ self.psi

            # 7. Solve the linear system A * psi_{n+1} = RHS for psi_{n+1}
            try:
                self.psi = np.linalg.solve(forward_matrix, rhs_vector)
            except np.linalg.LinAlgError:
                 print(f"Error: Linear system solve failed at step {_ + 1}. Matrix might be singular.")
                 break

            # 8. Normalize the wavefunction (optional but recommended for GPE)
            # GPE conserves norm, but numerical errors can accumulate.
            self.normalize_psi()

            # 9. Store the result
            self.psi_total.append(self.psi.copy())
            # --- End Time Stepping Loop ---

        return np.array(self.psi_total) # Return as a numpy array
