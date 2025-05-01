import numpy as np

class GPESolver:
    """
    Class implementation for solving the GPE PDE using the Crank-Nicolson scheme.
    Parameters:
        dt (float): Time step.
        dx (float): Spatial step.
        n (int): Number of grid points.
        steps (int): Number of time steps.
        g (float): Interaction strength.
        hbar (float): Reduced Planck constant.
        mass (float): Mass of the Condensate.
    """
    def __init__(self, dt: float, dx: float, n: int, steps: int, g:float, hbar: float = 1.0, mass: float = 1.0):
        """
        Args:
            dt: delta t
            dx: delta x
            n: number of grid points
            steps: number of time steps (time-evolution)
        """
        self.dt = dt
        self.dx = dx
        self.n = n
        self.steps = steps
        self.g = g
        self.hbar = hbar
        self.mass = mass

        self.x = None
        self.psi = None
        self.psi_total = None
        self.potential = None
        self.potential_ext = None
        self.linear_hamiltonian = None

        self.kinetic_coeff = self.hbar**2 / (2*self.mass *self.dx**2)

    def create_grid(self, x_min: float, x_max: float):
        """
        Create Space Grid
        Args:
            x_min:
            x_max:
        Returns:
            - Numpy array
        """
        self.x = np.linspace(x_min, x_max, self.n)
        return self.x

    def normalize_psi(self):
        """
        Normalizes the current wavefunction self.psi.
            Parameters:
                - None
            Returns:
                - Numpy array with normalized wave function
        """
        if self.psi is not None:
            norm = np.sqrt(np.sum(np.abs(self.psi)**2) * self.dx)
            if norm > 1e-12: # Avoid division by zero
                    self.psi = self.psi / norm
            else:
                print("Warning: Wavefunction norm is close to zero.")
        return self.psi

    # --- Initial Wavefunction Methods ---
    def gaussian_wave_packet(self, x0, sigma, k0) -> np.ndarray:
        """
        Generates a Gaussian wave packet.
            Parameters:
                x0 (float): Center of the Gaussian wave packet.
                sigma (float): Width of the Gaussian wave packet.
                k0 (float): Momentum of the Gaussian wave packet.
            Returns:
                self.psi (numpy.ndarray): The Gaussian wave packet.
        """
        if self.x is None:
            raise ValueError("Grid must be created before setting initial wavefunction.")
        # Normalization constant A ensures integral |psi|^2 dx = 1
        A = (1 / (sigma * np.sqrt(2*np.pi)))**0.5 # Correction for Gaussian normalization
        gaussian_part = np.exp(-(self.x - x0)**2 / (2 * sigma**2))
        plane_wave_part = np.exp(1j * k0 * self.x)
        self.psi = A * gaussian_part * plane_wave_part
        # Ensure normalization numerically
        self.normalize_psi()
        return self.psi

    def ground_state_gaussian(self, sigma=1.0) -> np.ndarray:
        """
            Generates a simple centered Gaussian (approx ground state for some potentials).
                Parameters:
                    sigma (float): Standard deviation of the Gaussian distribution.
                Returns:
                    np.ndarray: Ground state wavefunction.
        """
        if self.x is None:
            raise ValueError("Grid must be created before setting initial wavefunction.")
            # Normalization constant A ensures integral |psi|^2 dx = 1
        A = (1 / (sigma * np.sqrt(2*np.pi)))**0.5 # Correction for Gaussian normalization
        self.psi = A * np.exp(-self.x**2 / (2 * sigma**2))
        self.normalize_psi()
        return self.psi

    # Declare External Potentials
    def set_potential(self, potential_array: np.ndarray) -> np.ndarray:
        if potential_array.shape != (self.n,):
            raise ValueError(f"Potential array {potential_array.shape} must match grid size ({self.n},).")
        self.potential_ext = potential_array.astype(complex)
        return self.potential_ext

    def potential_zero(self) -> np.ndarray:
        """
        Set zero potential
            Parameters:
                None
            Returns:
                - Numpy array with zero potential
        """
        if self.x is None:
            raise ValueError("Grid must be created before setting the potential.")
        self.potential_ext = np.zeros_like(self.x, dtype=complex)
        return self.potential_ext

    def sw_potential(self):
        """
        Create Square Well potential
        Returns:
            - Numpy array with Square Well potential
        """
        self.potential = np.zeros_like(self.x)
        self.potential[0] = 1e30
        self.potential[-1] = 1e30
        return self.potential

    def sho_potential(self) -> np.ndarray:
        """
        Create SHO potential
        Returns:
            - Numpy array with harmonic potential
        """
        if self.x is None:
            raise ValueError("Grid must be created before setting the potential.")
        self.potential = 0.5 * self.x ** 2
        return self.potential

    def create_linear_hamiltonian(self):
        """
        Create linear Hamiltonian matrix
            Returns:
                - Numpy array with linear Hamiltonian matrix
        """
        if self.potential_ext is None:
            raise ValueError("External potential must be set before creating the Hamiltonian.")
        diag_kin = 2.0 * self.kinetic_coeff
        offdiag_kin = -1.0 * self.kinetic_coeff
        self.linear_hamiltonian = np.zeros((self.n, self.n), dtype=complex)
        # Fill diagonal elements
        np.fill_diagonal(self.linear_hamiltonian, diag_kin+self.potential_ext)
        # Fill non-diagonal elements of the matrix
        indices = np.arange(self.n-1)
        self.linear_hamiltonian[indices, indices+1] = offdiag_kin
        self.linear_hamiltonian[indices+1, indices] = offdiag_kin
        return self.linear_hamiltonian

    def solve_gpe_crank_nicolson(self):
        """
        Crank Nicolson-like implementation of GPE equation
            Parameters:
                -None
            Returns:
                - Numpy array with wavefunctions
        """
        # if  self.psi == None:
        #     raise ValueError("Initial wavefunction must be set first.")
        # if self.potential_ext is None:
        #     print("Warning: Exrernal potential not explicitly set. Using a zero potential.")
        #     self.potential_zero()
        # 1. Create the time-dependent linear part of H
        self.create_linear_hamiltonian()

        # 2. Setup time evolution
        self.psi_total = [self.psi.copy()]
        identity_matrix = np.identity(self.n, dtype=complex)
        c = 1j + self.dt / (2*self.hbar)

        # Perform Time Evolution
        for _ in range(self.steps):
            # 3. Include non-linear potential (from GPE equation)
            nonlinear_potential = self.g * np.abs(self.psi)**2
            # 4. Build the effective (full) Hamiltonian
            effective_Hamiltonian = self.linear_hamiltonian + np.diag(nonlinear_potential)
            # 5. Crank-Nicolson matrices A and B
            forward_matrix  = identity_matrix + c * effective_Hamiltonian
            backward_matrix = identity_matrix - c * effective_Hamiltonian
            rhs = backward_matrix @ self.psi
            self.psi = np.linalg.solve(forward_matrix, rhs)
            # Normalize the wave fucntion

            self.psi_total.append(self.psi.copy())
        return np.array(self.psi_total)
