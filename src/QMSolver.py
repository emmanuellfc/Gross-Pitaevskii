# Load numpy Module
import numpy as np

# Main Class Definition
class QMSolver:
    """
    A class for solving the time-dependent Schrödinger equation using the Crank-Nicolson scheme.

    Parameters:
        dt (float): Time step.
        dx (float): Spatial step.
        n (int): Number of grid points.
        steps (int): Number of time steps.
    """
    def __init__(self, dt: float, dx: float, n: int, steps: int):
        """
        Args:
            dt: delta t
            dx: delta x
            n: number of grid points
            steps: number of time steps (time-evolution)
        """
        self.dt = dt
        self.n = n
        self.steps = steps
        self.x = None
        self.dx = dx
        self.psi = None
        self.psi_total = None
        self.potential = None
        self.hamiltonian = None

    def create_grid(self, x_min: float, x_max: float):
        """
        Create Space Grid
        Args:
            x_min:
            x_max:
        Returns:
            - Numpy array
        """
        # self.dx = (x_max - x_min)/self.n
        self.x = np.linspace(x_min, x_max, self.n)
        return self.x

    def gaussian_wave_packet(self, x0, sigma, k0):
        """Generates a Gaussian wave packet."""
        A = (1 / (sigma * np.sqrt(np.pi))) ** 0.5
        self.psi = A * np.exp(-(self.x - x0) ** 2 / (2 * sigma ** 2)) * np.exp(1j * k0 * self.x)
        return self.psi

    def gaussian_wave(self):
        """
        Create initial condition for psi
        Returns:
            - Numpy array with initial condition
        """
        self.psi = (1/np.pi**0.25) * np.exp(-0.5*self.x**2)

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

    def sho_potential(self):
        """
        Create SHO potential
        Returns:
            - Numpy array with harmonic potential
        """
        self.potential = 0.5 * self.x ** 2
        return self.potential


    def tw_potential(self):
        """
        Create SHO potential
        Returns:
            - Numpy array with harmonic potential
        """
        self.potential = np.abs(self.x)
        return self.potential

    def potential_barrier(self, v0, x_left, x_right):
        """
        Create a Potential Barrier
        Args:
            v0:
            x_left:
            x_right:

        Returns:
            - Numpy array with barrier potential
        """
        self.potential = np.zeros_like(self.psi)
        for i, xi in enumerate(self.x):
            if x_left < xi < x_right:
                self.potential[i] = v0
        return self.potential

    def create_hamiltonian_fd(self):
        """
        Construct the Hamiltonian matrix following a Finite-Difference Scheme
        Returns:
            - Square matrix of the Hamiltonian
        """
        self.hamiltonian = np.zeros((self.n, self.n), dtype=complex)  # Initialize A as a complex matrix
        c = self.dt / (1j * self.dx ** 2)
        for j in range(self.n):
            self.hamiltonian[j, j] = 1 + 2 * c - 1j * self.dt * 10*self.potential[j]**2  # Diagonal elements
            if j > 0:
                self.hamiltonian[j, j - 1] = -c  # Lower diagonal
            if j < self.n - 1:
                self.hamiltonian[j, j + 1] = -c  # Upper diagonal
        return self.hamiltonian

    def create_hamiltonian_cn(self):
        """
        Construct the Hamiltonian matrix following a Crank-Nicolson Scheme
        Returns:
            - Square matrix of the Hamiltonian
        """
        # Construct the Hamiltonian matrix
        c: float = - 1 / (2 * self.dx ** 2)
        self.hamiltonian = np.zeros((self.n, self.n), dtype=complex)
        for i in range(1, self.n - 1):
            self.hamiltonian[i, i] = 2 * c + self.potential[i]
            self.hamiltonian[i, i] = 2 * c
            self.hamiltonian[i, i + 1] = -c
            self.hamiltonian[i, i - 1] = -c
        return self.hamiltonian

    def solve_finite_difference(self):
        """
        Solve the system using a Finite Difference Scheme
        Returns:
            Wave function at all times"""

        # Time evolution
        self.psi_total = []
        for n in range(self.steps):
            # Add psi squared to the Hamiltonian
            # for i in range(1, self.n - 1):
            #     print(self.psi[i])
            #     # print(foo.size)
            #     self.hamiltonian[i, i] += self.psi[i]**2
            # Solve for psi
            self.psi = np.linalg.solve(self.hamiltonian, self.psi)
            self.psi = self.psi / np.sqrt(np.sum(np.abs(self.psi) ** 2) * self.dx)
            # Append the wave function to psi total
            self.psi_total.append(self.psi)
        return self.psi_total

    def solve_crank_nicolson(self):
        """
        Solve the system using a Crank Nisolson Scheme
        Returns:
            Wave function at all times
        """
        identity_matrix = np.identity(self.n, dtype=complex)
        # Create matrices A and B
        forward_matrix  = identity_matrix + (1j / 2) * self.hamiltonian * self.dt
        backward_matrix = identity_matrix - (1j / 2) * self.hamiltonian * self.dt
        # Time evolution
        self.psi_total = []
        for n in range(self.steps):
            #

            # Solve the linear system
            self.psi = np.linalg.solve(forward_matrix, backward_matrix @ self.psi)
            # Normalize the wave-function
            self.psi = self.psi / np.sqrt(np.sum(np.abs(self.psi) ** 2) * self.dx)
            # Append the wave function to psi total
            self.psi_total.append(self.psi)
        return self.psi_total
