# Particle system from "Adaptive Monte Carlo Augmented with Normalizing Flows"

# Libraries
import torch
from ..utils import gram


class ParticleSystem:
    """
    Particle system with attractive interations at length scale a
    and in an harmonic potential of size L in each physical dimension
    """

    def __init__(self, n_particles=100, dim_phys=1):
        """Constructor

        Args:
            n_particles (int): Number of particles (default is 100)
            dim_phys (int): Dimension of the particles (default is 1)
            sigma (int): Radius of a particle
        """

        self.n_particles = n_particles
        self.dim_phys = dim_phys

    def U(self, x):
        """System's potential at the system level

        Args:
            x (torch.Tensor of shape (batch_size, n_particles, dim_phys)): Particles

        Returns:
            U (torch.Tensor of shape (batch_size,)) : Potential of the system
        """

        return

    def grad_U(self, x):
        """Gradient of the system's potential at the system level

        Args:
            x (torch.Tensor of shape (batch_size, n_particles, dim_phys)): Particles

        Returns:
            grad_U (torch.Tensor of shape (batch_size, n_particles, dim_phys)) : Gradient of the potential of the system
        """

        return

    def log_prob(self, x):
        """Compute the log probability of the particle system at temperature 1

        Args:
            x (torch.Tensor of shape (batch_size, n_particles, dim_phys)): Particles

        Returns
            log_prob (torch.Tensor of shape (batch_size,)) : Log-probability
        """

        return -self.U(x)


class ParticleSystemHW(ParticleSystem):
    """
    Particle system with attractive interactions at length scale a
    and in an harmonic potential of size L in each physical dimension
    """

    def __init__(self, n_particles, dim_phys, a=1.0, L=10.0):
        """Constructor

        Args:
            n_particles (int): Number of particles
            dim_phys (int): Dimension of the particles
            a (float): Attractive interaction length (default is 1.0)
            L (float): Length of the box (default is 10.0)
        """

        super().__init__(n_particles, dim_phys)
        self.a = a
        self.L = L

    def Wpp(self, y, y_norms):
        """Potential at the particle level

        Args:
            y (torch.Tensor of shape (batch_size, n_particles, dim_phys)): Particles
            y_norms (torch.Tensor of shape (batch_size, n_particles, n_particles)): Distance between two particles

        Returns:
            energies (torch.Tensor with the same shape as y) : Energy associated to each pair of particles
        """
        Wpp = - torch.exp(- 0.5 * y_norms ** 2 / self.a ** 2)
        return Wpp + torch.eye(self.n_particles, device=y.device,
                               dtype=y.dtype)

    def U(self, x):
        """Potential at the system level

        Args:
            x (torch.Tensor of shape (batch_size, n_particles, dim_phys)): Particles

        Returns:
            U (torch.Tensor of shape (batch_size,)) : Potential of the system
        """

        y = gram(x, self.L)
        y_norms = torch.linalg.norm(y, dim=-1)
        U_pp = 0.5 * self.Wpp(y, y_norms).sum((-2, -1)) / self.n_particles
        U_HW = 0.5 * (torch.norm(x, dim=-1) ** 2).sum(-1) / self.L**2
        return U_pp + U_HW

    def grad_U(self, x):
        """Gradient at the system level

        Args:
            x (torch.Tensor of shape (batch_size, n_particles, dim_phys)): Particles

        Returns:
            grad_U (torch.Tensor of shape (batch_size, n_particles, dim_phys)) : Gradient of the potential of the system
        """

        y = gram(x, self.L)
        y_norms = torch.linalg.norm(y, dim=-1)
        W = self.Wpp(y, y_norms)

        grad = y / self.a**2 * \
            W.view(-1, self.n_particles, self.n_particles, 1)

        return grad.sum(-2) / self.n_particles + x / self.L**2

    def log_prob(self, x):
        """Compute the log probability of the particle system at temperature 1

        Args:
            x (torch.Tensor of shape (batch_size, n_particles, dim_phys)): Particles

        Returns
            log_prob (torch.Tensor of shape (batch_size,)) : Log-probability
        """

        return -self.U(x)


class ParticleSystemPeriodic(ParticleSystem):
    """
    Particle system with attractive interations at length scale a
    and in an harmonic potential of size L in each physical dimension
    """

    def __init__(self, n_particles, dim_phys, a=1.0, L=10.0):
        """Constructor

        Args:
            n_particles (int): Number of particles
            dim_phys (int): Dimension of the particles
            a (float): Attractive interaction length
            L (float): Length of the box
        """

        super().__init__(n_particles, dim_phys)
        self.a = a
        self.L = L
        self.x_min = - L / 2
        self.x_max = L / 2
        self.y_min = - L / 2
        self.y_max = L / 2

    def Wpp(self, y):
        """Potential at the particle level

        Args:
            y (torch.Tensor of shape (batch_size, n_particles, n_particles)): Distance between two particles

        Returns:
            energies (torch.Tensor with the same shape as y) : Energy associated to each pair of particles
        """

        arg = torch.square(torch.sin(2. * torch.pi * y / (2 * self.L))).sum(-1)
        Wpp = -torch.exp(- (2 * self.L)**2 * arg / (4. * torch.pi**2 * self.a**2))
        return Wpp

    def U(self, x):
        """Potential at the system level

        The computations are optimized for the tri-diagonal pairwise structure of the potential

        Args:
            x (torch.Tensor of shape (batch_size, n_particles, dim_phys)): Particles

        Returns:
            U (torch.Tensor of shape (batch_size,)) : Potential of the system
        """

        y = gram(x, only_upper_tri=True)
        U_pp = self.Wpp(y).sum(1) / self.n_particles
        return U_pp

    def grad_U(self, x):
        raise NotImplementedError(
            'grad_U not implemented on ParticleSystemPeriodic')
