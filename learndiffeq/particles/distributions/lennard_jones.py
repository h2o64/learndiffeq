# Lennard-Jones particle system

# Libraries
import torch
from ..utils import gram, make_mask, reshape_tri


class LennardJones(torch.nn.Module):

    def __init__(self, n_particles, dim_phys, sigma=1.0, epsilon=1.0, r_min_sq=0.0):
        """Constructor

        Args:
            n_particles (int): Number of particles
            dim_phys (int): Dimension of the particles
            sigma (int): Radius of a particle
            epsilon (int): Base energy E_0 (default is 1.0)
            r_min_sq (float): Minimum square radius (default is 0.0)
        """

        # Call the parent constructor
        super(LennardJones, self).__init__()
        # Store the parameters
        self.sigma = sigma
        self.epsilon = epsilon
        self.r_min_sq = r_min_sq
        # Make the masks
        self.register_buffer('mask', make_mask(n_particles, dim_phys), persistent=False)
        self.register_buffer('mask_local',
                             torch.zeros((n_particles, n_particles, dim_phys, dim_phys), dtype=bool),
                             persistent=False
                             )
        arr = torch.arange(n_particles)
        self.mask_local[arr, arr] = True

    def Wpp(self, r_sq, sigma=None):
        """Lennard-Jones potential at the particle level

        Args:
            r_sq (torch.Tensor): Squared distance between two particles
            sigma (torch.Tensor): Value of sigma (default is None)

        Returns:
            energies (torch.Tensor with the same shape as r_sq) : Energy associated to each pair of particles
        """

        if sigma is not None:
            sigma_ = sigma
        else:
            sigma_ = self.sigma
        idr2 = sigma_ * sigma_ / r_sq
        idr6 = idr2 ** 3
        idr12 = idr6 * idr6
        return 4. * self.epsilon * (idr12 - idr6)

    def grad_Wpp(self, r_sq, sigma=None):
        """First derivative of the Lennard-Jones potential at the particle level

        Args:
            r_sq (torch.Tensor): Squared distance between two particles
            sigma (torch.Tensor): Value of sigma (default is None)

        Returns:
            grad_energies (torch.Tensor with the same shape as r_sq) : First derivative of the energy associated
                to each pair of particles
        """

        if sigma is not None:
            sigma_ = sigma
        else:
            sigma_ = self.sigma
        idr2 = sigma_ * sigma_ / r_sq
        idr6 = idr2 ** 3
        idr12 = idr6 * idr6
        return -12. * self.epsilon * (2. * (idr12 / r_sq) - (idr6 / r_sq))

    def grad_grad_Wpp(self, r_sq, sigma=None):
        """Second derivative of the Lennard-Jones potential at the particle level

        Args:
            r_sq (torch.Tensor): Squared distance between two particles
            sigma (torch.Tensor): Value of sigma (default is None)

        Returns:
            grad_grad_energies (torch.Tensor with the same shape as r_sq) : Second derivative of the energy associated
                to each pair of particles
        """

        r_cu = r_sq * r_sq
        if sigma is not None:
            sigma_ = sigma
        else:
            sigma_ = self.sigma
        idr2 = sigma_ * sigma_ / r_sq
        idr6 = idr2 ** 3
        idr12 = idr6 * idr6
        return 24. * self.epsilon * (7. * (idr12 / r_cu) - 2. * (idr6 / r_cu))

    def U(self, x, **kwargs):
        """Lennard-Jones potential at the system level

        The computations are optimized for the tri-diagonal pairwise structure of the potential

        Args:
            x (torch.Tensor of shape (batch_size, n_particles, dim_phys)): Particles
            kwargs (dict): Arguments to pass to Wpp

        Returns:
            U (torch.Tensor of shape (batch_size,)) : Potential of the system
        """

        # Compute the gram matrix
        y = gram(x, only_upper_tri=True)
        # Compute the squared norm
        y_norms_sq = torch.sum(torch.square(y), dim=-1) + self.r_min_sq
        # Compute the potential
        U_pp = self.Wpp(y_norms_sq, **kwargs)
        return U_pp.sum(1)

    def grad_U(self, x, return_U=False, **kwargs):
        """Gradient of the Lennard-Jones potential at the system level

        The computations are optimized for the tri-diagonal pairwise structure of the potential

        Args:
            x (torch.Tensor of shape (batch_size, n_particles, dim_phys)): Particles
            return_U (bool): Whether to compute U(x) at the same time (default is False)
            kwargs (dict): Arguments to pass to Wpp

        Returns:
            if return_U
                U (torch.Tensor of shape (batch_size,)) : Potential of the system
                grad_U (torch.Tensor of shape (batch_size, n_particles, dim_phys)) : Gradient of the potential of the system
            otherwise
                grad_U (torch.Tensor of shape (batch_size, n_particles, dim_phys)) : Gradient of the potential of the system
        """

        # Parse the shape
        batch_size, n_particles, dim_phys = x.shape
        # Compute the gram matrix
        y = gram(x, only_upper_tri=True)
        # Compute the squared norm
        y_norms_sq = torch.sum(torch.square(y), dim=-1, keepdim=True) + self.r_min_sq
        # Compute the gradient of the potential
        grad_U_pp = self.grad_Wpp(y_norms_sq, **kwargs)
        # Reshape things
        y_full = reshape_tri(y, -y, batch_size, n_particles, self.mask)
        grad_U_pp_full = reshape_tri(grad_U_pp, grad_U_pp, batch_size, n_particles, self.mask, expand=True)
        # Compute the gradient
        grad_U = -2. * (y_full * grad_U_pp_full).sum(dim=2)
        # Return everything
        if return_U:
            U = self.Wpp(y_norms_sq, **kwargs).squeeze(-1).sum(1)
            return U, grad_U
        else:
            return grad_U

    def hessian_U(self, x, return_grad_U=False, **kwargs):
        """Hessian of the Lennard-Jones potential at the system level

        The computations are optimized for the tri-diagonal pairwise structure of the potential

        Args:
            x (torch.Tensor of shape (batch_size, n_particles, dim_phys)): Particles
            return_grad_U (bool): Whether to compute \nabla U(x) at the same time (default is False)
            kwargs (dict): Arguments to pass to Wpp

        Returns:
            if return_U
                grad_U (torch.Tensor of shape (batch_size, n_particles, dim_phys)) : Gradient of the potential of the system
                hessian_U (torch.Tensor of shape (batch_size, n_particles, dim_phys, n_particles, dim_phys)) : Gradient of the potential of the system
            otherwise
                hessian_U (torch.Tensor of shape (batch_size, n_particles, dim_phys, n_particles, dim_phys)) : Gradient of the potential of the system
        """

        # Parse the shape
        batch_size, n_particles, dim_phys = x.shape
        # Compute the gram matrix
        y = gram(x, only_upper_tri=True)
        # Compute the squared norm
        y_norms_sq = torch.sum(torch.square(y), dim=-1, keepdim=True) + self.r_min_sq
        # Compute grad_Wpp and grad_grad_Wpp
        grad_Wpp = self.grad_Wpp(y_norms_sq, **kwargs)
        grad_grad_Wpp = self.grad_grad_Wpp(y_norms_sq, **kwargs)
        # Compute the matrix A
        A = 2 * grad_grad_Wpp.unsqueeze(-1) * torch.matmul(y.unsqueeze(-1), y.unsqueeze(-2))
        A += grad_Wpp.unsqueeze(-1) * torch.eye(dim_phys, device=x.device)
        A = reshape_tri(A, A, batch_size, n_particles, self.mask, expand=True)
        # Compute the jacobian
        jac = torch.empty((batch_size, n_particles, n_particles, dim_phys, dim_phys), device=x.device)
        mask_local_ = self.mask_local.unsqueeze(0).expand((batch_size, -1, -1, -1, -1))
        jac[mask_local_] = A.sum(dim=2).flatten()
        jac[~mask_local_] = -1 * A.flatten()
        # Return the jacobian
        if return_grad_U:
            y_full = reshape_tri(y, -y, batch_size, n_particles, self.mask)
            grad_U_pp_full = reshape_tri(grad_Wpp, grad_Wpp, batch_size, n_particles, self.mask, expand=True)
            grad_U = 2. * (y_full * grad_U_pp_full).sum(dim=2)
            return grad_U, 2. * jac.transpose(2, 3)
        else:
            return 2. * jac.transpose(2, 3)


class WCA(LennardJones):
    """Weeks-Chandler-Andersen system"""

    def __init__(self, n_particles, dim_phys, sigma=1.0, epsilon=1.0):
        """Constructor

        Args:
            n_particles (int): Number of particles
            dim_phys (int): Dimension of the particles
            sigma (int): Radius of a particle
            epsilon (int): Base energy E_0 (default is 1.0)
        """

        super().__init__(n_particles=n_particles, dim_phys=dim_phys, sigma=sigma, epsilon=epsilon)
        self.r_lim = pow(2. * self.sigma, 1. / 3.)

    def r_cond(self, r_sq):
        """Addition mask for WCA"""

        return r_sq < self.r_lim

    def Wpp(self, r_sq):
        """Weeks-Chandler-Andersen potential at the particle level

        Args:
            r_sq (torch.Tensor): Squared distance between two particles

        Returns:
            energies (torch.Tensor with the same shape as r_sq) : Energy associated to each pair of particles
        """

        return torch.where(self.r_cond(r_sq), super().Wpp(r_sq) + self.epsilon, torch.zeros_like(r_sq))

    def grad_Wpp(self, r_sq):
        """First derivative of the Weeks-Chandler-Andersen potential at the particle level

        Args:
            r_sq (torch.Tensor): Squared distance between two particles

        Returns:
            grad_energies (torch.Tensor with the same shape as r_sq) : First derivative of the energy associated
                to each pair of particles
        """

        return torch.where(self.r_cond(r_sq), super().grad_Wpp(r_sq), torch.zeros_like(r_sq))

    def grad_grad_Wpp(self, r_sq):
        """Second derivative of the Weeks-Chandler-Andersen potential at the particle level

        Args:
            r_sq (torch.Tensor): Squared distance between two particles

        Returns:
            grad_grad_energies (torch.Tensor with the same shape as r_sq) : Second derivative of the energy associated
                to each pair of particles
        """

        return torch.where(self.r_cond(r_sq), super().grad_grad_Wpp(r_sq), torch.zeros_like(r_sq))


class ModifiedLennardJones(LennardJones):
    """Lennard-Jones potential of the EACF paper"""

    def Wpp(self, r_sq):
        """Lennard-Jones potential at the particle level

        Args:
            r_sq (torch.Tensor): Squared distance between two particles

        Returns:
            energies (torch.Tensor with the same shape as r_sq) : Energy associated to each pair of particles
        """

        idr2 = self.sigma * self.sigma / r_sq
        idr4 = idr2 * idr2
        idr6 = idr4 * idr2
        idr12 = idr6 * idr6
        return 0.5 * self.epsilon * (idr12 - 2 * idr6)

    def grad_Wpp(self, r_sq):
        """First derivative of the Lennard-Jones potential at the particle level

        Args:
            r_sq (torch.Tensor): Squared distance between two particles

        Returns:
            grad_energies (torch.Tensor with the same shape as r_sq) : First derivative of the energy associated
                to each pair of particles
        """

        idr2 = self.sigma * self.sigma / r_sq
        idr4 = idr2 * idr2
        idr6 = idr4 * idr2
        idr12 = idr6 * idr6
        return 12. * 0.5 * self.epsilon * (2. * (idr12 / r_sq) - 2. * (idr6 / r_sq))

    def grad_grad_Wpp(self, r_sq):
        """Second derivative of the Lennard-Jones potential at the particle level

        Args:
            r_sq (torch.Tensor): Squared distance between two particles

        Returns:
            grad_grad_energies (torch.Tensor with the same shape as r_sq) : Second derivative of the energy associated
                to each pair of particles
        """

        r_sq2 = r_sq * r_sq
        idr2 = self.sigma * self.sigma / r_sq
        idr4 = idr2 * idr2
        idr6 = idr4 * idr2
        idr12 = idr6 * idr6
        return 24. * 0.5 * self.epsilon * (2. * (idr6 / r_sq2) - 7. * 2. * (idr12 / r_sq2))


class ShiftedLennardJones(LennardJones):
    """Make a shifted version of Lennard Jones"""

    def __init__(self, **kwargs):
        """Constructor of the shifted Lennard-Jones potential

        Args:
            kwargs (dict): Arguments to pass to the LennardJones constructor
        """

        super().__init__(**kwargs)
        self.sigma = 1.0

    def compute_sigma(self, delta):
        """Compute sigma from delta

        Args:
            delta (torch.Tensor): Value of delta

        Returns:
            sigma (torch.Tensor): Value of sigma
        """

        return delta * pow(2., -1./6.) + 1.

    def Wpp(self, r_sq, delta):
        """Shifted Lennard-Jones potential at the particle level

        Args:
            r_sq (torch.Tensor): Squared distance between two particles
            delta (torch.Tensor): Value of delta

        Returns:
            energies (torch.Tensor with the same shape as r_sq) : Energy associated to each pair of particles
        """

        if len(delta.shape) != len(r_sq.shape):
            delta = delta.view((r_sq.shape[0], *(1,) * (len(r_sq.shape)-1)))
        return super().Wpp(torch.square(torch.sqrt(r_sq) + delta), sigma=self.compute_sigma(delta))

    def grad_Wpp(self, r_sq, delta):
        """First derivative of the shifted Lennard-Jones potential at the particle level

        Args:
            r_sq (torch.Tensor): Squared distance between two particles
            delta (torch.Tensor): Value of delta

        Returns:
            grad_energies (torch.Tensor with the same shape as r_sq) : First derivative of the energy associated
                to each pair of particles
        """

        r = torch.sqrt(r_sq)
        return (1. + (delta / r)) * super().grad_Wpp(torch.square(r + delta), sigma=self.compute_sigma(delta))

    def grad_grad_Wpp(self, r_sq, delta):
        """Second derivative of the shifted Lennard-Jones potential at the particle level

        Args:
            r_sq (torch.Tensor): Squared distance between two particles
            delta (torch.Tensor): Value of delta

        Returns:
            grad_grad_energies (torch.Tensor with the same shape as r_sq) : Second derivative of the energy associated
                to each pair of particles
        """

        r = torch.sqrt(r_sq)
        lj_grad_grad_Wpp = super().grad_grad_Wpp(torch.square(r + delta), sigma=self.compute_sigma(delta))
        lj_grad_Wpp = super().grad_Wpp(torch.square(r + delta), sigma=self.compute_sigma(delta))
        r_sq_pow_three_half = torch.pow(r, 3.)
        ret = 2. * torch.square(delta) * r * lj_grad_grad_Wpp + 2 * r_sq_pow_three_half * lj_grad_grad_Wpp
        ret += 4. * delta * r_sq * lj_grad_grad_Wpp - delta * lj_grad_Wpp
        return 0.5 * ret / r_sq_pow_three_half

    def U(self, x, delta):
        """Shifted Lennard-Jones potential at the system level

        The computations are optimized for the tri-diagonal pairwise structure of the potential

        Args:
            x (torch.Tensor of shape (batch_size, n_particles, dim_phys)): Particles
            delta (torch.Tensor of shape (batch_size, 1, 1)): Delta

        Returns:
            U (torch.Tensor of shape (batch_size,)) : Potential of the system
        """

        return super().U(x=x, delta=delta)

    def grad_U(self, x, delta, return_U=False):
        """Gradient of the Lennard-Jones potential at the system level

        The computations are optimized for the tri-diagonal pairwise structure of the potential

        Args:
            x (torch.Tensor of shape (batch_size, n_particles, dim_phys)): Particles
            return_U (bool): Whether to compute U(x) at the same time (default is False)
            delta (torch.Tensor of shape (batch_size, 1, 1)): Delta

        Returns:
            if return_U
                U (torch.Tensor of shape (batch_size,)) : Potential of the system
                grad_U (torch.Tensor of shape (batch_size, n_particles, dim_phys)) : Gradient of the potential of the system
            otherwise
                grad_U (torch.Tensor of shape (batch_size, n_particles, dim_phys)) : Gradient of the potential of the system
        """

        return super().grad_U(x=x, delta=delta, return_U=return_U)

    def hessian_U(self, x, delta, return_grad_U=False):
        """Hessian of the Lennard-Jones potential at the system level

        The computations are optimized for the tri-diagonal pairwise structure of the potential

        Args:
            x (torch.Tensor of shape (batch_size, n_particles, dim_phys)): Particles
            return_grad_U (bool): Whether to compute \nabla U(x) at the same time (default is False)
            delta (torch.Tensor of shape (batch_size, 1, 1)): Delta

        Returns:
            if return_U
                grad_U (torch.Tensor of shape (batch_size, n_particles, dim_phys)) : Gradient of the potential of the system
                hessian_U (torch.Tensor of shape (batch_size, n_particles, dim_phys, n_particles, dim_phys)) : Gradient of the potential of the system
            otherwise
                hessian_U (torch.Tensor of shape (batch_size, n_particles, dim_phys, n_particles, dim_phys)) : Gradient of the potential of the system
        """

        return super().hessian_U(x=x, delta=delta, return_grad_U=return_grad_U)
