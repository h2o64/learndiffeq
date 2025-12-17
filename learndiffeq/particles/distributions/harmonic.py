# Construct an harmonic potential

# Libraries
import torch


class Harmonic(torch.nn.Module):

    def __init__(self, n_particles, dim_phys, k=1.0):
        """Constructor of an Harmonic potential. The origin of the harmonic potential is set
        at the origin.

        Args:
            n_particles (int): Number of particles
            dim_phys (int): Dimension of the particles
            k (float): Harmonic potential coefficient (default is 1.0)
        """

        # Call the parent constructor
        super(Harmonic, self).__init__()
        # Store the parameters
        self.n_particles = n_particles
        self.dim_phys = dim_phys
        self.k = k
        # Compute the hessian
        ones = torch.ones((self.n_particles, self.dim_phys, self.n_particles, self.dim_phys))
        self.hess_base = -ones / self.n_particles
        idx = torch.arange(self.n_particles)
        self.hess_base[idx, :, idx, :] += 1
        dim_mask = (torch.arange(self.dim_phys) != torch.arange(self.dim_phys)[:, None]).int()
        for i in range(self.dim_phys):
            self.hess_base[:, dim_mask[i, 0], :, dim_mask[i, 1]] = 0
        self.hess_base *= self.k

    def U(self, x):
        """Harmonic potential at the system level

        Args:
            x (torch.Tensor of shape (batch_size, n_particles, dim_phys)): Particles

        Returns:
            U (torch.Tensor of shape (batch_size,)) : Harmonic potential of the system
        """

        x_mean = x.mean(dim=1, keepdim=True)
        return 0.5 * self.k * torch.sum(torch.square(x - x_mean), dim=(1, 2))

    def grad_U(self, x, return_U=False):
        """Gradient of the harmonic potential

        Args:
            x (torch.Tensor of shape (batch_size, n_particles, dim_phys)): Particles
            return_U (bool): Whether to compute U(x) at the same time (default is False)

        Returns:
            if return_U
                U (torch.Tensor of shape (batch_size,)) : Potential of the system
                grad_U (torch.Tensor of shape (batch_size, n_particles, dim_phys)) : Gradient of the potential of the system
            otherwise
                grad_U (torch.Tensor of shape (batch_size, n_particles, dim_phys)) : Gradient of the potential of the system
        """

        x_mean = x.mean(dim=1, keepdim=True)
        grad_U = self.k * (x - x_mean)
        if return_U:
            return self.U(x), grad_U
        else:
            return grad_U

    def hessian_U(self, x, return_grad_U=False):
        """Hessian of the harmonic potential

        Args:
            x (torch.Tensor of shape (batch_size, n_particles, dim_phys)): Particles
            return_grad_U (bool): Whether to compute \nabla U(x) at the same time (default is False)

        Returns:
            if return_U
                grad_U (torch.Tensor of shape (batch_size, n_particles, dim_phys)) : Gradient of the potential of the system
                hessian_U (torch.Tensor of shape (batch_size, n_particles, dim_phys, n_particles, dim_phys)) : Gradient of the potential of the system
            otherwise
                hessian_U (torch.Tensor of shape (batch_size, n_particles, dim_phys, n_particles, dim_phys)) : Gradient of the potential of the system
        """

        hess_U = self.hess_base.unsqueeze(0).repeat((x.shape[0], 1, 1, 1, 1))
        if return_grad_U:
            return self.k * x, hess_U
        else:
            return hess_U

    def _apply(self, fn):
        """Apply fn to the distribution tensors"""

        new_self = super(Harmonic, self)._apply(fn)
        new_self.hess_base = fn(new_self.hess_base)
        return new_self
