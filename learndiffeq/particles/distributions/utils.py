# Utilities for working on particles

# Libraries
import torch
import math
from ..utils import box_particles


class BoxTransform(torch.distributions.transforms.Transform):
    """Box transformation

    Wrap the samples within a box on each call
    """

    domain = torch.distributions.constraints.real
    codomain = torch.distributions.constraints.real
    bijective = True
    sign = +1

    def __init__(self, L, cache_size=0):
        super().__init__(cache_size=cache_size)
        self.L = L

    def _call(self, x):
        return box_particles(x, self.L)

    def _inverse(self, y):
        return box_particles(y, self.L)

    def log_abs_det_jacobian(self, x, y):
        return torch.zeros_like(x)


class CenteringDistribution:
    """Wrapper to center the particles of a distribution when sampling"""

    def __init__(self, base_dist):
        """Constructor

        Args:
            base_dist (torch.distributions.Distribution): Base distribution
        """

        if not (isinstance(base_dist, torch.distributions.Independent) and isinstance(base_dist.base_dist, torch.distributions.Normal)):
            raise NotImplementedError(
                'The centering operator is not supported because the normalizing constant is wrong.')
        self.base_dist = base_dist

    def center_particles(self, x):
        return x - x.mean(dim=1, keepdim=True)

    def sample(self, sample_shape=torch.Size()):
        """Sample the distribution

        Args:
            sample_shape (tuple of int): Shape of the samples

        Returns
            samples (torch.Tensor of shape (*sample_shape, 2)): Samples
        """

        samples = self.base_dist.sample(sample_shape=sample_shape)
        return self.center_particles(samples)

    def log_prob(self, value):
        """Evaluate the log-likelihood of the distribution

        Args:
            value (torch.Tensor of shape (batch_size, 2)): Sample

        Returns
            log_prob (torch.Tensor of shape (batch_size,)): Log-likelihood of the samples
        """

        # Compute the log-prob
        log_prob = self.base_dist.log_prob(self.center_particles(value))
        # Remove the bad normalizing constant
        log_prob -= -0.5 * value.shape[1] * value.shape[2] * \
            math.log(2. * torch.pi * self.base_dist.base_dist.scale[0, 0]**2)
        # Add the good normalizing constant
        log_prob += -0.5 * (value.shape[1]-1) * value.shape[2] * math.log(2. *
                                                                          torch.pi * self.base_dist.base_dist.scale[0, 0]**2)
        # Return the log-prob
        return log_prob


two_over_one_sixth = math.pow(2, 1 / 6)


def generate_lj_2d_minima(n, sigma):
    """Generate the global minimum of the 2D Lennard-Jones system.

    Based on "Two-Dimensional Lennard-Jones Clusters" by Joyce C. Yang (http://yangacademy.com/lj.pdf)

    Args:
        n (int): Number of particles
        sigma (float): Radius of the particles

    Returns:
        minimum (torch.Tensor of shape (n_particles, 2)): The global minimum
    """

    # Coordinates
    x = torch.zeros((n,))
    y = torch.zeros((n,))
    # Indexes
    i = torch.arange(n) + 1
    # Make the initial hexagone
    x[0], y[0] = 0, 0
    x[1], y[1] = 0, 1
    x[2], y[2] = -1, 1
    x[3], y[3] = -1, 0
    x[4], y[4] = 0, -1
    x[5], y[5] = 1, -1
    x[6], y[6] = 1, 0
    mask_first = torch.ones((n,), dtype=bool)
    mask_first[:7] = False
    # Compute l
    l = torch.ceil((-3. + torch.sqrt(12. * i - 3.)) / 6.)
    # Compute r
    r = i - ((3. * l * (l - 1.)) + 1)
    # Compute s
    s = torch.where(r < 6. * l, torch.floor(r / l) + 1., 6. * torch.ones_like(r))
    # Compute a
    a = torch.where(s == 1, torch.zeros_like(s), ((s - 1.) * l) - 1.)
    # Compute k
    k = r - a
    # Compute x and y for all i > 7
    # s == 1
    mask_s = (s == 1) & mask_first
    mask_l = (l % 2 == 0) & mask_first
    if torch.sum(mask_s & mask_l) > 0:
        x[mask_s & mask_l] = (l / 2. + torch.pow(-1, k) * torch.floor(k / 2.))[mask_s & mask_l]
    if torch.sum(mask_s & ~mask_l) > 0:
        x[mask_s & ~mask_l] = ((l + 1.) / 2. - torch.pow(-1, k) * torch.floor(k / 2.))[mask_s & ~mask_l]
    y[mask_s] = (l - x)[mask_s]
    # s == 2
    mask_s = (s == 2) & mask_first
    mask_l = (l % 2 == 0) & mask_first
    if torch.sum(mask_s & mask_l) > 0:
        x[mask_s & mask_l] = (-l / 2. + torch.pow(-1, k) * torch.floor(k / 2.))[mask_s & mask_l]
    if torch.sum(mask_s & ~mask_l) > 0:
        x[mask_s & ~mask_l] = (-(l - 1.) / 2. - torch.pow(-1, k) * torch.floor(k / 2.))[mask_s & ~mask_l]
    y[mask_s] = l[mask_s]
    # s == 3
    mask_s = (s == 3) & mask_first
    mask_l = (l % 2 == 0) & mask_first
    if torch.sum(mask_s & mask_l) > 0:
        y[mask_s & mask_l] = (l / 2. + torch.pow(-1, k) * torch.floor(k / 2.))[mask_s & mask_l]
    if torch.sum(mask_s & ~mask_l) > 0:
        y[mask_s & ~mask_l] = ((l + 1.) / 2. - torch.pow(-1, k) * torch.floor(k / 2.))[mask_s & ~mask_l]
    x[mask_s] = -l[mask_s]
    # s == 4
    mask_s = (s == 4) & mask_first
    mask_l = (l % 2 == 0) & mask_first
    if torch.sum(mask_s & mask_l) > 0:
        x[mask_s & mask_l] = (-l / 2. - torch.pow(-1, k) * torch.floor(k / 2.))[mask_s & mask_l]
    if torch.sum(mask_s & ~mask_l) > 0:
        x[mask_s & ~mask_l] = (-(l + 1.) / 2. + torch.pow(-1, k) * torch.floor(k / 2.))[mask_s & ~mask_l]
    y[mask_s] = (-l - x)[mask_s]
    # s == 5
    mask_s = (s == 5) & mask_first
    mask_l = (l % 2 == 0) & mask_first
    if torch.sum(mask_s & mask_l) > 0:
        x[mask_s & mask_l] = (l / 2. - torch.pow(-1, k) * torch.floor(k / 2.))[mask_s & mask_l]
    if torch.sum(mask_s & ~mask_l) > 0:
        x[mask_s & ~mask_l] = ((l - 1.) / 2. + torch.pow(-1, k) * torch.floor(k / 2.))[mask_s & ~mask_l]
    y[mask_s] = -l[mask_s]
    # s == 6
    mask_s = (s == 6) & mask_first
    mask_l = (l % 2 == 0) & mask_first
    if torch.sum(mask_s & mask_l) > 0:
        y[mask_s & mask_l] = (-l / 2. - torch.pow(-1, k) * torch.floor(k / 2.))[mask_s & mask_l]
    if torch.sum(mask_s & ~mask_l) > 0:
        y[mask_s & ~mask_l] = (-(l + 1.) / 2. + torch.pow(-1, k) * torch.floor(k / 2.))[mask_s & ~mask_l]
    x[mask_s] = l[mask_s]
    # Build (x,y)
    points = torch.stack([x, y], dim=-1)
    # Project points
    P = torch.FloatTensor([
        [1.0, 0.5],
        [0, math.sqrt(3) / 2]
    ])
    points_proj = torch.matmul(P.unsqueeze(0), points.unsqueeze(-1)).squeeze(-1)
    return two_over_one_sixth * sigma * points_proj


class LJGaussianNoOverlap(torch.nn.Module):
    """Gaussian with the means based on the global minima to avoid overlap"""

    def __init__(self, device, n_particles, sigma=1.0, sigma_factor=4.0, scale=0.5):
        """Constructor

        Args:
            device (torch.device): Device used for computations
            n_particles (int): Number of particles
            sigma (float): Radius of the particles
            sigma_factor (float): Factor on the radius of the particles to ensure
                no overlap after variance
            scale (float): Scale of each Gaussian coordinate
        """

        super().__init__()
        self.dist = torch.distributions.Independent(
            base_distribution=torch.distributions.Normal(
                loc=generate_lj_2d_minima(n_particles, sigma_factor * sigma).to(device),
                scale=scale * torch.ones((n_particles, 2), device=device)
            ),
            reinterpreted_batch_ndims=2
        )
        self.x_min = 1.25 * self.dist.base_dist.loc[:, 0].min()
        self.x_max = 1.25 * self.dist.base_dist.loc[:, 0].max()
        self.y_min = 1.25 * self.dist.base_dist.loc[:, 1].min()
        self.y_max = 1.25 * self.dist.base_dist.loc[:, 1].max()

    def sample(self, sample_shape):
        """Sample the distribution

        Args:
            sample_shape (tuple of int): Shape of the samples

        Returns
            samples (torch.Tensor of shape (*sample_shape, n_particles, 2)): Samples
        """

        return self.dist.sample(sample_shape)

    def log_prob(self, value):
        """Evaluate the log-likelihood of the distribution

        Args:
            value (torch.Tensor of shape (batch_size, n_particles, 2)): Sample

        Returns
            log_prob (torch.Tensor of shape (batch_size,)): Log-likelihood of the samples
        """

        return self.dist.log_prob(value)

    def _apply(self, fn):
        """Apply the fn function on the distribution

        Args:
            fn (function): Function to apply on tensors
        """

        new_self = super(LJGaussianNoOverlap, self)._apply(fn)
        new_self.dist.base_dist.loc = fn(new_self.dist.base_dist.loc)
        new_self.dist.base_dist.scale = fn(new_self.dist.base_dist.scale)
        return new_self


class SumDistsWithGrad(torch.nn.Module):

    def __init__(self, dist1, dist2):
        super().__init__()
        self.dist1 = dist1
        self.dist2 = dist2

    def U(self, x):
        """Potential of the sum of the two potentials

        Args:
            x (torch.Tensor of shape (batch_size, *data_shape)): Inputs

        Returns:
            U (torch.Tensor of shape (batch_size,)) : Potential
        """

        return self.dist1.U(x) + self.dist2.U(x)

    def grad_U(self, x, return_U=False):
        """Gradient of the sum of the two potentials

        Args:
            x (torch.Tensor of shape (batch_size, *data_shape)): Inputs
            return_U (bool): Whether to compute U(x) at the same time (default is False)

        Returns:
            if return_U
                U (torch.Tensor of shape (batch_size,)) : Potential
                grad_U (torch.Tensor of shape (batch_size, *data_shape)) : Gradient
            otherwise
                grad_U (torch.Tensor of shape (batch_size, *data_shape)) : Gradient
        """

        if return_U:
            U_1, grad_U_1 = self.dist1.grad_U(x, return_U=True)
            U_2, grad_U_2 = self.dist2.grad_U(x, return_U=True)
            return U_1 + U_2, grad_U_1 + grad_U_2
        else:
            grad_U_1 = self.dist1.grad_U(x, return_U=False)
            grad_U_2 = self.dist2.grad_U(x, return_U=False)
            return grad_U_1 + grad_U_2

    def hessian_U(self, x, return_grad_U=False):
        """Hessian of the sum of the two potentials

        Args:
            x (torch.Tensor of shape (batch_size, *data_shape)): Particles
            return_grad_U (bool): Whether to compute \nabla U(x) at the same time (default is False)

        Returns:
            if return_U
                grad_U (torch.Tensor of shape (batch_size, *data_shape)) : Gradient of the potential of the system
                hessian_U (torch.Tensor of shape (batch_size, *data_shape, *data_shape)) : Gradient of the potential of the system
            otherwise
                hessian_U (torch.Tensor of shape (batch_size, *data_shape, *data_shape)) : Gradient of the potential of the system
        """

        if return_grad_U:
            grad_U_1, hess_U_1 = self.dist1.hessian_U(x, return_grad_U=True)
            grad_U_2, hess_U_2 = self.dist2.hessian_U(x, return_grad_U=True)
            return grad_U_1 + grad_U_2, hess_U_1 + hess_U_2
        else:
            hess_U_1 = self.dist1.hessian_U(x, return_grad_U=False)
            hess_U_2 = self.dist2.hessian_U(x, return_grad_U=False)
            return hess_U_1 + hess_U_2
