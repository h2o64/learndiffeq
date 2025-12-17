# Traingular mixture of Gaussians

# Libraries
import math
import torch


class TriangularMixture(torch.nn.Module):
    """
    Traingular mixture of Gaussians from 'Flow Straight and Fast: Learning
    to Generate and Transfer Data with Rectified Flow'
    """

    x_min = -15
    x_max = 15
    y_min = -15
    y_max = 15

    def __init__(self, device, triangle_one=True):
        """Constructor

        Args:
            device (torch.device): Device used for computations
            triangle_one (bool): Whether to represent the upper or lower triangle (default is True)
        """

        super().__init__()
        self.dummy_param = torch.nn.Parameter(torch.empty(0, device=device))
        if triangle_one:
            locs = torch.tensor([
                [10.0 * math.sqrt(3) / 2., 10.0 / 2.],
                [-10.0 * math.sqrt(3) / 2., 10.0 / 2.],
                [0.0, - 10.0 * math.sqrt(3) / 2.]
            ])
        else:
            locs = torch.tensor([
                [10.0 * math.sqrt(3) / 2., - 10.0 / 2.],
                [-10.0 * math.sqrt(3) / 2., - 10.0 / 2.],
                [0.0, 10.0 * math.sqrt(3) / 2.]
            ])
        mix = torch.distributions.Categorical(
            torch.tensor([1 / 3] * 3).to(device))
        comp = torch.distributions.MultivariateNormal(loc=locs.float().to(
            device), covariance_matrix=0.3 * torch.stack([torch.eye(2)] * 3).to(device))
        self.dist = torch.distributions.MixtureSameFamily(mix, comp)

    def sample(self, sample_shape):
        """Sample the distribution

        Args:
            sample_shape (tuple of int): Shape of the samples

        Returns
            samples (torch.Tensor of shape (*sample_shape, 2)): Samples
        """

        return self.dist.sample(sample_shape)

    def log_prob(self, value):
        """Evaluate the log-likelihood of the distribution

        Args:
            value (torch.Tensor of shape (batch_size, 2)): Sample

        Returns
            log_prob (torch.Tensor of shape (batch_size,)): Log-likelihood of the samples
        """

        return self.dist.log_prob(value)

    def _apply(self, fn):
        """Apply the fn function on the distribution

        Args:
            fn (function): Function to apply on tensors
        """

        new_self = super(TriangularMixture, self)._apply(fn)
        new_self.dist.mixture_distribution.probs = fn(
            new_self.dist.mixture_distribution.probs)
        new_self.dist.component_distribution.loc = fn(
            new_self.dist.component_distribution.loc)
        new_self.dist.component_distribution._unbroadcasted_scale_tril = fn(
            new_self.dist.component_distribution._unbroadcasted_scale_tril)
        return new_self
