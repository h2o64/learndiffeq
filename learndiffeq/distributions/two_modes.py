# Circular mixture of Gaussians

# Libraries
import torch


class TwoModes(torch.nn.Module):
    """Mixture of two Gaussian"""

    def __init__(self, device, dim, a=1.0, centered=True, equilibrated=False):
        """Constructor

        Args:
            device (torch.device): Device used for computations
            dim (int): Dimension
            a (float): Distance between modes (default is 1.0)
            centered (bool): Whether to center the distribution (default is True)
            equilibrated (bool): Whether to equilibrate the models (default is False)
        """

        super().__init__()
        # Make the means
        means = torch.stack([
            -a * torch.ones((dim,), device=device),
            a * torch.ones((dim,), device=device)
        ])
        if centered:
            means += (a/3.) * torch.ones((dim,), device=device)
        # Make the covariances
        covs = torch.stack([0.05 * torch.eye(dim, device=device)] * 2)
        # Make the weights
        if equilibrated:
            weights = torch.ones((2,), device=device)
        else:
            weights = torch.FloatTensor([2, 1]).to(device)
        # Make the distribution
        mix = torch.distributions.Categorical(weights)
        comp = torch.distributions.MultivariateNormal(loc=means, covariance_matrix=covs)
        self.dist = torch.distributions.MixtureSameFamily(mix, comp)
        mean = self.dist.mean.cpu()
        scale = self.dist.stddev.cpu()
        self.x_min = float(mean[0] - 2 * scale[0])
        self.x_max = float(mean[0] + 2 * scale[0])
        self.y_min = float(mean[1] - 2 * scale[1])
        self.y_max = float(mean[1] + 2 * scale[1])

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

        new_self = super(TwoModes, self)._apply(fn)
        new_self.dist.mixture_distribution.probs = fn(
            new_self.dist.mixture_distribution.probs)
        new_self.dist.component_distribution.loc = fn(
            new_self.dist.component_distribution.loc)
        new_self.dist.component_distribution._unbroadcasted_scale_tril = fn(
            new_self.dist.component_distribution._unbroadcasted_scale_tril)
        return new_self
