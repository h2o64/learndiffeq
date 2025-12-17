# Circular mixture of Gaussians

# Libraries
import torch


class CircularMixture(torch.nn.Module):
    """Circular mixture of 8 Gaussians

    Taken from https://github.com/lifeitech/fce-2d/blob/main/util.py
    """

    def __init__(self, device, scale=0.7, radius=10.0):
        """Constructor

        Args:
            device (torch.device): Device used for computations
            scale (float): Scale of each Gaussian (default is 0.7)
            radius (float): Radius of the circle (default is 10.0)
        """

        super().__init__()
        self.dummy_param = torch.nn.Parameter(torch.empty(0, device=device))
        locs = torch.stack([
            radius * torch.Tensor([
                torch.tensor(i * torch.pi / 4).sin(),
                torch.tensor(i * torch.pi / 4).cos()
            ]) for i in range(8)
        ]).to(device)
        mix = torch.distributions.Categorical(
            torch.tensor([1 / 8] * 8, device=device).to(device))
        comp = torch.distributions.MultivariateNormal(
            loc=locs,
            covariance_matrix=torch.stack(
                [scale * torch.eye(2, device=device)] * 8)
        )
        self.dist = torch.distributions.MixtureSameFamily(mix, comp)
        self.x_min = float(locs[:, 0].min()) - 5 * scale
        self.x_max = float(locs[:, 0].max()) + 5 * scale
        self.y_min = float(locs[:, 1].min()) - 5 * scale
        self.y_max = float(locs[:, 1].max()) + 5 * scale

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

        new_self = super(CircularMixture, self)._apply(fn)
        new_self.dist.mixture_distribution.probs = fn(
            new_self.dist.mixture_distribution.probs)
        new_self.dist.component_distribution.loc = fn(
            new_self.dist.component_distribution.loc)
        new_self.dist.component_distribution._unbroadcasted_scale_tril = fn(
            new_self.dist.component_distribution._unbroadcasted_scale_tril)
        return new_self
