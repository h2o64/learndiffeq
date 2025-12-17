# Two moons in Pytorch fashion

# Libraries
from .utils import PolarTransform
import math
import torch


class TwoMoons(torch.nn.Module):
    """Two moonths distribution"""

    def __init__(self, device, width=0.05, prop=0.5, validate_args=False):
        """Constructor

        Args:
            device (torch.device): Device used for computations
            width (float): Width of each moon (default is 0.05)
            prop (float): Relative weights of the moons (default is 0.5)
        """

        super().__init__()
        self.dummy_param = torch.nn.Parameter(torch.empty(0, device=device))
        # Base distribution
        base_distribution = torch.distributions.Normal(
            loc=torch.tensor([1.0, torch.pi / 2]).to(device),
            scale=torch.tensor([width, torch.pi / 6]).to(device),
        )
        # Half circle 1
        self.circle_dist1 = torch.distributions.TransformedDistribution(
            base_distribution=base_distribution,
            transforms=PolarTransform())
        # Half circle 2
        self.circle_dist2 = torch.distributions.TransformedDistribution(
            base_distribution=base_distribution,
            transforms=[
                PolarTransform(),
                torch.distributions.transforms.AffineTransform(
                    loc=torch.tensor([1.0, 0.3]).to(device),
                    scale=torch.tensor([1.0, -1.0]).to(device)
                )
            ])
        # Categorical distribution
        self.prop = prop
        # Set the extreme values
        self.x_min = -3
        self.x_max = 3
        self.y_min = -2
        self.y_max = 2

    def sample(self, sample_shape=torch.Size()):
        """Sample the distribution

        Args:
            sample_shape (tuple of int): Shape of the samples

        Returns
            samples (torch.Tensor of shape (*sample_shape, 2)): Samples
        """

        # Sample categories
        categories = torch.rand(*sample_shape).flatten() <= self.prop
        # Sample the moons
        n_moon2 = int(categories.sum())
        n_moon1 = categories.shape[0] - n_moon2
        moon1_samples = self.circle_dist1.sample(sample_shape=(n_moon1,))
        moon2_samples = self.circle_dist2.sample(sample_shape=(n_moon2,))
        ret = torch.zeros((categories.shape[0], 2)).to(moon1_samples.device)
        ret[categories.bool()] = moon2_samples
        ret[~categories.bool()] = moon1_samples
        return ret.reshape((*sample_shape, -1))

    def log_prob(self, value):
        """Evaluate the log-likelihood of the distribution

        Args:
            value (torch.Tensor of shape (batch_size, 2)): Sample

        Returns
            log_prob (torch.Tensor of shape (batch_size,)): Log-likelihood of the samples
        """

        return torch.logsumexp(torch.stack([
            math.log(self.prop) + self.circle_dist1.log_prob(value),
            math.log(1.0 - self.prop) + self.circle_dist2.log_prob(value)
        ], dim=-1), dim=-1)

    def _apply(self, fn):
        """Apply the fn function on the distribution

        Args:
            fn (function): Function to apply on tensors
        """

        new_self = super(TwoMoons, self)._apply(fn)
        new_self.circle_dist1.base_dist.base_dist.loc = fn(
            new_self.circle_dist1.base_dist.base_dist.loc)
        new_self.circle_dist1.base_dist.base_dist.scale = fn(
            new_self.circle_dist1.base_dist.base_dist.scale)
        new_self.circle_dist2.base_dist.base_dist.loc = fn(
            new_self.circle_dist2.base_dist.base_dist.loc)
        new_self.circle_dist2.base_dist.base_dist.scale = fn(
            new_self.circle_dist2.base_dist.base_dist.scale)
        new_self.circle_dist2.transforms[1].loc = fn(
            new_self.circle_dist2.transforms[1].loc)
        new_self.circle_dist2.transforms[1].scale = fn(
            new_self.circle_dist2.transforms[1].scale)
        return new_self
