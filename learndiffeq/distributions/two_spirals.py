# Two spirals

# Libraries
from operator import mul
from functools import reduce
import math
import torch


class TwoSpirals(torch.nn.Module):
    """Two spirals distribution

    Taken from https://github.com/lifeitech/fce-2d/blob/main/util.py
    """

    x_min = -4
    x_max = 4
    y_min = -4
    y_max = 4

    def __init__(self, device):
        """Constructor

        Args:
            device (torch.device): Device used for computations
        """

        super().__init__()
        self.dummy_param = torch.nn.Parameter(torch.empty(0, device=device))

    def sample(self, sample_shape):
        """Sample the distribution

        Args:
            sample_shape (tuple of int): Shape of the samples

        Returns
            samples (torch.Tensor of shape (*sample_shape, 2)): Samples
        """

        n_samples = reduce(mul, sample_shape, 1)
        if n_samples % 2 != 0:
            raise Exception('The size of sample_shape (= {} = {}) should be a multiple of 2.'.format(
                sample_shape, n_samples))
        n = torch.sqrt(torch.rand(
            n_samples // 2, device=self.dummy_param.device)) * 540 * (2 * math.pi) / 360
        d1x = - torch.cos(n) * n + torch.rand(n_samples // 2,
                                              device=self.dummy_param.device) * 0.5
        d1y = torch.sin(n) * n + torch.rand(n_samples // 2,
                                            device=self.dummy_param.device) * 0.5
        x = torch.cat([torch.stack([d1x, d1y], dim=1),
                       torch.stack([-d1x, -d1y], dim=1)], dim=0) / 3
        return (x + 0.1 * torch.randn_like(x)).view((*sample_shape, -1))

    def log_prob(self, value):
        raise NotImplementedError(
            'log_prob is not implemented for two spirals.')
