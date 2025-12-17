# Pinwheel

# Libraries
from operator import mul
from functools import reduce
import math
import torch


class Pinwheel(torch.nn.Module):
    """Pinwheel distribution

    Taken from https://github.com/lifeitech/fce-2d/blob/main/util.py
    """

    x_min = -4
    x_max = 4
    y_min = -4
    y_max = 4

    def __init__(self, device, radial_std=0.3, tangential_std=0.1, num_classes=5, rate=0.25):
        """Constructor

        Args:
            device (torch.device): Device used for computations
            radial_std (float): Standard deviation of the radial normal disribution (default is 0.3)
            tangential_std (float): Standard deviation of the tangantial normal disribution (default is 0.1)
            num_classes (int): Number of modes (default is 5)
            rate (float): Decay rate of the spirals (default is 0.25)
        """

        super().__init__()
        self.device = device
        self.dummy_param = torch.nn.Parameter(torch.empty(0, device=device))
        self.radial_std = radial_std
        self.tangential_std = tangential_std
        self.num_classes = num_classes
        self.rate = rate

    def sample(self, sample_shape):
        """Sample the distribution

        Args:
            sample_shape (tuple of int): Shape of the samples

        Returns
            samples (torch.Tensor of shape (*sample_shape, 2)): Samples
        """

        n_samples = reduce(mul, sample_shape, 1)
        if n_samples % self.num_classes != 0:
            raise Exception('The size of sample_shape (= {} = {}) should be a multiple of num_classes (= {}).'.format(
                sample_shape, n_samples, self.num_classes))
        num_per_class = n_samples // self.num_classes
        rads = torch.linspace(
            0, 2 * math.pi, self.num_classes + 1)[:-1].to(self.dummy_param.device)
        features = torch.randn(
            (self.num_classes * num_per_class, 2), device=self.dummy_param.device)
        features *= torch.tensor([self.radial_std,
                                 self.tangential_std]).to(self.dummy_param.device)
        features[:, 0] += 1.0
        labels = torch.stack([torch.arange(self.num_classes)]
                             * num_per_class, dim=1).flatten()
        angles = rads[labels] + self.rate * torch.exp(features[:, 0])
        rotations = torch.stack(
            [torch.cos(angles), -torch.sin(angles), torch.sin(angles), torch.cos(angles)])
        rotations = torch.reshape(rotations.T, (-1, 2, 2))
        data = 2 * torch.einsum("ti,tij->tj", features, rotations)
        data = data[torch.randperm(data.shape[0])]
        return data.view((*sample_shape, -1))

    def log_prob(self, value):
        raise NotImplementedError('log_prob is not implemented for pinwheel.')
