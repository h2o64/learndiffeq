# Checkboard distribution

# Libraries
import torch


class Checkerboard(torch.nn.Module):
    """Checkerboard distribution

    Taken from https://github.com/lifeitech/fce-2d/blob/main/util.py
    """

    def __init__(self, device, width=4):
        """Constructor

        Args:
            device (torch.device): Device used for computations
            width (int): Number of squares (default is 4)
        """

        super().__init__()
        self.device = device
        self.dummy_param = torch.nn.Parameter(torch.empty(0, device=device))
        self.width = width
        self.x_min = -4
        self.x_max = (self.width - 2) * 2
        self.y_min = -4
        self.y_max = 4

    def sample(self, sample_shape):
        """Sample the distribution

        Args:
            sample_shape (tuple of int): Shape of the samples

        Returns
            samples (torch.Tensor of shape (*sample_shape, 2)): Samples
        """

        x1 = torch.rand(
            sample_shape, device=self.dummy_param.device) * self.width - 2
        x2_ = torch.rand(sample_shape, device=self.dummy_param.device)
        x2_ -= torch.randint(0, 2, (*sample_shape,),
                             device=self.dummy_param.device) * 2
        x2 = x2_ + x1.floor() % 2
        return torch.stack([x1, x2], dim=1) * 2

    def log_prob(self, value):
        raise NotImplementedError(
            'log_prob is not implemented for Checkerboard.')
