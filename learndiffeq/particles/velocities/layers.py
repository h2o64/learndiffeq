# Custom layers

# Libraries
import torch
from ..utils import box_particles


class AngleEncoding(torch.nn.Module):
    """Angle encoding layer

    Custom layer which transformers a tensor x of shape (..., 1+d) into
    y of shape (1, 1+2*d) where
        y[...,2i] = sin(2*pi*x[...,i]/L)
        y[...,2i+1] = cos(2*pi*x[...,i]/L)
    for i > 1

    This makes sure the data is contained within [-L,L]^d. Note that the first dimension is untouched
    """

    def __init__(self, L):
        """Constructor

        Args:
            L (float): Length of the box
        """

        super(AngleEncoding, self).__init__()
        self.L = L

    def forward(self, x):
        ret = torch.concatenate([
            x[..., 0, None],
            torch.sin(2. * torch.pi * x[..., 1:] / self.L),
            torch.cos(2. * torch.pi * x[..., 1:] / self.L),
        ], dim=-1)
        return ret


class BoxingLayer(torch.nn.Module):
    """Layer which boxes the particles in non-training mode"""

    def __init__(self, L):
        """Constructor

        Args:
            L (float): Length of the box
        """

        super(BoxingLayer, self).__init__()
        self.L = L

    def forward(self, x):
        if self.training:
            return x
        else:
            return box_particles(x, self.L)
