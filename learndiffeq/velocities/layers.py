# Custom layers

# Libraries
import torch
import math


class SirenLayer(torch.nn.Linear):
    """Siren layer

    This is a Linear layer with a sinusoidal activation function x -> sin(w0 * x)

    Introduced in in "Implicit Neural Representations with Periodic Activation Functions" (arXiv:2006.09661)
    """

    def __init__(self, in_features, out_features, w0=30.0, w0_scaling=False, device=None, dtype=None):
        self.w0 = w0
        self.w0_scaling = w0_scaling
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            bias=True,
            device=device,
            dtype=dtype
        )

    def reset_parameters(self):
        if self.w0_scaling:
            torch.nn.init.kaiming_uniform_(
                self.weight, a=math.sqrt(self.w0 - 1.))
        else:
            torch.nn.init.kaiming_uniform_(self.weight, a=0.)
        torch.nn.init.zeros_(self.bias)

    def forward(self, input):
        return torch.sin(self.w0 * torch.nn.functional.linear(input, self.weight, self.bias))


class ResidualLayer(torch.nn.Module):
    """Residual layer

    Given a layer m, this module defines m' such that

        m'(x) = m(x) + x

    """

    def __init__(self, module):
        """Constructor

        Args:
            module (torch.nn.Module): Module m
        """

        super(ResidualLayer, self).__init__()
        self.module = module

    def forward(self, x):
        ret = self.module(x)
        if x.shape == ret.shape:
            return ret + x
        else:
            return ret
