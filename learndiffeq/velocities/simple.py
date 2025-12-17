# Libraries
import torch
import math
from .layers import SirenLayer, ResidualLayer
from ..particles.velocities.layers import AngleEncoding, BoxingLayer


@torch.no_grad()
def init_weights(m, full_zero=False):
    """Custom initialisation from "Batch Normalization: Accelerating Deep Network Training
    by Reducing Internal Covariate Shift" (arXiv:1502.03167)"""

    if isinstance(m, torch.nn.Linear):
        fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(m.weight)
        if full_zero:
            torch.nn.init.normal_(
                m.weight,
                std=1e-6 / math.sqrt(fan_in),
            )
        else:
            torch.nn.init.trunc_normal_(
                m.weight,
                std=1 / fan_in,
            )
        torch.nn.init.zeros_(m.bias)


class SimpleTimeNetworkWrapper(torch.nn.Module):
    """Simple time wrapper for nn.Module

    Simply stack t and x in a MLP.
    """

    def __init__(self, net, full_zero=False):
        """Constructor

        Args:
            net (torch.nn.Module): Base MLP (or more advanced)
            full_zero (bool): Initialize weights with zeros (default is False)
        """

        super(SimpleTimeNetworkWrapper, self).__init__()
        self.net = net
        self.net.apply(lambda m: init_weights(m, full_zero=full_zero))

    def forward(self, t, x):
        if len(t.size()) == 0:
            t = t.view((1,) * len(x.shape))
            t = t.expand((*x.shape[:-1], -1))
        return self.net(torch.concat([t, x], dim=-1))


def make_mlp(
        input_dim,
        output_dim,
        hidden_layers,
        activation=torch.nn.SiLU,
        L=None,
        w0=-1.,
        use_residual_layers=False,
        use_boxing_layer=False):
    """Make an MLP

    Args:
        input_dim (int): Input dimension
        output_dim (int): Output dimension
        hidden_layers (tuple or list of int): Size of each layer
        activation (function): Activation function (default is torch.nn.SiLU)
        L (float): Length of the box (if not None which is the default, append AngleEncoding layer)
        w0 (float): Scaling factor for Siren's layers (-1. is default and means Siren's layers are disabled)
        use_residual_layers (bool): Whether to use residual layers (default is False)
        use_boxing_layer (bool): Add a boxing layer with L is not None (default is False)
    """

    # Make the sequence of sizes
    layers = [input_dim] + hidden_layers + [output_dim]
    ret = []
    # Use an angular encoding
    if L is not None:
        if use_boxing_layer:
            ret.append(BoxingLayer(L=L))
        else:
            ret.append(AngleEncoding(L=L))
            layers[0] = 2 * (input_dim - 1) + 1
    # Build the sequence
    if w0 is not None and w0 > 0:
        for i in range(len(layers) - 2):
            m = SirenLayer(layers[i], layers[i + 1], w0_scaling=(i != 0), w0=w0)
            if use_residual_layers:
                m = ResidualLayer(m)
            ret.append(m)
        ret.append(torch.nn.Linear(layers[-2], layers[-1]))
    else:
        for i in range(len(layers) - 1):
            ret.append(torch.nn.Linear(layers[i], layers[i + 1]))
            if i < len(layers) - 2:
                a = activation()
                if use_residual_layers:
                    a = ResidualLayer(a)
                ret.append(a)
    return torch.nn.Sequential(*ret)


def make_simple_velocity(
        dim,
        hidden_layers,
        activation=torch.nn.SiLU,
        full_zero=False,
        L=None,
        w0=-1.,
        use_residual_layers=False):
    """Make an simple velocity field using an MLP

    Args:
        dim (int): Dimension of the data
        hidden_layers (tuple or list of int): Size of each layer
        activation (function): Activation function (default is torch.nn.SiLU)
        full_zero (bool): Initialize weights with zeros (default is False)
        L (float): Length of the box (if not None which is the default, append AngleEncoding layer)
        w0 (float): Scaling factor for Siren's layers (-1. is default and means Siren's layers are disabled)
        use_residual_layers (bool): Whether to use residual layers (default is False)
    """

    return SimpleTimeNetworkWrapper(make_mlp(dim + 1, dim, hidden_layers=hidden_layers, activation=activation, L=L,
                                             w0=w0, use_residual_layers=use_residual_layers), full_zero=full_zero)
