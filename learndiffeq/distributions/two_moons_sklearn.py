# Two moons (from scikit-learn)

# Libraries
import torch


def generate_moons(n_samples, device, noise):
    """Generate two moons

    Adapted from https://github.com/scikit-learn/scikit-learn/blob/364c77e04/sklearn/datasets/_samples_generator.py
    """

    n_samples_out = n_samples // 2
    n_samples_in = n_samples - n_samples_out
    ls_in = torch.linspace(0, torch.pi, n_samples_in, device=device)
    ls_out = torch.linspace(0, torch.pi, n_samples_out, device=device)
    outer_circ_x = torch.cos(ls_out)
    outer_circ_y = torch.sin(ls_out)
    inner_circ_x = 1 - torch.cos(ls_in)
    inner_circ_y = 1 - torch.sin(ls_in) - .5
    X = torch.stack([torch.concatenate([outer_circ_x, inner_circ_x]),
                     torch.concatenate([outer_circ_y, inner_circ_y])]).T
    if noise is not None:
        X += torch.rand((n_samples, 1), device=device) * noise
    return X


class TwoMoonsSklearn(torch.nn.Module):
    """Two moons distribution from scikit-learn"""

    x_min = -1
    x_max = 2
    y_min = -0.5
    y_max = 1

    def __init__(self, device, noise=1e-4, affine=None):
        """Constructor

        Args:
            device (torch.device): Device used for computations
            noise (float): Noise level used (default is 1e-4)
            affine (tuple of float): Apply an affine shift
        """

        super().__init__()
        self.dummy_param = torch.nn.Parameter(torch.empty(0, device=device))
        self.noise = noise
        self.affine = affine

    def sample(self, sample_shape=torch.Size()):
        """Sample the distribution

        Args:
            sample_shape (tuple of int): Shape of the samples

        Returns
            samples (torch.Tensor of shape (*sample_shape, 2)): Samples
        """

        ret = generate_moons(
            n_samples=sample_shape[0], device=self.dummy_param.device, noise=self.noise)
        if self.affine:
            ret = self.affine[0] * ret + self.affine[1]
        return ret

    def log_prob(self, value):
        raise NotImplementedError(
            'log_prob is not implemented for TwoMoonsSklearn')
