# Learning an velocity field with ODEs

# Libraries
import torch
from functools import reduce
import pytorch_lightning as pl


class LearnBase(pl.LightningModule):

    def __init__(self, data_shape, rho0, use_ot_coupling=False, **kwargs):
        """Constructor for the LearnBase object

        Args:
            data_shape (tuple for int): Shape of the data
            rho0 (torch.distributions.Distribution): base distribution
            optimizer_type (str): Optimizer to use. Either 'adam' (default) or 'radam'
            use_ot_coupling (bool): Whether to use an OT coupling
            kwargs (dict): Argumenrs for velocity field
        """

        super().__init__()
        self.save_hyperparameters()
        # Get the shape of the data
        self.dim = reduce(lambda x, y: x * y, data_shape, 1)
        self.sum_indexes = tuple(range(1, len(data_shape) + 1))
        self.data_shape_ones = tuple([1] * len(data_shape))
        # Get rho0
        self.rho0 = rho0

    def configure_optimizers(self):
        raise NotImplementedError('`configure_optimizers` function is not yet implemented.')

    def _apply_dist(self, dist, fn):
        """Apply the function fn to the distribution dist

        Only a few torch.distribution.Distribution are supported
            - torch.distributions.Normal
            - torch.distributions.MultivariateNormal
            - torch.distributions.VonMises
            - torch.distributions.Uniform
        Otherwise it has to be a torch.nn.Module
        """

        if not isinstance(dist, torch.nn.Module):
            if hasattr(dist, 'base_dist'):
                return self._apply_dist(dist.base_dist, fn)
            else:
                if isinstance(dist, torch.distributions.Normal):
                    dist.loc = fn(dist.loc)
                    dist.scale = fn(dist.scale)
                elif isinstance(dist, torch.distributions.MultivariateNormal):
                    dist.loc = fn(dist.loc)
                    dist._unbroadcasted_scale_tril = fn(
                        dist._unbroadcasted_scale_tril)
                elif isinstance(dist, torch.distributions.VonMises):
                    dist.loc = fn(dist.loc)
                    dist.concentration = fn(dist.concentration)
                    dist._proposal_r = fn(dist._proposal_r)
                elif isinstance(dist, torch.distributions.Uniform):
                    dist.low = fn(dist.low)
                    dist.high = fn(dist.high)
                else:
                    raise "Distribution {} not compatible.".format(str(dist))

    def _apply(self, fn):
        """Apply fn to the LearnBase object"""

        new_self = super(LearnBase, self)._apply(fn)
        new_self._apply_dist(new_self.rho0, fn)
        return new_self

    def on_save_checkpoint(self, checkpoint):
        """Save the base distribution during checkpointing"""
        if not isinstance(self.rho0, torch.nn.Module):
            checkpoint['rho0'] = self.rho0

    def on_load_checkpoint(self, checkpoint):
        """Restore the base distribution during checkpointing"""
        if not isinstance(self.rho0, torch.nn.Module):
            self.rho0 = checkpoint.get('rho0')
