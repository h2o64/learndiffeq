# Implementation of Flow Matching with Species

# Libraries
import torch
from ..common.learn_ode import LearnODE
from ..common.makers import make_velocity
from ..common.utils import torch_optimizers
from ..particles.utils import ot_permutation, linear_assignment_permutation
from ..particles.utils import separate_by_species, rebuild_from_index

class FlowMatchingWithSpecies(LearnODE):

    def __init__(
            self,
            n_particles,
            dim_phys,
            rho0,
            lr,
            velocity_type,
            velocity_kwargs=None,
            optimizer_type='adam',
            ot_particles=False,
            use_linear_assignment_particles=False,
            force_automatic_div=False,
            skip_sorting=False,
            **kwargs):
        """Constructor for the RiemannianFlowMatching object

        Args:
            n_particles (int): Number of particles
            dim_phys (int): Dimension of the particles
            L (float): Size of the torus
            rho0 (torch.distributions.Distribution): Base distribution
            lr (float): Learning rate
            velocity_type (str): Type of velocities given to make_velocity (eg. 'mlp,128,128,128')
            velocity_kwargs (dict): Arguments for the velocity constructor
            optimizer_type (str): Type of optimizer (default is 'adam')
            ot_particles (bool): Use OT pairing between rho0 and rho1 at each training step (default is False)
            use_linear_assignment_particles (bool): Whether to use the hungarian algorithm (default is False)
            force_automatic_div (bool): Whether to skip self.compute_div (default is False)
            skip_sorting (bool): Whether to skip the sorting part in the training/validation loops (default is False)
            kwargs (dict): Other arguments to pass to the LearnBase constructor
        """

        # Call the parent constructor
        super().__init__(
            n_particles=n_particles,
            dim_phys=dim_phys,
            data_shape=(n_particles, dim_phys),
            rho0=rho0,
            lr=lr,
            velocity_type=velocity_type,
            velocity_kwargs=velocity_kwargs,
            optimizer_type=optimizer_type,
            ot_particles=ot_particles,
            force_automatic_div=force_automatic_div,
            skip_sorting=skip_sorting
            # **kwargs
        )

        # Build the velocity field
        if velocity_kwargs is None:
            velocity_kwargs = {}
        self.b = make_velocity(
            self.hparams.data_shape,
            self.dim,
            self.hparams.velocity_type,
            **velocity_kwargs
        )
        self.n_species = None

    def training_step(self, batch, batch_idx):
        # Get samples from rho1
        x1, a1 = batch
        batch_size = x1.shape[0]
        # Sample rho0
        ret0 = self.rho0.sample(sample_shape=(batch_size,))
        if isinstance(ret0, tuple):
            # Extract a0 and x0
            a0, x0 = ret0
            per_species_ot = True
            # Sort the species
            if not self.hparams.skip_sorting:
                idx_sorted0 = a0.argsort(dim=-1)
                a0 = torch.gather(a0, 1, idx_sorted0)
                x0 = torch.gather(x0, 1, idx_sorted0.unsqueeze(-1).expand((-1, -1, x0.shape[-1])))
                idx_sorted1 = a1.argsort(dim=-1)
                a1 = torch.gather(a1, 1, idx_sorted1)
                x1 = torch.gather(x1, 1, idx_sorted1.unsqueeze(-1).expand((-1, -1, x1.shape[-1])))
            # Check the composition
            is_same_composition = torch.equal(a0, a1)
            if not is_same_composition:
                raise ValueError('a0 and a1 need to have the same composition.')
            # Get the number of species
            if self.n_species is None:
                self.n_species = torch.unique(a0[0]).shape[0]
        else:
            # Extract a0 and set x0
            a0, x0 = a1, ret0
            per_species_ot = False
        # Apply OT depending on the base distribution
        if per_species_ot:
            # Apply OT for each specie group
            indexes0 = separate_by_species(a0, self.n_species, x0.shape[-1])
            indexes1 = separate_by_species(a1, self.n_species, x1.shape[-1])
            if self.hparams.use_linear_assignment_particles:
                x0 = torch.concat([
                    linear_assignment_permutation(
                        rebuild_from_index(i, indexes0, x0),
                        rebuild_from_index(i, indexes1, x1),
                        L=None
                    )[0] for i in range(self.n_species)
                ], dim=1)
            elif self.hparams.ot_particles:
                x0 = torch.concat([
                    ot_permutation(
                        rebuild_from_index(i, indexes0, x0),
                        rebuild_from_index(i, indexes1, x1),
                        L=None
                    )[0] for i in range(self.n_species)
                ], dim=1)
        else:
            if self.hparams.use_linear_assignment_particles:
                x0, _ = linear_assignment_permutation(x0, x1, L=None)
            elif self.hparams.ot_particles:
                x0, _ = ot_permutation(x0, x1, L=self.hparams.get('L', None))
        # Sample t uniformly
        t = torch.rand((batch_size, *self.data_shape_ones), device=self.device)
        # Compute x_t
        x_t =  t * x1 + (1. - t) * x0
        # Compute the loss
        loss = torch.sum(torch.square(self.b(t, x_t, a=a1) - (x1 - x0)),
                         dim=self.sum_indexes) / self.dim
        # Average the loss
        loss = loss.mean()
        self.log("loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # Get samples from rho1
        x1, a1 = batch
        batch_size = x1.shape[0]
        # Sample rho0
        ret0 = self.rho0.sample(sample_shape=(batch_size,))
        if isinstance(ret0, tuple):
            # Extract a0 and x0
            a0, x0 = ret0
            per_species_ot = True
            # Sort the species
            if not self.hparams.skip_sorting:
                idx_sorted0 = a0.argsort(dim=-1)
                a0 = torch.gather(a0, 1, idx_sorted0)
                x0 = torch.gather(x0, 1, idx_sorted0.unsqueeze(-1).expand((-1, -1, x0.shape[-1])))
                idx_sorted1 = a1.argsort(dim=-1)
                a1 = torch.gather(a1, 1, idx_sorted1)
                x1 = torch.gather(x1, 1, idx_sorted1.unsqueeze(-1).expand((-1, -1, x1.shape[-1])))
            # Check the composition
            is_same_composition = torch.equal(a0, a1)
            if not is_same_composition:
                raise ValueError('a0 and a1 need to have the same composition.')
            # Get the number of species
            if self.n_species is None:
                self.n_species = torch.unique(a0[0]).shape[0]
        else:
            # Extract a0 and set x0
            a0, x0 = a1, ret0
            per_species_ot = False
        # Apply OT depending on the base distribution
        if per_species_ot:
            # Apply OT for each specie group
            indexes0 = separate_by_species(a0, self.n_species, x0.shape[-1])
            indexes1 = separate_by_species(a1, self.n_species, x1.shape[-1])
            if self.hparams.use_linear_assignment_particles:
                x0 = torch.concat([
                    linear_assignment_permutation(
                        rebuild_from_index(i, indexes0, x0),
                        rebuild_from_index(i, indexes1, x1),
                        L=None
                    )[0] for i in range(self.n_species)
                ], dim=1)
            elif self.hparams.ot_particles:
                x0 = torch.concat([
                    ot_permutation(
                        rebuild_from_index(i, indexes0, x0),
                        rebuild_from_index(i, indexes1, x1),
                        L=None
                    )[0] for i in range(self.n_species)
                ], dim=1)
        else:
            if self.hparams.use_linear_assignment_particles:
                x0, _ = linear_assignment_permutation(x0, x1, L=None)
            elif self.hparams.ot_particles:
                x0, _ = ot_permutation(x0, x1, L=self.hparams.get('L', None))
        # Sample t uniformly
        t = torch.rand((batch_size, *self.data_shape_ones), device=self.device)
        # Compute x_t
        x_t =  t * x1 + (1. - t) * x0
        # Compute the loss
        loss = torch.sum(torch.square(self.b(t, x_t, a=a1) - (x1 - x0)),
                         dim=self.sum_indexes) / self.dim
        # Average the loss
        loss = loss.mean()
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optim_kwargs = {'lr': self.hparams.lr}
        if 'weight_decay' in self.hparams:
            optim_kwargs['weight_decay'] = self.hparams.weight_decay
        optimizer = torch_optimizers[self.hparams.optimizer_type](self.b.parameters(), **optim_kwargs)
        return optimizer

    def sample(self, a, values, **kwargs):
        """Sample the associated ODE

        Args:
                a (torch.Tensor of shape (batch_size, n_particles)): Initial species
            values (torch.Tensor of shape (batch_size, n_particles, dim_phys)): Initial values
                kwargs (dict): Argument to pass to sample

        Return:
            if keep_intermediates:
                        a (torch.Tensor of shape (n_steps, batch_size, n_particles)): Initial species
                samples (torch.Tensor of shape (n_steps, batch_size, n_particles, dim_phys)): intermediate samples
                if return_log_jac
                    log_jac (torch.Tensor of shape (n_steps, batch_size)): log-jacobian
            else:
                        a (torch.Tensor of shape (batch_size, n_particles)): Initial species
                samples (torch.Tensor of shape (batch_size, n_particles, dim_phys)): intermediate samples
                if return_log_jac
                    log_jac (torch.Tensor of shape (batch_size,)): log-jacobian
        """
        ret = super().sample(
            values,
            b_kwargs={'a': a},
            **kwargs
        )
        if kwargs.get('return_log_jac', False):
            if kwargs.get('keep_intermediates', True):
                a = a.unsqueeze(0).expand((ret[0].shape[0], -1, -1))
            ret = (a, ret[0], ret[1])
        else:
            if kwargs.get('keep_intermediates', True):
                a = a.unsqueeze(0).expand((ret.shape[0], -1, -1))
            ret = a, ret
        return ret

    def forward(self, a, z, **kwargs):
        """Forward call to the normalizing flow

        Args:
            a (torch.Tensor of shape (batch_size, n_particules)): Species
            z (torch.Tensor of shape (batch_size, n_particles, dim_phys)): Samples
            kwargs (dict): Arguments to pass to forward

        Return:
            a (torch.Tensor of shape (batch_size, n_particules)): Species
            x (torch.Tensor of shape (batch_size, *data_shape)): Samples
            if return_log_jac
                log_jac (torch.Tensor of shape (batch_size,)): log-jacobian
        """

        ret = self.sample(a, z, **kwargs)
        return ret

    def inverse(self, a, x, **kwargs):
        """Inverse call to the normalizing flow

        Args:
            a (torch.Tensor of shape (batch_size, n_particules)): Species
            x (torch.Tensor of shape (batch_size, n_particles, dim_phys)): Samples
            kwargs (dict): Arguments to pass to inverse

        Return:
            a (torch.Tensor of shape (batch_size, n_particules)): Species
            z (torch.Tensor of shape (batch_size, n_particles, dim_phys)): Samples
            if return_log_jac
                log_jac (torch.Tensor of shape (batch_size,)): log-jacobian
        """

        ret = self.sample(a, x, reverse_time=True, **kwargs)
        return ret
    
    def log_prob(self, a, x, keep_b_parameters=False, **kwargs):
        """Sample the normalizing flow and return the likelihood

        Args:
            x (torch.Tensor of shape (batch_size, *data_shape)): Samples
            a (torch.Tensor of shape (batch_size, n_particles)): Reference species (default is None)
            kwargs (dict): Argument to pass to sample_and_log_prob

        Return:
            log_prob (torch.Tensor of shape (batch_size)): Log-likelihood
        """
        z, log_jac = self.inverse(a, x, return_log_jac=True, keep_intermediates=False, keep_b_parameters=keep_b_parameters, **kwargs)[-2:]
        ret = self.rho0.log_prob(a, z)
        return ret + log_jac.flatten()

    def sample_and_log_prob(self, sample_shape, **kwargs):
        """Sample the normalizing flow and return the likelihood

        Args:
            sample_shape (tuple of int): Number of samples
            kwargs (dict): Argument to pass to sample_and_log_prob

        Return:
            samples (torch.Tensor of shape (batch_size, n_particles, dim_phys)): Samples
            log_prob (torch.Tensor of shape (batch_size)): Log-likelihood
        """

        ret0 = self.rho0.sample(sample_shape)
        if not isinstance(ret0, tuple):
            raise ValueError('rho0 should return both a and X')
        else:
            a0_samples, x0_samples = ret0
        a_samples, x_samples, log_jac = self.sample(
            a0_samples,
            x0_samples,
            reverse_time=False,
            return_log_jac=True,
            keep_intermediates=False,
            **kwargs
        )
        log_prob = self.rho0.log_prob(a0_samples, x0_samples) + log_jac.flatten()
        return a_samples, x_samples, log_prob