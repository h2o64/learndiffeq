# Implementation of Flow Matching

# Libraries
import torch
from enum import Enum
from ..common.learn_ode import LearnODE
from ..common.makers import make_velocity
from ..common.utils import sample_ot_coupling, torch_optimizers
from ..particles.utils import ot_permutation, linear_assignment_permutation, superpose_points


class InterpolationPathType(Enum):
    """Enumerate for the interpolation path type"""
    LINEAR = 1

    @classmethod
    def from_str(cls, s):
        return cls[s.upper()]


class InterpolationPath:

    def __init__(self, interpolation_type, sigma):
        self.interpolation_type = InterpolationPathType.from_str(interpolation_type)
        self.sigma_ = sigma

    def mu(self, t, x0, x1):
        if self.interpolation_type == InterpolationPathType.LINEAR:
            return t * x1 + (1. - t) * x0
        else:
            raise NotImplementedError(
                "Interpolation path type {} not found.".format(self.interpolation_type))

    def mu_dot(self, t, x0, x1):
        if self.interpolation_type == InterpolationPathType.LINEAR:
            return x1 - x0
        else:
            raise NotImplementedError(
                "Interpolation path type {} not found.".format(self.interpolation_type))

    def sigma(self, t):
        if self.interpolation_type == InterpolationPathType.LINEAR:
            return self.sigma_
        else:
            raise NotImplementedError(
                "Interpolation path type {} not found.".format(self.interpolation_type))

    def sigma_dot(self, t):
        if self.interpolation_type == InterpolationPathType.LINEAR:
            return 0.0
        else:
            raise NotImplementedError(
                "Interpolation path type {} not found.".format(self.interpolation_type))

    def u(self, x, t, x0, x1):
        ret = self.sigma_dot(t) * (x - self.mu(t, x0, x1)) / self.sigma(t)
        ret += self.mu_dot(t, x0, x1)
        return ret


class FlowMatching(LearnODE):

    def __init__(
            self,
            data_shape,
            rho0,
            lr,
            velocity_type,
            velocity_kwargs=None,
            interpolation_type='linear',
            sigma=1e-3,
            optimizer_type='adam',
            ot_particles=False,
            use_linear_assignment_particles=False,
            use_superpose_points_particles=False,
            use_ot_coupling=False,
            use_pre_computed_ot=False,
            antithetic=True,
            force_automatic_div=False,
            **kwargs):
        """Constructor for the FlowMatching object

        Args:
            data_shape (tuple for int): Shape of the data
            rho0 (torch.distributions.Distribution): Base distribution
            lr (float): Learning rate
            velocity_type (str): Type of velocities given to make_velocity (eg. 'mlp,128,128,128')
            velocity_kwargs (dict): Arguments for the velocity constructor
            interpolation_type (str): Type of interpolant (default is 'linear')
            sigma (float): Value of sigma (default is 1e-3)
            optimizer_type (str): Type of optimizer (default is 'adam')
            ot_particles (bool): Use OT pairing between rho0 and rho1 at each training step (default is False)
            use_linear_assignment_particles (bool): Whether to use the hungarian algorithm (default is False)
            use_superpose_points_particles (bool): Whether to use the point superposition (default is False)
            use_ot_coupling (bool): Whether to use an OT coupling (default is False)
            use_pre_computed_ot (bool): Use a precomputed pairing from the dataset (default is False)
            antithetic (bool): Whether to use the anthitetic trick when the model leverages normal samples (default is True)
            force_automatic_div (bool): Whether to skip self.compute_div (default is False)
            kwargs (dict): Other arguments to pass to the LearnBase constructor
        """

        # Call the parent constructor
        super().__init__(
            data_shape=data_shape,
            rho0=rho0,
            lr=lr,
            velocity_type=velocity_type,
            velocity_kwargs=velocity_kwargs,
            interpolation_type=interpolation_type,
            sigma=sigma,
            optimizer_type=optimizer_type,
            ot_particles=ot_particles,
            use_ot_coupling=use_ot_coupling,
            use_pre_computed_ot=use_pre_computed_ot,
            antithetic=antithetic,
            force_automatic_div=force_automatic_div,
            **kwargs
        )

        # Instantiate the interpolation path
        self.interpolation_path = InterpolationPath(
            interpolation_type=self.hparams.interpolation_type,
            sigma=self.hparams.sigma
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

    def training_step(self, batch, batch_idx):
        # Use precomputed OT
        if self.hparams.use_pre_computed_ot:
            # Get samples from rho0 and rho1 which were matched
            x0, x1 = batch
        else:
            # Get samples from rho1
            x1, _ = batch
            # Sample rho0
            x0 = self.rho0.sample(sample_shape=(x1.shape[0],))
            # Apply OT
            if self.hparams.use_linear_assignment_particles:
                x0, _ = linear_assignment_permutation(x0, x1, L=self.hparams.get('L', None))
            elif self.hparams.ot_particles:
                x0, _ = ot_permutation(x0, x1, L=self.hparams.get('L', None))
            elif self.hparams.use_ot_coupling:
                x0, x1 = sample_ot_coupling(x0.view((-1, self.dim)), x1.view((-1, self.dim)))
                x0, x1 = x0.view((-1, *self.hparams.data_shape)), x1.view((-1, *self.hparams.data_shape))
            # Superpose particles
            if self.hparams.use_superpose_points_particles:
                x0, _ = superpose_points(x0, x1)
        # Sample t uniformly
        t = torch.rand(
            (x1.shape[0], *self.data_shape_ones), device=self.device)
        # Sample a noise
        z = torch.randn_like(x1)
        # Compute x_t
        x_t = self.interpolation_path.mu(t, x0, x1) + self.interpolation_path.sigma(t) * z
        # Compute the loss
        loss = torch.sum(torch.square(self.b(t, x_t) - self.interpolation_path.u(x_t, t, x0, x1)),
                         dim=self.sum_indexes) / self.dim
        # Apply the anthithetic trick
        if self.hparams.antithetic:
            # Compute x_t_antithetic
            x_t_antithetic = self.interpolation_path.mu(t, x0, x1) - self.interpolation_path.sigma(t) * z
            # Compute the loss
            loss_antithetic = torch.sum(torch.square(self.b(t, x_t) - self.interpolation_path.u(x_t_antithetic, t, x0, x1)),
                                        dim=self.sum_indexes) / self.dim
            # Sum to the main loss
            loss += loss_antithetic
        # Average the loss
        loss = loss.mean()
        self.log("loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # Use precomputed OT
        if self.hparams.use_pre_computed_ot:
            # Get samples from rho0 and rho1 which were matched
            x0, x1 = batch
        else:
            # Get samples from rho1
            x1, _ = batch
            # Sample rho0
            x0 = self.rho0.sample(sample_shape=(x1.shape[0],))
            # Apply OT
            if self.hparams.use_linear_assignment_particles:
                x0, _ = linear_assignment_permutation(x0, x1, L=self.hparams.get('L', None))
            elif self.hparams.ot_particles:
                x0, _ = ot_permutation(x0, x1, L=self.hparams.get('L', None))
            elif self.hparams.use_ot_coupling:
                x0, x1 = sample_ot_coupling(x0.view((-1, self.dim)), x1.view((-1, self.dim)))
                x0, x1 = x0.view((-1, *self.hparams.data_shape)), x1.view((-1, *self.hparams.data_shape))
            # Superpose particles
            if self.hparams.use_superpose_points_particles:
                x0, _ = superpose_points(x0, x1)
        # Sample t uniformly
        t = torch.rand(
            (x1.shape[0], *self.data_shape_ones), device=self.device)
        # Sample a noise
        z = torch.randn_like(x1)
        # Compute x_t
        x_t = self.interpolation_path.mu(t, x0, x1) + self.interpolation_path.sigma(t) * z
        # Compute the loss
        loss = torch.sum(torch.square(self.b(t, x_t) - self.interpolation_path.u(x_t, t, x0, x1)),
                         dim=self.sum_indexes) / self.dim
        # Apply the anthithetic trick
        if self.hparams.antithetic:
            # Compute x_t_antithetic
            x_t_antithetic = self.interpolation_path.mu(t, x0, x1) - self.interpolation_path.sigma(t) * z
            # Compute the loss
            loss_antithetic = torch.sum(torch.square(self.b(t, x_t) - self.interpolation_path.u(x_t_antithetic, t, x0, x1)),
                                        dim=self.sum_indexes) / self.dim
            # Sum to the main loss
            loss += loss_antithetic
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
