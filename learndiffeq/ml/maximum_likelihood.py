# Implementation of Maximum Likelihood

# Libraries
import torch
from ..common.learn_ode import LearnODE
from ..common.makers import make_velocity
from ..common.utils import torch_optimizers


class MaximumLikelihood(LearnODE):

    def __init__(
            self,
            data_shape,
            rho0,
            lr,
            velocity_type,
            velocity_kwargs=None,
            optimizer_type='adam',
            force_automatic_div=False,
            L=None,
            solver_approx=False,
            solver_method='euler',
            solver_n_steps=1000,
            solver_kwargs=None,
            **kwargs):
        """Constructor for the MaximumLikelihood object

        Args:
            data_shape (tuple of int): Shape of the data
            rho0 (torch.distributions.Distribution): Base distribution
            lr (float): Learning rate
            velocity_type (str): Type of velocities given to make_velocity (eg. 'mlp,128,128,128')
            velocity_kwargs (dict): Arguments for the velocity constructor
            optimizer_type (str): Type of optimizer (default is 'adam')
            force_automatic_div (bool): Whether to skip self.compute_div (default is False)
            L (float): Length of the box (default is None)
            kwargs (dict): Other arguments to pass to the LearnBase constructor
        """

        # Call the parent constructor
        super().__init__(
            data_shape=data_shape,
            rho0=rho0,
            lr=lr,
            velocity_type=velocity_type,
            velocity_kwargs=velocity_kwargs,
            optimizer_type=optimizer_type,
            force_automatic_div=force_automatic_div,
            L=L
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
        self.use_species = 'n_species' in velocity_kwargs
        # Build the solver details
        self.solver_details = {
            'method': solver_method,
            'n_steps': solver_n_steps,
            'approx': solver_approx,
        }
        if solver_kwargs is None:
            self.solver_kwargs = {}
        else:
            self.solver_kwargs = solver_kwargs

    def training_step(self, batch, batch_idx):
        # Get samples from rho1 and compute the loss
        if self.use_species:
            x1, a1 = batch
            loss = -self.log_prob(a1, x1, keep_b_parameters=True)
        else:
            x1, _ = batch
            loss = -self.log_prob(x1, keep_b_parameters=True)
        # Average the loss
        loss = loss.mean()
        self.log("loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # Get samples from rho1 and compute the loss
        if self.use_species:
            x1, a1 = batch
            loss = -self.log_prob(a1, x1)
        else:
            x1, _ = batch
            loss = -self.log_prob(x1)
        # Average the loss
        loss = loss.mean()
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch_optimizers[self.hparams.optimizer_type](self.b.parameters(), lr=self.hparams.lr)
        return optimizer

    def sample(self, values, a=None, reverse_time=False, return_log_jac=False,
               keep_intermediates=True, b_kwargs=None, keep_b_parameters=False, **kwargs):
        """Sample the associated ODE

        Args:
            values (torch.Tensor of shape (batch_size, *data_shape)): Initial values
                a (torch.Tensor of shape (batch_size, n_particles)): Initial species (default is None)
            reverse_time (bool): Whether to integrate reverse in time (default is False)
            return_log_jac (bool): Whether to return the log-jacobian (default is False)
            keep_intermediates (bool): Whether to keep the intermediate times (default is False)

        Return:

            if keep_intermediates:
                        if a is not None:
                                a (torch.Tensor of shape (n_steps, batch_size, n_particles)): Species
                samples (torch.Tensor of shape (n_steps, batch_size, *data_shape)): intermediate samples
                if return_log_jac
                    log_jac (torch.Tensor of shape (n_steps, batch_size)): log-jacobian
            else:
                        if a is not None:
                                a (torch.Tensor of shape (batch_size, n_particles)): Species
                samples (torch.Tensor of shape (batch_size, *data_shape)): intermediate samples
                if return_log_jac
                    log_jac (torch.Tensor of shape (batch_size,)): log-jacobian
        """
        ret = super().sample(
            torch.remainder(values, self.hparams.L) if self.hparams.L is not None else values,
            reverse_time=reverse_time,
            return_log_jac=return_log_jac,
            keep_intermediates=keep_intermediates,
            b_kwargs={'a': a} if a is not None else b_kwargs,
            keep_b_parameters=keep_b_parameters,
            **self.solver_details,
            **self.solver_kwargs
        )
        if self.hparams.L is not None:
            if return_log_jac:
                if a is not None:
                    if keep_intermediates:
                        a = a.unsqueeze(0).expand((ret[0].shape[0], -1, -1))
                    ret = (a, torch.remainder(ret[0], self.hparams.L), ret[1])
                else:
                    ret = (torch.remainder(ret[0], self.hparams.L), ret[1])
            else:
                if a is not None:
                    if keep_intermediates:
                        a = a.unsqueeze(0).expand((ret.shape[0], -1, -1))
                    ret = (a, torch.remainder(ret, self.hparams.L))
                else:
                    ret = torch.remainder(ret, self.hparams.L)
        return ret

    def forward(self, z, a=None, return_log_jac=True, keep_b_parameters=False, **kwargs):
        """Forward call to the normalizing flow

        Args:
            z (torch.Tensor of shape (batch_size, *data_shape)): Base samples
            a (torch.Tensor of shape (batch_size, n_particles)): Species (default is None)
            return_log_jac (bool): Whether to return the log-jacobian (default is False)

        Return:
                if a is not None:
                a (torch.Tensor of shape (batch_size, n_particles)): Species
            x (torch.Tensor of shape (batch_size, *data_shape)): Samples
            if return_log_jac
                log_jac (torch.Tensor of shape (batch_size,)): log-jacobian
        """
        ret = super().forward(
            torch.remainder(z, self.hparams.L) if self.hparams.L is not None else z,
            b_kwargs={'a': a} if a is not None else None,
            keep_b_parameters=keep_b_parameters,
            **self.solver_details,
            **self.solver_kwargs
        )
        if self.hparams.L is not None:
            if return_log_jac:
                ret = (torch.remainder(ret[0], self.hparams.L), ret[1])
                if a is not None:
                    ret = (a, *ret)
            else:
                ret = torch.remainder(ret, self.hparams.L)
                if a is not None:
                    ret = (a, ret)
        return ret

    def inverse(self, x, a=None, return_log_jac=True, keep_b_parameters=False, **kwargs):
        """Inverse call to the normalizing flow

        Args:
            x (torch.Tensor of shape (batch_size, *data_shape)): Target samples
            a (torch.Tensor of shape (batch_size, n_particles)): Species (default is None)
            return_log_jac (bool): Whether to return the log-jacobian (default is False)

        Return:
                if a is not None:
                a (torch.Tensor of shape (batch_size, n_particles)): Species
            z (torch.Tensor of shape (batch_size, *data_shape)): Samples
            if return_log_jac
                log_jac (torch.Tensor of shape (batch_size,)): log-jacobian
        """
        ret = super().inverse(
            torch.remainder(x, self.hparams.L) if self.hparams.L is not None else x,
            b_kwargs={'a': a} if a is not None else None,
            keep_b_parameters=keep_b_parameters,
            **self.solver_details,
            **self.solver_kwargs
        )
        if self.hparams.L is not None:
            if return_log_jac:
                ret = (torch.remainder(ret[0], self.hparams.L), ret[1])
                if a is not None:
                    ret = (a, *ret)
            else:
                ret = torch.remainder(ret, self.hparams.L)
                if a is not None:
                    ret = (a, ret)
        return ret

    def log_prob(self, x, a=None, keep_b_parameters=False, **kwargs):
        """Sample the normalizing flow and return the likelihood

        Args:
            x (torch.Tensor of shape (batch_size, *data_shape)): Samples
            a (torch.Tensor of shape (batch_size, n_particles)): Reference species (default is None)
            kwargs (dict): Argument to pass to sample_and_log_prob

        Return:
            log_prob (torch.Tensor of shape (batch_size)): Log-likelihood
        """
        z, log_jac = self.inverse(x, a=a, return_log_jac=True,
                                  keep_b_parameters=keep_b_parameters)[-2:]
        if a is not None:
            ret = self.rho0.log_prob(a, z)
        else:
            ret = self.rho0.log_prob(z)
        return ret + log_jac.flatten()

    def sample_and_log_prob(self, sample_shape, keep_b_parameters=False, **kwargs):
        """Sample the normalizing flow and return the likelihood

        Args:
            sample_shape (tuple of int): Number of samples

        Return:

                if self.use_species:
                a (torch.Tensor of shape (batch_size, n_particles)): Species
            samples (torch.Tensor of shape (batch_size, *data_shape)): Samples
            log_prob (torch.Tensor of shape (batch_size)): Log-likelihood
        """

        if self.use_species:
            base_a, base_samples = self.rho0.sample(sample_shape)
            log_prob = self.rho0.log_prob(base_a, base_samples)
        else:
            base_a, base_samples = None, self.rho0.sample(sample_shape)
            log_prob = self.rho0.log_prob(base_samples)
        ret_samples = self.sample(base_samples,
                                  a=base_a,
                                  reverse_time=False,
                                  return_log_jac=True,
                                  keep_intermediates=False,
                                  keep_b_parameters=keep_b_parameters
                                  )
        if self.use_species:
            base_a, samples, log_jac = ret_samples
        else:
            base_a, samples, log_jac = None, ret_samples[0], ret_samples[1]
        log_prob += log_jac.flatten()
        if base_a is not None:
            return base_a, samples, log_prob
        else:
            return samples, log_prob
        
class MaximumLikelihoodWithTwoDatasets(MaximumLikelihood):
    
    def training_step(self, batch, batch_idx):
        # Get samples from rho1 and compute the loss
        if self.use_species:
            x0, a0, x1, a1, = batch
            loss = -self.log_prob(a1, x1, keep_b_parameters=True)
        else:
            x0, a0, x1, a1, = batch
            loss = -self.log_prob(x1, keep_b_parameters=True)
        # Average the loss
        loss = loss.mean()
        self.log("loss", loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        # Get samples from rho1 and compute the loss
        if self.use_species:
            x0, a0, x1, a1, = batch
            loss = -self.log_prob(a1, x1)
        else:
            x0, a0, x1, a1, = batch
            loss = -self.log_prob(x1)
        # Average the loss
        loss = loss.mean()
        self.log("val_loss", loss, prog_bar=True)
        return loss


