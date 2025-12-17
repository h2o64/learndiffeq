# Learning an ODE

# Libraries
import torch
from .learn_base import LearnBase
from .utils import ModuleAndDivergence, VelocityWithKwargs
from .ode_solvers import odeint_skip_intermediates, odeint_adjoint_skip_intermediates
from torchdiffeq import odeint, odeint_adjoint
from contextlib import nullcontext


class LearnODE(LearnBase):

    def __init__(self, force_automatic_div=False, **kwargs):
        super().__init__(force_automatic_div=force_automatic_div, **kwargs)

    def sample(self, values, method='euler', n_steps=1000, reverse_time=False, approx=False, return_log_jac=False,
               keep_intermediates=True, b_kwargs=None, keep_b_parameters=False, **solver_kwargs):
        """Sample the associated ODE

        Args:
            values (torch.Tensor of shape (batch_size, *data_shape)): Initial values
            method (str): Integration method (default is euler)
            n_steps (int): Number of integration steps (default is 1000)
            reverse_time (bool): Whether to integrate reverse in time (default is False)
            approx (bool): Whether to approximate divergence with hutchinson’s trace estimator (default is False)
            return_log_jac (bool): Whether to return the log-jacobian (default is False)
            keep_intermediates (bool): Whether to keep the intermediate times (default is False)
            b_kwargs (dict): Arguments to pass to the velocity field (default is None)
            keep_b_parameters (bool): Whether to keep the dependancy with respect to b's parameters (default is False)
            solver_kwargs (dict): Kwargs to pass to the solver

        Return:
            if keep_intermediates:
                samples (torch.Tensor of shape (n_steps, batch_size, *data_shape)): intermediate samples
                if return_log_jac
                    log_jac (torch.Tensor of shape (n_steps, batch_size)): log-jacobian
            else:
                samples (torch.Tensor of shape (batch_size, *data_shape)): intermediate samples
                if return_log_jac
                    log_jac (torch.Tensor of shape (batch_size,)): log-jacobian
        """

        t = torch.linspace(0.0, 1.0, n_steps).to(values.device)
        if reverse_time:
            t = 1.0 - t
        if keep_intermediates:
            if keep_b_parameters:
                integrator = odeint_adjoint
            else:
                integrator = odeint
        else:
            if keep_intermediates:
                integrator = odeint_adjoint_skip_intermediates
            else:
                integrator = odeint_skip_intermediates
        if b_kwargs is not None:
            b_ = VelocityWithKwargs(self.b, **b_kwargs)
        else:
            b_ = self.b
        if not return_log_jac:
            grad_context = nullcontext() if keep_b_parameters else torch.no_grad()
            with grad_context:
                return integrator(
                    func=b_,
                    y0=values,
                    t=t,
                    method=method,
                    **solver_kwargs
                )
        else:
            # Disable flow's parameter gradients
            if not keep_b_parameters:
                parameters_states = []
                for p in b_.parameters():
                    parameters_states.append(p.requires_grad)
                    p.requires_grad_(False)
            # Run the solver
            x, log_jac = integrator(
                func=ModuleAndDivergence(b_, self.hparams.data_shape, approx=approx, reverse_time=reverse_time,
                                         force_automatic_div=self.hparams.force_automatic_div),
                y0=(values, torch.zeros(
                    (values.shape[0], 1), device=values.device)),
                t=t,
                method=method,
                **solver_kwargs
            )
            # Restore flow's parameter gradients
            if not keep_b_parameters:
                for i, p in enumerate(b_.parameters()):
                    p.requires_grad_(parameters_states[i])
            return x, log_jac

    def forward(self, z, method='euler', n_steps=1000, approx=False, return_log_jac=True,
                b_kwargs=None, keep_b_parameters=False, **solver_kwargs):
        """Forward call to the normalizing flow

        Args:
            z (torch.Tensor of shape (batch_size, *data_shape)): Base samples
            method (str): Integration method (default is euler)
            n_steps (int): Number of integration steps (default is 1000)
            approx (bool): Whether to approximate divergence with hutchinson’s trace estimator (default is False)
            return_log_jac (bool): Whether to return the log-jacobian (default is False)
            b_kwargs (dict): Arguments to pass to the velocity field (default is None)
            solver_kwargs (dict): Kwargs to pass to the solver

        Return:
            x (torch.Tensor of shape (batch_size, *data_shape)): Samples
            if return_log_jac
                log_jac (torch.Tensor of shape (batch_size,)): log-jacobian
        """

        ret = self.sample(
            z,
            method=method,
            n_steps=n_steps,
            reverse_time=False,
            approx=approx,
            return_log_jac=return_log_jac,
            keep_intermediates=False,
            b_kwargs=b_kwargs,
            keep_b_parameters=keep_b_parameters,
            **solver_kwargs
        )
        if return_log_jac:
            return ret[0], -ret[1].flatten()
        else:
            return ret

    def inverse(self, x, method='euler', n_steps=1000, approx=False, return_log_jac=True,
                b_kwargs=None, keep_b_parameters=False, **solver_kwargs):
        """Inverse call to the normalizing flow

        Args:
            x (torch.Tensor of shape (batch_size, *data_shape)): Target samples
            method (str): Integration method (default is euler)
            n_steps (int): Number of integration steps (default is 1000)
            approx (bool): Whether to approximate divergence with hutchinson’s trace estimator (default is False)
            return_log_jac (bool): Whether to return the log-jacobian (default is False)
            b_kwargs (dict): Arguments to pass to the velocity field (default is None)
            solver_kwargs (dict): Kwargs to pass to the solver

        Return:
            z (torch.Tensor of shape (batch_size, *data_shape)): Samples
            if return_log_jac
                log_jac (torch.Tensor of shape (batch_size,)): log-jacobian
        """
        ret = self.sample(
            x,
            method=method,
            n_steps=n_steps,
            reverse_time=True,
            approx=approx,
            return_log_jac=return_log_jac,
            keep_intermediates=False,
            b_kwargs=b_kwargs,
            keep_b_parameters=keep_b_parameters,
            **solver_kwargs
        )
        if return_log_jac:
            return ret[0], -ret[1].flatten()
        else:
            return ret

    def sample_and_log_prob(self, sample_shape, method='euler', n_steps=1000,
                            approx=False, keep_b_parameters=False, **solver_kwargs):
        """Sample the normalizing flow and return the likelihood

        Args:
            sample_shape (tuple of int): Shape of the samples
            method (str): Integration method (default is euler)
            n_steps (int): Number of integration steps (default is 1000)
            approx (bool): Whether to approximate divergence with hutchinson’s trace estimator (default is False)
            solver_kwargs (dict): Kwargs to pass to the solver

        Return:
            samples (torch.Tensor of shape (batch_size, *data_shape)): Samples
            log_prob (torch.Tensor of shape (batch_size)): Log-likelihood
        """

        base_samples = self.rho0.sample(sample_shape)
        samples, log_jac = self.sample(base_samples, method=method, n_steps=n_steps, reverse_time=False,
                                       approx=approx, return_log_jac=True, keep_intermediates=False, keep_b_parameters=keep_b_parameters,
                                       **solver_kwargs
                                       )
        log_prob = self.rho0.log_prob(base_samples) + log_jac.flatten()
        return samples, log_prob
