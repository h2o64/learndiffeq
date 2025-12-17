# Learning an ODE with stochastic interpolants

# Libraries
import torch
from ..common.learn_ode import LearnODE
from .base import InterpolantBase
from .utils import VelocityFromVelocityandScoreOrEta

class InterpolantODE(InterpolantBase, LearnODE):

    def __init__(
            self,
            data_shape,
            rho0,
            interpolant_type,
            gamma_type,
            losses,
            lr,
            velocity_types,
            velocity_kwargs=None,
            optimizer_type='adam',
            gradient_clip_val=None,
            noise_scale=1.0,
            ot_particles=False,
            use_linear_assignment_particles=False,
            use_superpose_points_particles=False,
            use_ot_coupling=False,
            use_pre_computed_ot=False,
            use_onesided=False,
            antithetic=True,
            interpolant_kwargs=None,
            force_automatic_div=False,
            **kwargs):
        """Constructor for the InterpolantODE object

        Args:
            data_shape (tuple for int): Shape of the data
            rho0 (torch.distributions.Distribution): Base distribution
            interpolant_type (str): Type of interpolant (in 'linear','linear_brownian','linear_torus','trigonometric' or 'encoding_decoding')
            gamma_type (str): Type of gamma (in 'brownian', 'bsquared','zero','sin_squared' or 'sigmoid')
            losses (list of str): Losses to use (sublist of 'b','v','s','eta')
            lr (float or dict): Learning rate. If lr is a dictionnary with keys in losses, then it will apply a per network learning rate
            velocity_types (dict of str): Type of velocities given to make_velocity (eg. velocity_types['b'] = 'mlp,128,128,128')
            velocity_kwargs (dict): Arguments for the velocity constructor
            optimizer_type (str or dict): Optimizer to use (either of 'adam','adamw','nadam' or 'sgd') (default is 'adam')
                If optimizer_type is a dictionnary with keys in losses, then it will apply a per network optimizer type
            gradient_clip_val (dict): Gradient clipping values for each loss
            noise_scale (float): Scale of the noise for gamma (default is 1.0)
            ot_particles (bool): Use OT pairing between rho0 and rho1 at each training step (default is False)
            use_linear_assignment_particles (bool): Whether to use the hungarian algorithm (default is False)
            use_superpose_points_particles (bool): Whether to use the point superposition (default is False)
            use_ot_coupling (bool): Whether to use an OT coupling
            use_pre_computed_ot (bool): Use a precomputed pairing from the dataset (default is False)
            use_onesided (bool): Use one-sided interpolants (default is False)
            antithetic (bool): Whether to use the anthitetic trick when the model leverages normal samples (default is True)
            interpolant_kwargs (dict): Arguments to pass to the interpolant classs
            force_automatic_div (bool): Whether to skip self.compute_div (default is False)
            kwargs (dict): Other arguments to pass to the LearnBase constructor
        """

        # Call the parent constructor
        super().__init__(
            data_shape=data_shape,
            rho0=rho0,
            interpolant_type=interpolant_type,
            gamma_type=gamma_type,
            losses=losses,
            lr=lr,
            velocity_types=velocity_types,
            velocity_kwargs=velocity_kwargs,
            optimizer_type=optimizer_type,
            gradient_clip_val=gradient_clip_val,
            noise_scale=noise_scale,
            ot_particles=ot_particles,
            use_linear_assignment_particles=use_linear_assignment_particles,
            use_superpose_points_particles=use_superpose_points_particles,
            use_pre_computed_ot=use_pre_computed_ot,
            use_onesided=use_onesided,
            antithetic=antithetic,
            interpolant_kwargs=interpolant_kwargs,
            force_automatic_div=force_automatic_div,
            **kwargs
        )

        # Check the arguments
        if 'b' not in self.hparams.losses and 'v' not in self.hparams.losses:
            raise ValueError('Either b or v has to be defined in losses.')

        # Manually set s in the score/eta-based velocity field
        if 'b' in self.hparams.losses and 'score' in self.hparams.velocity_types['b']:
            # This velocity field only works with one-sided interpolants
            if not self.hparams.use_onesided:
                raise NotImplementedError('Score-based velocities require a one-sided velocity type.')
            # Use the particles one or the MLP one
            if 's' in self.hparams.losses:
                net_ = self.s
                net_type = 'score'
            elif 'eta' in self.hparams.losses:
                net_ = self.eta
                net_type = 'eta'
            else:
                net_ = None
                net_type = 'score'
            hidden_layers = list(map(int, self.hparams.velocity_types['b'].split(',')[1:]))
            self.b = ScoreBasedVelocity(self.dim, net=net_, net_type=net_type, hidden_layers=hidden_layers,
                                        **self.hparams.velocity_kwargs['b'])
            self.b.int = self.interpolant
            # Disable optimization on b as the real deal
            if net_ is not None:
                self.skip_opt.add('b')

        # Make sure b is well defined
        if not hasattr(self, 'b'):
            # Build b from v and s in the classic setting
            if not self.hparams.use_onesided and hasattr(self, 'v'):
                if hasattr(self, 's'):
                    self.b = VelocityFromVelocityandScoreOrEta(
                        self.v, self.s, self.interpolant.gamma_squared_dot, self.interpolant.gamma_dot, net_type='score')
                elif hasattr(self, 'eta'):
                    self.b = VelocityFromVelocityandScoreOrEta(
                        self.v, self.eta, self.interpolant.gamma_squared_dot, self.interpolant.gamma_dot, net_type='eta')
                else:
                    raise NotImplementedError('Velocity b is not defined. You need either s or eta if you skip b.')
                self.skip_opt.add('b')
            else:
                raise ValueError('Velocity b undefined.')


class InterpolantODEGradReg(InterpolantODE):

    def __init__(self, grad_U, reg_val, **kwargs):
        """Contructor for InterpolantODEGradReg

        Args:
            grad_U (function): Gradient of the target distribution
            reg_val (float): Regularization value
            **kwargs: Arguments for InterpolantODE
        """
        super(InterpolantODEGradReg, self).__init__(grad_U=grad_U, reg_val=reg_val, **kwargs)
        # Save the parameters
        self.grad_U = grad_U

    def loss_b(self, t, x0, x1, x_t):
        """Compute the loss for b (see Eq (2.12))

        Args:
            t (torch.Tensor of shape (batch_size, (1,) * len(data_shape))): Times
            x0 (torch.Tensor of shape (batch_size, *data_shape)): rho0 samples
            x1 (torch.Tensor of shape (batch_size, *data_shape)): rho1 samples
            x_t (tuple of four torch.Tensor):
                x_t[0] is i_t_dot (same shape as x0 or x1): It corresponds to the derivative of I(t,x0,x1) with respect to t
                x_t[1] is x_t_p (same shape as x0 or x1): It Corresponds to I(t,x0,x1) + gamma(t) * z
                x_t[2] is x_t_m (same shape as x0 or x1): It Corresponds to I(t,x0,x1) - gamma(t) * z
                x_t[3] is factor_t (same shape as t): It corresponds to gamma(t) or alpha(t)
                x_t[4] is noise (same shape as x0 or x1): It corresponds to the Gaussian noise

        Returns:
            loss (float): Loss evaluation
        """

        # Unpack x_t
        i_t_dot, x_t_p, x_t_m, _, noise = x_t
        # Compute dIt_dt and gamma_dot
        if not self.hparams.use_onesided:
            gamma_dot = self.interpolant.gamma_dot(t)
        # Compute the loss
        b_ = self.b(t, x_t_p)
        loss = 0.5 * torch.sum(torch.square(b_), dim=self.sum_indexes)
        if self.hparams.use_onesided:
            loss -= torch.sum(i_t_dot * b_, dim=self.sum_indexes)
        else:
            loss -= torch.sum((i_t_dot + gamma_dot * noise) * b_, dim=self.sum_indexes)
        # Add the regularization
        loss += self.hparams.reg_val * torch.sum(torch.square(b_ - self.grad_U(x_t_p)), dim=self.sum_indexes)
        # Apply the antithetic trick
        if self.hparams.antithetic:
            b_antithetic_ = self.b(t, x_t_m)
            loss_antithetic = 0.5 * torch.sum(torch.square(b_antithetic_), dim=self.sum_indexes)
            if self.hparams.use_onesided:
                dIt_dt_antithetic = self.interpolant.interpolant_time_dot(t, -x0, x1)
                loss_antithetic -= torch.sum(dIt_dt_antithetic * b_antithetic_, dim=self.sum_indexes)
            else:
                loss_antithetic -= torch.sum((i_t_dot - gamma_dot * noise) * b_antithetic_, dim=self.sum_indexes)
            loss_antithetic += self.hparams.reg_val * \
                torch.sum(torch.square(b_antithetic_ - self.grad_U(x_t_m)), dim=self.sum_indexes)
            loss += loss_antithetic
        return loss.mean()