# Learning a SDE with stochastic interpolants

# Libraries
from ..common.learn_sde import LearnSDE
from .base import InterpolantBase
from .utils import SDEFromVelocityAndScore


class InterpolantSDE(InterpolantBase, LearnSDE):

    def __init__(
            self,
            data_shape,
            rho0,
            interpolant_type,
            gamma_type,
            epsilon,
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
            **kwargs):
        """Constructor for the InterpolantSDE object

        Args:
            data_shape (tuple for int): Shape of the data
            rho0 (torch.distributions.Distribution): Base distribution
            interpolant_type (str): Type of interpolant (in 'linear','linear_brownian','linear_torus','trigonometric' or 'encoding_decoding')
            gamma_type (str): Type of gamma (in 'brownian', 'bsquared','zero','sin_squared' or 'sigmoid')
            epsilon (float): Epsilon in the drift definition
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
            kwargs (dict): Other arguments to pass to the LearnBase constructor
        """

        # Call the parent constructor
        super().__init__(
            data_shape=data_shape,
            rho0=rho0,
            interpolant_type=interpolant_type,
            gamma_type=gamma_type,
            epsilon=epsilon,
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
            **kwargs
        )

        # Check the arguments
        if 'v' not in self.hparams.losses:
            raise ValueError('The velocity v has to be defined in losses.')
        if 's' not in self.hparams.losses:
            raise ValueError('The score s has to be defined in losses.')

    def sde_maker(self, reverse_time):
        return SDEFromVelocityAndScore(
            self.v,
            self.s,
            self.hparams.epsilon,
            self.interpolant.gamma_squared_dot,
            reverse=reverse_time)
