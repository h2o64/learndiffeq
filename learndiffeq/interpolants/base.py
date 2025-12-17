# Using stochastic interpolants

# Libraries
import torch
from ..common.learn_base import LearnBase
from ..common.makers import make_velocity
from ..common.utils import sample_ot_coupling, torch_optimizers
from ..particles.utils import ot_permutation, linear_assignment_permutation, superpose_points
from .interpolants import Interpolant, OneSidedInterpolant

class InterpolantBase(LearnBase):

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
            **kwargs):
        """Constructor for the InterpolantBase object

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
            use_ot_coupling=use_ot_coupling,
            use_pre_computed_ot=use_pre_computed_ot,
            use_onesided=use_onesided,
            antithetic=antithetic,
            interpolant_kwargs=interpolant_kwargs,
            **kwargs
        )

        # Make the interpolant
        if self.hparams.use_onesided:
            interpolant_class = OneSidedInterpolant
        else:
            interpolant_class = Interpolant
        if self.hparams.interpolant_kwargs is not None:
            interpolant_kwargs_ = self.hparams.interpolant_kwargs
        else:
            interpolant_kwargs_ = {}
        self.interpolant = interpolant_class(
            interpolant_type=self.hparams.interpolant_type,
            gamma_type=self.hparams.gamma_type,
            noise_scale=self.hparams.noise_scale,
            **interpolant_kwargs_
        )

        # Get the dictionnary of velocity_kwargs
        if self.hparams.velocity_kwargs is None:
            velocity_kwargs_dic = {}
        else:
            velocity_kwargs_dic = dict(self.hparams.velocity_kwargs)
        for net_type in self.hparams.losses:
            if net_type not in velocity_kwargs_dic:
                velocity_kwargs_dic[net_type] = {}

        # Cannot use v with the one-sided interpolant
        if 'v' in self.hparams.losses and self.hparams.use_onesided:
            raise NotImplemented('The loss v cannot be used with the one-sided interpolant.')

        # Build the networks you want to train
        self.skip_opt = set()
        if 's' in self.hparams.losses:
            self.s = make_velocity(
                self.hparams.data_shape,
                self.dim,
                self.hparams.velocity_types['s'],
                **velocity_kwargs_dic['s'])
        if 'eta' in self.hparams.losses:
            self.eta = make_velocity(
                self.hparams.data_shape,
                self.dim,
                self.hparams.velocity_types['eta'],
                **velocity_kwargs_dic['eta'])
        if 'b' in self.hparams.losses:
            self.b = make_velocity(
                self.hparams.data_shape,
                self.dim,
                self.hparams.velocity_types['b'],
                **velocity_kwargs_dic['b'])
        if 'v' in self.hparams.losses:
            self.v = make_velocity(
                self.hparams.data_shape,
                self.dim,
                self.hparams.velocity_types['v'],
                **velocity_kwargs_dic['v'])

        # Disable the antithetic trick if not needed
        if not self.hparams.use_onesided and self.hparams.gamma_type == 'zero':
            self.hparams.antithetic = False

        # Make an intermediate dict for gradient clipping values
        if isinstance(self.hparams.gradient_clip_val, dict):
            self.gradient_clip_vals = self.hparams.gradient_clip_val
        else:
            self.gradient_clip_vals = {
                net_type: self.hparams.gradient_clip_val for net_type in self.hparams.losses if net_type not in self.skip_opt
            }
        self.gradient_clip_vals = [self.gradient_clip_vals[net_type]
                                   for net_type in self.hparams.losses if net_type not in self.skip_opt]

        # Disable automatic optimization
        self.automatic_optimization = False

    def configure_optimizers(self):
        """Define the optimizers"""

        # Make an intermediate dict for learning rates
        if isinstance(self.hparams.lr, dict):
            lrs = self.hparams.lr
        else:
            lrs = {
                net_type: self.hparams.lr for net_type in self.hparams.losses if net_type not in self.skip_opt
            }

        # Make an intermediate dict for optimizer types
        if isinstance(self.hparams.optimizer_type, dict):
            optimizers_types = self.hparams.optimizer_type
        else:
            optimizers_types = {
                net_type: self.hparams.optimizer_type for net_type in self.hparams.losses if net_type not in self.skip_opt}

        return [torch_optimizers[optimizers_types[net_type]](getattr(self, net_type).parameters(), lr=lrs[net_type])
                for net_type in self.hparams.losses if net_type not in self.skip_opt]

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
        # Apply the antithetic trick
        if self.hparams.antithetic:
            b_antithetic_ = self.b(t, x_t_m)
            loss_antithetic = 0.5 * torch.sum(torch.square(b_antithetic_), dim=self.sum_indexes)
            if self.hparams.use_onesided:
                dIt_dt_antithetic = self.interpolant.interpolant_time_dot(t, -x0, x1)
                loss_antithetic -= torch.sum(dIt_dt_antithetic * b_antithetic_, dim=self.sum_indexes)
            else:
                loss_antithetic -= torch.sum((i_t_dot - gamma_dot * noise) * b_antithetic_, dim=self.sum_indexes)
            loss += loss_antithetic
        return loss.mean()

    def loss_v(self, t, x0, x1, x_t):
        """Compute the loss for v (see Eq (2.25))

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
        i_t_dot, x_t_p, x_t_m, gamma_t, z = x_t
        # Compute the loss
        v_ = self.v(t, x_t_p)
        loss = 0.5 * torch.sum(torch.square(v_), dim=self.sum_indexes)
        loss -= torch.sum(i_t_dot * v_, dim=self.sum_indexes)
        # Apply the antithetic trick
        if self.hparams.antithetic:
            v_antithetic_ = self.v(t, x_t_m)
            loss_antithetic = 0.5 * torch.sum(torch.square(v_antithetic_), dim=self.sum_indexes)
            loss_antithetic -= torch.sum(i_t_dot * v_antithetic_, dim=self.sum_indexes)
            loss += loss_antithetic
        return loss.mean()

    def loss_s(self, t, x0, x1, x_t):
        """Compute the loss for s (see Eq (2.15))

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
        _, x_t_p, x_t_m, factor_t, noise = x_t
        # Compute the loss
        s_ = self.s(t, x_t_p)
        loss = 0.5 * torch.sum(torch.square(s_), dim=self.sum_indexes)
        loss += torch.sum(noise * s_, dim=self.sum_indexes) / factor_t.flatten()
        # Apply the antithetic trick
        if self.hparams.antithetic:
            s_antithetic_ = self.s(t, x_t_m)
            loss_antithetic = 0.5 * torch.sum(torch.square(s_antithetic_), dim=self.sum_indexes)
            loss_antithetic -= torch.sum(noise * s_antithetic_, dim=self.sum_indexes) / factor_t.flatten()
            loss += loss_antithetic
        return loss.mean()

    # def loss_s_pot_based(self, t, x0, x1, x_t):
    #     """Compute the loss for s for potential-based networks (see Eq (2.15))

    #     Args:
    #         t (torch.Tensor of shape (batch_size, (1,) * len(data_shape))): Times
    #         x0 (torch.Tensor of shape (batch_size, *data_shape)): rho0 samples
    #         x1 (torch.Tensor of shape (batch_size, *data_shape)): rho1 samples
    #         x_t (tuple of four torch.Tensor):
    #             x_t[0] is i_t_dot (same shape as x0 or x1): It corresponds to the derivative of I(t,x0,x1) with respect to t
    #             x_t[1] is x_t_p (same shape as x0 or x1): It Corresponds to I(t,x0,x1) + gamma(t) * z
    #             x_t[2] is x_t_m (same shape as x0 or x1): It Corresponds to I(t,x0,x1) - gamma(t) * z
    #             x_t[3] is factor_t (same shape as t): It corresponds to gamma(t) or alpha(t)
    #             x_t[4] is noise (same shape as x0 or x1): It corresponds to the Gaussian noise

    #     Returns:
    #         loss (float): Loss evaluation
    #     """

    #     # Unpack x_t
    #     _, x_t_p, x_t_m, factor_t, noise = x_t
    #     # Compute the loss
    #     s_b_particles_ = self.s.b_particle_factor * self.s.b_particle(t, x_t_p)
    #     s_b_f_ = self.s.threshold_fun(t, self.s.f(x_t_p))
    #     loss = 0.5 * torch.sum(torch.square(s_b_particles_), dim=self.sum_indexes)
    #     loss += torch.sum(noise * s_b_particles_, dim=self.sum_indexes) / factor_t.flatten()
    #     loss += torch.sum(s_b_particles_ * s_b_f_, dim=self.sum_indexes)
    #     # Apply the antithetic trick
    #     if self.hparams.antithetic:
    #         s_b_particles_antithetic_ = self.s.b_particle_factor * self.s.b_particle(t, x_t_m)
    #         s_b_f_antithetic_ = self.s.threshold_fun(t, self.s.f(x_t_m))
    #         loss_antithetic = 0.5 * torch.sum(torch.square(s_b_particles_antithetic_), dim=self.sum_indexes)
    #         loss_antithetic -= torch.sum(noise * s_b_particles_antithetic_, dim=self.sum_indexes) / factor_t.flatten()
    #         loss_antithetic += torch.sum(s_b_particles_antithetic_ * s_b_f_antithetic_, dim=self.sum_indexes)
    #         loss += loss_antithetic
    #     return loss.mean()

    def loss_eta(self, t, x0, x1, x_t):
        """Compute the loss for eta_0

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
        _, x_t_p, x_t_m, _, noise = x_t
        # Compute the loss
        eta_ = self.eta(t, x_t_p)
        loss = 0.5 * torch.sum(torch.square(eta_), dim=self.sum_indexes)
        loss -= torch.sum(noise * eta_, dim=self.sum_indexes)
        # Apply the antithetic trick
        if self.hparams.antithetic:
            eta_antithetic_ = self.eta(t, x_t_m)
            loss_antithetic = 0.5 * torch.sum(torch.square(eta_antithetic_), dim=self.sum_indexes)
            loss_antithetic += torch.sum(noise * eta_antithetic_, dim=self.sum_indexes)
            loss += loss_antithetic
        return loss.mean()

    def get_loss(self, loss_type, t, x0, x1, x_t):
        """Compute the right loss

        Args:
            loss_type (str): Type of loss to compute (can be any of 'b','v','s','eta')
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

        # Compute the loss
        if loss_type == 'b':
            return self.loss_b(t, x0, x1, x_t)
        elif loss_type == 'v':
            return self.loss_v(t, x0, x1, x_t)
        elif loss_type == 's':
            # if self.s_pot_based:
            #     return self.loss_s_pot_based(t, x0, x1, x_t)
            # else:
            #     return self.loss_s(t, x0, x1, x_t)
            return self.loss_s(t, x0, x1, x_t)
        elif loss_type == 'eta':
            return self.loss_eta(t, x0, x1, x_t)
        else:
            raise NotImplemented('Loss {} not implemented.'.format(loss_type))

    def training_step(self, batch, batch_idx):
        """Training step

        Args:
            batch (torch.Tensor or tuple of torch.Tensor with shape (batch_size, ...)): Batch of data
            batch_idx (int): Index of the current batch
        """

        # Get the optimizers
        optimizers = self.optimizers()
        if not isinstance(optimizers, list) and (len(self.hparams.losses) - len(self.skip_opt)) == 1:
            optimizers = [optimizers]
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
        # Compute x_t
        with torch.no_grad():
            x_t = self.interpolant(t, x0, x1, antithetic=self.hparams.antithetic)
        # Compute the losses
        losses = {}
        for loss_id, loss_type in enumerate(filter(lambda x: x not in self.skip_opt, self.hparams.losses)):
            optimizers[loss_id].zero_grad()
            loss = self.get_loss(loss_type, t, x0, x1, x_t)
            losses['loss_' + loss_type] = loss
            self.manual_backward(loss)
            if self.gradient_clip_vals[loss_id] is not None:
                self.clip_gradients(optimizers[loss_id], gradient_clip_val=self.gradient_clip_vals[loss_id],
                                    gradient_clip_algorithm="norm")
            optimizers[loss_id].step()
        self.log_dict(losses, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        """Training step

        Args:
            batch (torch.Tensor or tuple of torch.Tensor with shape (batch_size, ...)): Batch of data
            batch_idx (int): Index of the current batch
        """

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
        # Compute x_t
        with torch.no_grad():
            x_t = self.interpolant(t, x0, x1, antithetic=self.hparams.antithetic)
        # Compute the losses
        losses = {}
        for loss_type in filter(lambda x: x not in self.skip_opt, self.hparams.losses):
            loss = self.get_loss(loss_type, t, x0, x1, x_t)
            losses['val_loss_' + loss_type] = loss
        self.log_dict(losses, prog_bar=True)
