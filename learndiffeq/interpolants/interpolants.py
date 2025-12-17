# Interpolants

# Libraries
from enum import Enum
import torch
import math
from ..particles.utils import move_to_torus

def sigmoid(x):
    """Sigmoid function using the math library"""
    return 1 / (1 + math.exp(-x))


class InterpolantType(Enum):
    """Enumerate for the interpolant type"""
    LINEAR = 1
    LINEAR_BROWNIAN = 2
    LINEAR_TORUS = 3
    TRIGONOMETRIC = 4
    ENCODING_DECODING = 5

    @classmethod
    def from_str(cls, s):
        return cls[s.upper()]


class GammaType(Enum):
    """Enumerate for the gamma type"""
    BROWNIAN = 1
    BSQUARED = 2
    ZERO = 3
    SIN_SQUARED = 4
    SIGMOID = 5

    @classmethod
    def from_str(cls, s):
        return cls[s.upper()]


class Interpolant(torch.nn.Module):

    def __init__(self, interpolant_type, gamma_type, noise_scale=1.0):
        """Constructor

        Args:
            interpolant_type (str): Type of interpolant (in 'linear','linear_torus','trigonometric' or 'encoding_decoding')
            gamma_type (str): Type of gamma (in 'brownian', 'bsquared','zero','sin_squared' or 'sigmoid')
            noise_scale (float): Scale of the noise (i.e., multiplicative factor on gamma) (default is 1.0)
        """

        # Record the interpolant type and the gamma type
        self.interpolant_type = InterpolantType.from_str(interpolant_type)
        self.gamma_type = GammaType.from_str(gamma_type)
        self.noise_scale = noise_scale
        # Various exception
        if self.interpolant_type == InterpolantType.ENCODING_DECODING and self.gamma_type != 'sin_squared':
            raise NotImplementedError(
                "Interpolant type {} is not compatible with gamma type {}. Use gamma type sin_squared instead.".format(
                    self.interpolant_type, self.gamma_type))
        super(Interpolant, self).__init__()

    def interpolant(self, t, x0, x1):
        """Interpolant function I_t(x0, x1) (see Eq (2.1) without gamma)

        Args:
            t (torch.Tensor of shape (batch_size, (1,) * len(data_shape))): Times
            x0 (torch.Tensor of shape (batch_size, *data_shape)): rho0 samples
            x1 (torch.Tensor of shape (batch_size, *data_shape)): rho1 samples

        Returns:
            x_t (torch.Tensor of shape (batch_size, *data_shape)): Evaluation of the interpolant function
        """

        if self.interpolant_type == InterpolantType.LINEAR:
            return (1.0 - t) * x0 + t * x1
        elif self.interpolant_type == InterpolantType.LINEAR_TORUS:
            return x0 + t * move_to_torus(x1 - x0, L=10.0)
        elif self.interpolant_type == InterpolantType.TRIGONOMETRIC:
            return torch.sqrt(1.0 - torch.square(self.gamma(t))) * \
                (torch.cos(torch.pi * t / 2.0) * x0 + torch.sin(torch.pi * t / 2.0) * x1)
        elif self.interpolant_type == InterpolantType.ENCODING_DECODING:
            return torch.where(
                t < 0.5,
                torch.square(torch.cos(torch.pi * t)) * x0,
                torch.square(torch.sin(torch.pi * t)) * x1
            )
        else:
            raise NotImplementedError(
                "Interpolant type {} not found.".format(self.interpolant_type))

    def interpolant_time_dot(self, t, x0, x1):
        """Time derivative of the interpolant function I_t(x0, x1) (see Eq (2.1) without gamma)

        Args:
            t (torch.Tensor of shape (batch_size, (1,) * len(data_shape))): Times
            x0 (torch.Tensor of shape (batch_size, *data_shape)): rho0 samples
            x1 (torch.Tensor of shape (batch_size, *data_shape)): rho1 samples

        Returns:
            x_t (torch.Tensor of shape (batch_size, *data_shape)): Evaluation of the time
                derivative of the interpolant function
        """

        if self.interpolant_type == InterpolantType.LINEAR:
            return x1 - x0
        elif self.interpolant_type == InterpolantType.LINEAR_TORUS:
            return move_to_torus(x1 - x0, L=10.0)
        elif self.interpolant_type == InterpolantType.TRIGONOMETRIC:
            # Compute alpha_dot
            alpha_dot = -self.gamma_squared_dot(t) / torch.sqrt(1 -
                                                                torch.square(self.gamma(t))) * torch.cos(torch.pi * t / 2.)
            alpha_dot -= torch.pi * torch.sqrt(1 - torch.square(self.gamma(t))) * torch.sin(torch.pi * t / 2) / 2.
            # Compute beta_dot
            beta_dot = -self.gamma_squared_dot(t) / torch.sqrt(1 -
                                                               torch.square(self.gamma(t))) * torch.sin(torch.pi * t / 2.)
            beta_dot += torch.pi * torch.sqrt(1 - torch.square(self.gamma(t))) * torch.cos(math.pi * t / 2.) / 2.
            # Return the derivative
            return alpha_dot * x0 + beta_dot * x1
        elif self.interpolant_type == InterpolantType.ENCODING_DECODING:
            factor = 2. * torch.pi * torch.cos(torch.pi * t) * torch.sin(torch.pi * t)
            return torch.where(t < 0.5, -factor * x0, factor * x1)
        else:
            raise NotImplementedError(
                "Interpolant type {} not found.".format(self.interpolant_type))

    def gamma(self, t):
        """Gamma function (see Eq (2.1))

        Args:
            t (torch.Tensor): Times

        Returns:
            gamma_t (torch.Tensor with the same shape as t): Evaluation of the gamma function
        """

        if self.gamma_type == GammaType.BROWNIAN:
            ret = torch.sqrt(t * (1.0 - t))
        elif self.gamma_type == GammaType.BSQUARED:
            ret = t * (1.0 - t)
        elif self.gamma_type == GammaType.ZERO or self.gamma_type is None:
            ret = torch.zeros_like(t)
        elif self.gamma_type == GammaType.SIN_SQUARED:
            ret = torch.square(torch.sin(torch.pi * t))
        elif self.gamma_type == GammaType.SIGMOID:
            f = 10.0
            ret = torch.sigmoid(f * (t - 0.5) + 1.) - torch.sigmoid(f * (t - 0.5) - 1.) - \
                sigmoid((-f / 2.) + 1.) + sigmoid((-f / 2.) - 1.)
        else:
            raise NotImplementedError(
                "Gamma type {} not found.".format(self.gamma_type))
        return self.noise_scale * ret

    def gamma_dot(self, t):
        """Derivative of the gamma function (see Eq (2.1))

        Args:
            t (torch.Tensor): Times

        Returns:
            dot_gamma_t (torch.Tensor with the same shape as t): Evaluation of the
                derivative of the gamma function
        """

        if self.gamma_type == GammaType.BROWNIAN:
            ret = (0.5 - t) / torch.sqrt(t * (1.0 - t))
        elif self.gamma_type == GammaType.BSQUARED:
            ret = 1.0 - 2.0 * t
        elif self.gamma_type == GammaType.ZERO or self.gamma_type is None:
            ret = torch.zeros_like(t)
        elif self.gamma_type == GammaType.SIN_SQUARED:
            ret = 2 * torch.pi * torch.sin(torch.pi * t) * torch.cos(torch.pi * t)
        elif self.gamma_type == GammaType.SIGMOID:
            f = 10.0
            ret = f * (1. - torch.sigmoid(1. + f * (t - 0.5))) * torch.sigmoid(1. + f * (t - 0.5))
            ret -= f * (1. - torch.sigmoid(-1. + f * (t - 0.5))) * torch.sigmoid(-1. + f * (t - 0.5))
        else:
            raise NotImplementedError(
                "Gamma type {} not found.".format(self.gamma_type))
        return self.noise_scale * ret

    def gamma_squared_dot(self, t):
        """Derivative of the squared gamma function divided by 2 (see Eq (2.1))

        Args:
            t (torch.Tensor): Times

        Returns:
            dot_gamma_sq_t (torch.Tensor with the same shape as t): Evaluation of the
                derivative of the squared gamma function divided by 2
        """

        if self.gamma_type == GammaType.BROWNIAN:
            ret = 0.5 - t
        elif self.gamma_type == GammaType.BSQUARED:
            ret = self.gamma(t) * self.gamma_dot(t)
        elif self.gamma_type == GammaType.ZERO or self.gamma_type is None:
            ret = torch.zeros_like(t)
        elif self.gamma_type == GammaType.SIN_SQUARED:
            ret = 2 * torch.pi * torch.pow(torch.sin(torch.pi * t), 3) * torch.cos(torch.pi * t)
        elif self.gamma_type == GammaType.SIGMOID:
            ret = self.gamma(t) * self.gamma_dot(t)
        else:
            raise NotImplementedError(
                "Gamma type {} not found.".format(self.gamma_type))
        return self.noise_scale * self.noise_scale * ret

    def forward(self, t, x0, x1, antithetic=False):
        """Call to the interpolant

        Args:
            t (torch.Tensor of shape (batch_size, (1,) * len(data_shape))): Times
            x0 (torch.Tensor of shape (batch_size, *data_shape)): rho0 samples
            x1 (torch.Tensor of shape (batch_size, *data_shape)): rho1 samples
            antithetic (bool): Whether to use the antithetic trick

        Returns:
            i_t_dot (torch.Tensor of shape (batch_size, *data_shape)): Evaluation of the derivative of the interpolant function
            x_t_p (torch.Tensor of shape (batch_size, *data_shape)): Evaluation of the interpolant
            x_t_m (torch.Tensor of shape (batch_size, *data_shape) or None): Antithetic evaluation of the interpolant
                (None is antithetic is False)
            gamma_t (torch.Tensor of shape (batch_size, (1,) * len(data_shape))): Evaluation of the gamma function
            noise (torch.Tensor with the same shape as x0 or x1): Independent standard centered Gaussian samples
        """

        # Compute the ingredients for the interpolant
        z = torch.randn_like(x0)
        i_t = self.interpolant(t, x0, x1)
        i_t_dot = self.interpolant_time_dot(t, x0, x1)
        gamma_t = self.gamma(t)
        # Compute the antithetic version
        if antithetic:
            x_t_p = i_t + gamma_t * z
            x_t_m = i_t - gamma_t * z
            return i_t_dot, x_t_p, x_t_m, gamma_t, z
        else:
            return i_t_dot, i_t + gamma_t * z, None, gamma_t, z


class OneSidedInterpolant(Interpolant):
    """Implementation of the one-sided spatially linear interpolant (see Eq (3.12) for the former
    and Eq (3.1) for the later)

    WARNING: This is the reverse of Sec. 3.3 has the normal distribution is
    here at time t=0
    """

    def __init__(self, interpolant_type, gamma_type, noise_scale=1.0):
        """Constructor

        Args:
            interpolant_type (str): Type of interpolant (in 'linear','linear_brownian','linear_torus','trigonometric' or 'encoding_decoding')
            gamma_type (str): Type of gamma (in 'brownian', 'bsquared','zero','sin_squared' or 'sigmoid')
            noise_scale (float): Scale of the noise (i.e., multiplicative factor on gamma) (default is 1.0)
        """

        # Various exception
        if interpolant_type == InterpolantType.ENCODING_DECODING:
            raise NotImplementedError("Interpolant type encoding_decoding is not compatible the one-sided interpolant.")
        super(OneSidedInterpolant, self).__init__(interpolant_type, gamma_type, noise_scale)

    def alpha(self, t):
        """Alpha function (see Eq (3.1))

        Args:
            t (torch.Tensor): Times

        Returns:
            alpha_t (torch.Tensor with the same shape as t): Evaluation of the alpha function
        """

        if self.interpolant_type == InterpolantType.LINEAR_BROWNIAN:
            return torch.sqrt(1. - t)
        elif self.interpolant_type == InterpolantType.LINEAR:
            return 1. - t
        elif self.interpolant_type == InterpolantType.TRIGONOMETRIC:
            return torch.sqrt(1. - torch.square(self.gamma(t))) * torch.cos(torch.pi * t / 2.)
        else:
            raise NotImplementedError(
                "Interpolant type {} not found.".format(self.interpolant_type))

    def alpha_dot(self, t):
        """Derivative of the alpha function (see Eq (3.1))

        Args:
            t (torch.Tensor): Times

        Returns:
            dot_alpha_t (torch.Tensor with the same shape as t): Evaluation of the derivative of
                the alpha function
        """

        if self.interpolant_type == InterpolantType.LINEAR_BROWNIAN:
            return -1. / (2. * torch.sqrt(1. - t))
        elif self.interpolant_type == InterpolantType.LINEAR:
            return -torch.ones_like(t)
        elif self.interpolant_type == InterpolantType.TRIGONOMETRIC:
            ret = -self.gamma_squared_dot(t) / torch.sqrt(1 - torch.square(self.gamma(t))
                                                          ) * torch.cos(torch.pi * t / 2.)
            ret -= torch.pi * torch.sqrt(1 - torch.square(self.gamma(t))) * torch.sin(torch.pi * t / 2) / 2.
            return ret
        else:
            raise NotImplementedError(
                "Interpolant type {} not found.".format(self.interpolant_type))

    def alpha_squared_dot(self, t):
        """Derivative of the squared alpha function divided by two (see Eq (3.1))

        Args:
            t (torch.Tensor): Times

        Returns:
            dot_alpha_sq_t (torch.Tensor with the same shape as t): Evaluation of the derivative
                of the squared alpha function divided by two
        """

        if self.interpolant_type == InterpolantType.LINEAR_BROWNIAN:
            return -0.5 * torch.ones_like(t)
        elif self.interpolant_type == InterpolantType.LINEAR:
            return t - 1.0
        elif self.interpolant_type == InterpolantType.TRIGONOMETRIC:
            return self.alpha_dot(t) * self.alpha(t)
        else:
            raise NotImplementedError(
                "Interpolant type {} not found.".format(self.interpolant_type))

    def beta(self, t):
        """Beta function (see Eq (3.1))

        Args:
            t (torch.Tensor): Times

        Returns:
            beta_t (torch.Tensor with the same shape as t): Evaluation of the beta function
        """

        if self.interpolant_type == InterpolantType.LINEAR or self.interpolant_type == InterpolantType.LINEAR_BROWNIAN:
            return t
        elif self.interpolant_type == InterpolantType.TRIGONOMETRIC:
            return torch.sqrt(1. - torch.square(self.gamma(t))) * torch.sin(torch.pi * t / 2.)
        else:
            raise NotImplementedError(
                "Interpolant type {} not found.".format(self.interpolant_type))

    def beta_dot(self, t):
        """Derivative of the beta function (see Eq (3.1))

        Args:
            t (torch.Tensor): Times

        Returns:
            dot_beta_t (torch.Tensor with the same shape as t): Evaluation of the derivative of
                the beta function
        """

        if self.interpolant_type == InterpolantType.LINEAR or self.interpolant_type == InterpolantType.LINEAR_BROWNIAN:
            return torch.ones_like(t)
        elif self.interpolant_type == InterpolantType.TRIGONOMETRIC:
            ret = -self.gamma_squared_dot(t) / torch.sqrt(1 - torch.square(self.gamma(t))
                                                          ) * torch.sin(torch.pi * t / 2.)
            ret += torch.pi * torch.sqrt(1 - torch.square(self.gamma(t))) * torch.cos(math.pi * t / 2.) / 2.
            return ret
        else:
            raise NotImplementedError(
                "Interpolant type {} not found.".format(self.interpolant_type))

    def beta_squared_dot(self, t):
        """Derivative of the squared beta function divided by two (see Eq (3.1))

        Args:
            t (torch.Tensor): Times

        Returns:
            dot_beta_sq_t (torch.Tensor with the same shape as t): Evaluation of the derivative
                of the squared beta function divided by two
        """
        if self.interpolant_type == InterpolantType.LINEAR_BROWNIAN or self.interpolant_type == InterpolantType.LINEAR:
            return t
        elif self.interpolant_type == InterpolantType.TRIGONOMETRIC:
            return self.beta_dot(t) * self.beta(t)
        else:
            raise NotImplementedError(
                "Interpolant type {} not found.".format(self.interpolant_type))

    def interpolant(self, t, x0, x1):
        """Interpolant function I_t(x0, x1) (see Eq (2.1) without gamma)

        Args:
            t (torch.Tensor of shape (batch_size, (1,) * len(data_shape))): Times
            x0 (torch.Tensor of shape (batch_size, *data_shape)): rho0 samples
            x1 (torch.Tensor of shape (batch_size, *data_shape)): rho1 samples

        Returns:
            x_t (torch.Tensor of shape (batch_size, *data_shape)): Evaluation of the interpolant function
        """

        return self.alpha(t) * x0 + self.beta(t) * x1

    def interpolant_time_dot(self, t, x0, x1):
        """Time derivative of the interpolant function I_t(x0, x1) (see Eq (2.1) without gamma)

        Args:
            t (torch.Tensor of shape (batch_size, (1,) * len(data_shape))): Times
            x0 (torch.Tensor of shape (batch_size, *data_shape)): rho0 samples
            x1 (torch.Tensor of shape (batch_size, *data_shape)): rho1 samples

        Returns:
            x_t (torch.Tensor of shape (batch_size, *data_shape)): Evaluation of the time
                derivative of the interpolant function
        """

        return self.alpha_dot(t) * x0 + self.beta_dot(t) * x1

    def forward(self, t, x0, x1, antithetic=True):
        """Call to the interpolant

        Args:
            t (torch.Tensor of shape (batch_size, (1,) * len(data_shape))): Times
            x0 (torch.Tensor of shape (batch_size, *data_shape)): rho0 samples
            x1 (torch.Tensor of shape (batch_size, *data_shape)): rho1 samples
            antithetic (bool): Whether to use the antithetic trick

        Returns:
            i_t_dot (torch.Tensor of shape (batch_size, *data_shape)): Evaluation of the derivative of the interpolant function
            x_t_p (torch.Tensor of shape (batch_size, *data_shape)): Evaluation of the interpolant
            x_t_m (torch.Tensor of shape (batch_size, *data_shape) or None): Antithetic evaluation of the interpolant
                (None if antithetic is False)
            alpha_t (torch.Tensor of shape (batch_size, (1,) * len(data_shape))): Evaluation of the gamma function
            noise (torch.Tensor with the same shape as x0 or x1): Independent standard centered Gaussian samples from the base
        """

        # Compute alpha appart
        alpha_t = self.alpha(t)
        # Compute the interpolant with the antithetic trick
        x_t_dot = self.interpolant_time_dot(t, x0, x1)
        if antithetic:
            x_t_p = self.interpolant(t, x0, x1)
            x_t_m = self.interpolant(t, -x0, x1)
            return x_t_dot, x_t_p, x_t_m, alpha_t, x0
        else:
            return x_t_dot, self.interpolant(t, x0, x1), None, alpha_t, x0