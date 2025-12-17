# Various helpers

# Libraries
import math
import torch
from ..common.utils import compute_div_fn, compute_div_approx_fn


class VelocityFromVelocityandScoreOrEta(torch.nn.Module):
    """Define a velocity field b from v and s or eta"""

    def __init__(self, v, net, gamma_gamma_dot, gamma_dot, net_type):
        """Constructor of VelocityFromVandS

        Args:
            v (torch.nn.Module): Velocity field
            s (torch.nn.Module): Score function
            gamma_gamma_dot (function): Returns gamma(t) * \\dot{gamma}(t)
        """

        # Call parent's contructor
        super().__init__()
        # Store the neural networks
        self.v = v
        self.net = net
        self.net_is_score = net_type == 'score'
        # Store gamma_gamma_dot and gamma_dot
        self.gamma_gamma_dot = gamma_gamma_dot
        self.gamma_dot = gamma_dot

    def forward(self, t, x):
        """Compute b(t, x)

        Args:
            t (torch.Tensor of shape (batch_size, (1,) * len(data_shape))): Times
            X (torch.Tensor of shape (batch_size, *data_shape)): States

        Returns:
            Y (torch.Tensor of shape (batch_size, *data_shape)): New states
        """

        if self.net_is_score:
            return self.v(t, x) - self.gamma_gamma_dot(t) * self.net(t, x)
        else:
            return self.v(t, x) + self.gamma_dot(t) * self.net(t, x)

    def compute_div_module(self, module_name, t, x, return_f_val, e, make_e):
        """Compute div(m)(t, x) where m is either v or s

        Args:
            module_name (str): Either 'v' or 's'
            t (torch.Tensor of shape (batch_size, (1,) * len(data_shape))): Times
            X (torch.Tensor of shape (batch_size, *data_shape)): States
            return_f_val (bool): Return the value of b(t, x) (default is False)
            e (torch.Tensor of shape (batch_size, *data_shape)): Noise for Hutchison's estimator
            make_e (function taking a shape and device as input): Make the e vector from x's shape

        Returns:
            if return_f_val:
                Y (torch.Tensor of shape (batch_size, *data_shape)): New states
                div (torch.Tensor of shape (batch_size,)): Divergence value
            otherwise
                div (torch.Tensor of shape (batch_size,)): Divergence value
        """

        # Get the module
        m = getattr(self, module_name)
        # Compute the div
        if hasattr(m, 'compute_div'):
            if e is not None:
                return m.compute_div(t, x, return_f_val=return_f_val, e=e, make_e=make_e)
            else:
                return m.compute_div(t, x, return_f_val=return_f_val)
        else:
            # Check if m_div_fn exists
            div_fn_name = module_name + '_div_fn'
            if not hasattr(self, div_fn_name):
                setattr(self, div_fn_name, compute_div_fn(m))
            if not hasattr(self, div_fn_name + '_approx'):
                setattr(self, div_fn_name + '_approx', compute_div_approx_fn(m))
            if e is not None:
                return getattr(self, div_fn_name)(t, x, e=e, return_f_val=return_f_val)
            else:
                return getattr(self, div_fn_name + '_approx')(t, x, return_f_val=return_f_val)

    def compute_div(self, t, x, return_f_val=False, e=None, make_e=None):
        """Compute div(b)(t, x)

        Args:
            t (torch.Tensor of shape (batch_size, (1,) * len(data_shape))): Times
            X (torch.Tensor of shape (batch_size, *data_shape)): States
            return_f_val (bool): Return the value of b(t, x) (default is False)
            e (torch.Tensor of shape (batch_size, *data_shape)): Noise for Hutchison's estimator
            make_e (function taking a shape and device as input): Make the e vector from x's shape

        Returns:
            if return_f_val:
                Y (torch.Tensor of shape (batch_size, *data_shape)): New states
                div (torch.Tensor of shape (batch_size,)): Divergence value
            otherwise
                div (torch.Tensor of shape (batch_size,)): Divergence value
        """

        if return_f_val:
            # Compute the divergence of v
            v_val, v_div = self.compute_div_module('v', t, x, return_f_val=True, e=e, make_e=e)
            # Compute the divergence of the net
            net_val, net_div = self.compute_div_module('net', t, x, return_f_val=True, e=e, make_e=e)
            # Compute the final values
            if self.net_is_score:
                gamma_gamma_dot_t = self.gamma_gamma_dot(t)
                f_val = v_val - gamma_gamma_dot_t * net_val
                div = v_div - gamma_gamma_dot_t.flatten() * net_div
            else:
                gamma_dot_t = self.gamma_dot(t)
                f_val = v_val + gamma_dot_t * net_val
                div = v_div + gamma_dot_t.flatten() * net_div
            return f_val, div
        else:
            # Compute the divergence of v
            v_div = self.compute_div_module('v', t, x, return_f_val=False, e=e, make_e=e)
            # Compute the divergence of s
            net_div = self.compute_div_module('net', t, x, return_f_val=False, e=e, make_e=e)
            # Compute the final values
            if self.net_is_score:
                div = v_div - self.gamma_gamma_dot(t).flatten() * net_div
            else:
                div = v_div + self.gamma_dot(t).flatten() * net_div
            return div


class SDEFromVelocityAndScore(torch.nn.Module):
    """Define a SDE using v and s"""

    noise_type = 'additive'
    sde_type = 'stratonovich'

    def __init__(self, v, s, epsilon, gamma_gamma_dot, reverse=False):
        """Constructor for SDEFromVelocityAndScore (Equations (2.16) and (2.19))

        This is for torchsde

        Args:
            v (torch.nn.Module): Velocity field
            s (torch.nn.Module): Score function
            epsilon (float): Value of epsilon
            gamma_gamma_dot (function): Returns gamma(t) * \\dot{gamma}(t)
            reverse (bool): Whether to model the reverse SDE (default is False)
        """

        # Call the parent's constructor
        super(SDEFromVelocityAndScore, self).__init__()
        # Store the neural networks
        self.v = v
        self.s = s
        # Store epsilon
        self.epsilon = epsilon
        self.sqrt_two_epsilon = torch.tensor(math.sqrt(2 * self.epsilon))
        # Store the function gamma(t) * \dot{gamma}(t)
        self.gamma_gamma_dot = gamma_gamma_dot
        # Whether we want the reverse SDE
        self.reverse = reverse

    def f(self, t, y):
        """Define the drift"""

        if self.reverse:
            ret = self.v(1.0 - t, y) - (self.epsilon + self.gamma_gamma_dot(1.0 - t)) * self.s(1.0 - t, y)
            ret *= -1
        else:
            ret = self.v(t, y) + (self.epsilon - self.gamma_gamma_dot(t)) * self.s(t, y)
        return ret

    def g(self, t, y):
        """Define the diffusion"""

        return self.sqrt_two_epsilon * torch.ones_like(y).unsqueeze(-1)
