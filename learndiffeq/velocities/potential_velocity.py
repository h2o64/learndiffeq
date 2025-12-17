# Make a custom velocity network for potential based velocity field

# Librairies
from ..common.utils import compute_div_fn
from .simple import make_simple_velocity
import torch

class ScoreBasedVelocity(torch.nn.Module):
    """The velocity field is modeled as a linear combination between x and s(t,x)

            b(t,x) = coef1(t) * x + coef2(t) * net(t,x)

    where coef1 and coef2 are derived from the spatially independent one-sided interpolant,
    and net can be either the score s or the denoiser eta. Here net is an instance of
    make_simple_velocity."""

    def __init__(self, dim, mean_rho1, f=None, net=None, net_type='score', **velocity_args):
        """Constructor

        Args:
            dim (int): Dimension of the data
            mean_rho1 (tensor.Tensor of shape (dim,)): Mean of the target distribution
            f (function): Function (eg. gradient of a potential)
                if f is not None, then s(t,x) is replaced by
                    s(t,x) = (1 - t^{\\eta}) * n(t,x) + t^{\\eta} * f(x)
                in the formula above. Here n(t,x) is an instance of ParticlesVelocity.
                (Default is None)
            s (function): Function to specify s(t,x) (override existing networks if not None)
            time_scaling (float): Power scaling for time (default is 1.)
            velocity_args (dict): Arguments for make_simple_velocity
        """

        # Call the parent constructor
        super(ScoreBasedVelocity, self).__init__()
        # Make the velocity fields
        self.dim = dim
        if net is None:
            self.net = make_simple_velocity(dim=dim, **velocity_args)
        else:
            self.net = net
        self.net_is_score = net_type == 'score'
        self.f = f
        # Save the interpolant
        self.int = None
        self.mean_rho1 = mean_rho1
        # Compute the divergence
        self.eval_net_div_fn = compute_div_fn(self.eval_net, return_f_val=True)

    def compute_coefs(self, t):
        """Compute coef1 and coef1

        Args:
            t (torch.Tensor): Time

        Returns:
            coef1 (torch.Tensor of same shape as t) : coef1
            coef2 (torch.Tensor of same shape as t) : coef2
        """

        # Compute the first coefficient
        coef1 = self.int.beta_dot(t) / self.int.beta(t)
        # Compute the second coefficient
        if self.net_is_score:
            coef2 = coef1 * torch.square(self.int.alpha(t)) - self.int.alpha_squared_dot(t)
        else:
            coef2 = coef1 * self.int.alpha(t) - self.int.alpha_dot(t)
        # Return both coefficients
        return coef1, coef2

    def eval_net(self, t, X):
        """Evaluate s(t,x)

        The output changes depending if f is None or not.

        Args:
            t (torch.Tensor of shape (batch_size, 1)): Times
            X (torch.Tensor of shape (batch_size, dim)): State

        Returns:
            Y (torch.Tensor of shape (batch_size, dim)): New state
        """

        if self.f is not None:
            return self.net(t, X) + self.f(X)
        else:
            return self.net(t, X)

    def forward(self, t, X):
        """Compute b(t, X)

        Args:
            t (torch.Tensor of shape (batch_size, 1,)): Times
            X (torch.Tensor of shape (batch_size, dim)): State

        Returns:
            Y (torch.Tensor of shape (batch_size, dim)): State
        """

        # Get the coefficients
        coef1, coef2 = self.compute_coefs(t)
        # Compute the velocity field
        zero_t = torch.zeros_like(t)
        ones_t = torch.ones_like(t)
        # Compute the state
        ret = coef1 * X + coef2 * self.eval_net(t, X)
        # Compute the edge conditions
        ret = torch.where(t == 0.0, self.int.alpha_dot(zero_t) * X + self.int.beta_dot(zero_t)
                          * self.mean_rho1.unsqueeze(0).expand((X.shape[0], -1)), ret)
        ret = torch.where(t == 1.0, self.int.beta_dot(ones_t) * X, ret)
        return ret

    def compute_div(self, t, X, return_f_val=False, e=None, make_e=None):
        """Compute div(b)(t, X)

        WARNING: Hutchinson’s trace estimator is not supported

        Args:
            t (torch.Tensor of shape (batch_size, 1)): Times
            X (torch.Tensor of shape (batch_size, dim)): Particles
            return_f_val (bool): Return the value of b(t, X) (default is False)
            e (torch.Tensor of shape (batch_size, dim)): Noise for Hutchison's estimator
            make_e (function taking a shape and device as input): Make the e vector from X's shape

        Returns:
            if return_f_val:
                Y (torch.Tensor of shape (batch_size, dim)): New state
                div (torch.Tensor of shape (batch_size,)): Divergence value
            otherwise
                div (torch.Tensor of shape (batch_size,)): Divergence value
        """

        # WARNING: Hutchinson’s trace estimator is not supported
        if e is not None or make_e is not None:
            raise NotImplementedError("Hutchinson’s trace estimator is not supported")
        # Reshape t if needed
        if len(t.size()) == 0:
            t = t.view((1, 1)).expand((X.shape[0], -1))
        # Get the coefficients
        coef1, coef2 = self.compute_coefs(t)
        # Compute the divergence
        zero_t = torch.zeros_like(t)
        ones_t = torch.ones_like(t)
        if return_f_val:
            # Compute the divergence of the network
            net_val, net_div = self.eval_net_div_fn(t, X)
            # Compute the divergence of the other term
            x_val, x_div = X, self.dim
            # Compute the linear combination
            val = coef1 * x_val + coef2 * net_val
            div = coef1.flatten() * x_div + coef2.flatten() * net_div
            # Be aware oof the t == 0 case
            val = torch.where(t == 0.0, self.int.alpha_dot(zero_t) * X + self.int.beta_dot(zero_t)
                              * self.mean_rho1.unsqueeze(0).expand((X.shape[0], -1)), val)
            val = torch.where(t == 1.0, self.int.beta_dot(ones_t) * X, val)
            div = torch.where(t.flatten() == 0.0, self.int.alpha_dot(zero_t).flatten() * self.dim, div)
            div = torch.where(t.flatten() == 1.0, self.int.beta_dot(ones_t).flatten() * self.dim, div)
            return val, div
        else:
            # Compute the divergence of the network
            net_div = self.eval_net_div_fn(t, X)[1]
            # Compute the divergence of the other term
            x_div = self.dim
            # Compute the linear combination
            div = coef1.flatten() * x_div + coef2.flatten() * net_div
            # Be aware oof the t == 0 case
            div = torch.where(t.flatten() == 0.0, self.int.alpha_dot(zero_t).flatten() * self.dim, div)
            div = torch.where(t.flatten() == 1.0, self.int.beta_dot(ones_t).flatten() * self.dim, div)
            return div

    def _apply(self, fn):
        new_self = super(ScoreBasedVelocity, self)._apply(fn)
        new_self.mean_rho1 = fn(new_self.mean_rho1)
        return new_self
