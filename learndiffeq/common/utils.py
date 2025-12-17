# Various helpers

# Libraries
import torch
import ot
import numpy as np

# Vectorized trace function
trace_vec = torch.func.vmap(torch.trace)

# Pytorch optimizers
torch_optimizers = {
    'adam': torch.optim.Adam,
    'adamw': torch.optim.AdamW,
    'nadam': torch.optim.NAdam,
    'sgd': torch.optim.SGD
}


class ReshapeTransform:
    """Simple alternative to torch.distributions.transforms.ReshapeTransform"""

    def __init__(self, dist, in_shape, out_shape, sum_last=False):
        self.dist = dist
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.len_out_shape = len(out_shape)
        self.sum_last = sum_last

    def log_prob(self, value):
        batch_shape = value.shape[:-self.len_out_shape]
        if self.sum_last:
            return self.dist.log_prob(value.view(-1, *self.in_shape)).sum(-1).view(batch_shape)
        else:
            return self.dist.log_prob(value.view(-1, *self.in_shape)).view(batch_shape)

    def sample(self, sample_shape=torch.Size()):
        return self.dist.sample(sample_shape=sample_shape).view((*sample_shape, *self.out_shape))


def compute_div_fn(f, return_f_val=False):
    """Compute divergence of f in a jax fashion

    Args:
        f (function): Function taking (t,x) as input
        return_f_val (bool): Whether to return f(t,x)

    Return
        div_fn (function): Function taking (t, x) as input where
                t (torch.Tensor of shape (batch_size, 1)): Times
                x (torch.Tensor of shape (batch_size, dim)): States
            and returning
                if return_f_val is True
                    f_val (torch.Tensor of shape (batch_size, dim)): Value of f(t,x)
                    div (torch.Tensor of shape (batch_size,)): Divergence of f(t,.) at x
                otherwise
                    div (torch.Tensor of shape (batch_size,)): Divergence of f(t,.) at x
    """

    # Make a unit version of f
    def f_unit(t, x):
        return f(t.unsqueeze(0), x.unsqueeze(0)).squeeze(0)
    if return_f_val:
        # Make a duplicate version of f
        def f_alt(t, x):
            ret = f_unit(t, x)
            return ret, ret
        jac_fn = torch.func.jacrev(f_alt, argnums=1, has_aux=True)

        def jac_trace_fn(t, x):
            jac, f_val = jac_fn(t, x)
            return f_val, torch.trace(jac)
        div_fn = torch.vmap(jac_trace_fn, randomness='different')
    else:
        jac_fn = torch.func.jacrev(f_unit, argnums=1)
        div_fn = torch.vmap(lambda t, x: torch.trace(jac_fn(t, x)), randomness='different')
    return div_fn


def compute_div_with_a_fn(f, return_f_val=False):
    """Compute divergence of f in a jax fashion

    Args:
        f (function): Function taking (t,x,a) as input
        return_f_val (bool): Whether to return f(t,x,a)

    Return
        div_fn (function): Function taking (t, x) as input where
                t (torch.Tensor of shape (batch_size, 1)): Times
                x (torch.Tensor of shape (batch_size, dim)): States
                a (torch.Tensor of shape (batch_size, *shape)): Species
            and returning
                if return_f_val is True
                    f_val (torch.Tensor of shape (batch_size, dim)): Value of f(t, x, a)
                    div (torch.Tensor of shape (batch_size,)): Divergence of f(t,.,a) at x
                otherwise
                    div (torch.Tensor of shape (batch_size,)): Divergence of f(t,.,a) at x
    """

    # Make a unit version of f
    def f_unit(t, x, a):
        return f(t.unsqueeze(0), x.unsqueeze(0), a.unsqueeze(0)).squeeze(0)
    if return_f_val:
        # Make a duplicate version of f
        def f_alt(t, x, a):
            ret = f_unit(t, x, a)
            return ret, ret
        jac_fn = torch.func.jacrev(f_alt, argnums=1, has_aux=True)

        def jac_trace_fn(t, x, a):
            jac, f_val = jac_fn(t, x, a)
            return f_val, torch.trace(jac)
        div_fn = torch.vmap(jac_trace_fn, randomness='different')
    else:
        jac_fn = torch.func.jacrev(f_unit, argnums=1)
        div_fn = torch.vmap(lambda t, x, a: torch.trace(jac_fn(t, x, a)), randomness='different')
    return div_fn


def compute_div_approx_fn(f, return_f_val=False):
    """Compute divergence of f using the Hutchinson’s trace estimator in a Jax fashion

    Args:
        f (function): Function taking (t,x) as input
        return_f_val (bool): Whether to return f(t,x)

    Return
        div_fn (function): Function taking (t, x) as input where
                t (torch.Tensor of shape (batch_size, 1)): Times
                x (torch.Tensor of shape (batch_size, dim)): States
                e (torch.Tensor of shape (batch_size, dim)): Noise
            and returning
                if return_f_val is True
                    f_val (torch.Tensor of shape (batch_size, dim)): Value of f(t,x)
                    div (torch.Tensor of shape (batch_size,)): Divergence of f(t,.) at x
                otherwise
                    div (torch.Tensor of shape (batch_size,)): Divergence of f(t,.) at x
    """
    def div_fn(t, x, e):
        f_val, e_dzdx = torch.func.jvp(lambda x: f(t, x), (x,), (e,))  # Compute directional derivative
        approx_tr_dzdx = (e * e_dzdx).sum(dim=-1)  # Compute the scalar product
        if return_f_val:
            return f_val, approx_tr_dzdx
        else:
            return approx_tr_dzdx
    return div_fn


def compute_div_approx_with_a_fn(f, return_f_val=False):
    """Compute divergence of f using the Hutchinson’s trace estimator in a Jax fashion

    Args:
        f (function): Function taking (t,x,a) as input
        return_f_val (bool): Whether to return f(t,x,a)

    Return
        div_fn (function): Function taking (t, x) as input where
                t (torch.Tensor of shape (batch_size, 1)): Times
                x (torch.Tensor of shape (batch_size, dim)): States
                a (torch.Tensor of shape (batch_size, *shape)): Species
                e (torch.Tensor of shape (batch_size, dim)): Noise
            and returning
                if return_f_val is True
                    f_val (torch.Tensor of shape (batch_size, dim)): Value of f(t,x,a)
                    div (torch.Tensor of shape (batch_size,)): Divergence of f(t,.,a) at x
                otherwise
                    div (torch.Tensor of shape (batch_size,)): Divergence of f(t,.,a) at x
    """
    def div_fn(t, x, a, e):
        f_val, e_dzdx = torch.func.jvp(lambda x: f(t, x, a), (x,), (e,))  # Compute directional derivative
        approx_tr_dzdx = (e * e_dzdx).sum(dim=-1)  # Compute the scalar product
        if return_f_val:
            return f_val, approx_tr_dzdx
        else:
            return approx_tr_dzdx
    return div_fn


def sample_ot_coupling(x0, x1):
    """Sample from the OT coupling

    Inputs:
        - x0 (torch.Tensor of shape (batch_size, dim)): samples from rho0
        - x1 (torch.Tensor of shape (batch_size, n_particles, dim_phys)): samples from rho1

    Outputs:
        - x0, x1 (tuple of torch.Tensor of shape (batch_size, dim)): Samples from the coupling
    """

    # Resample x0, x1 according to transport matrix
    a1, b1 = ot.unif(x0.size()[0]), ot.unif(x1.size()[0])
    M = torch.square(torch.cdist(x0, x1))
    M = M / M.max()
    pi = ot.emd(a1, b1, M.detach().cpu().numpy())
    # Sample random interpolations on pi
    p = pi.flatten()
    p = p / p.sum()
    choices = np.random.choice(pi.shape[0] * pi.shape[1], p=p, size=x1.shape[0])
    i, j = np.divmod(choices, pi.shape[1])
    return x0[i], x1[j]


class VelocityWithKwargs(torch.nn.Module):
    def __init__(self, net, **kwargs):
        super().__init__()
        self.net = net
        self.kwargs = kwargs

    def forward(self, t, x):
        return self.net(t, x, **self.kwargs)

    def compute_div(self, *args, **kwargs):
        return self.net.compute_div(*args, **kwargs, **self.kwargs)


class ModuleAndDivergence(torch.nn.Module):
    def __init__(self, module, data_shape, approx=False, randemaker=False, reverse_time=False,
                 force_automatic_div=False):
        """Compute the drift for the continuity equation

        Args:
            module (torch.nn.Module): learned velocity field
            data_shape (tuple of int): Shape of the data
            approx (bool): Whether to use the approximate divergence (default is False)
            randemaker (bool): Whether to use a randemaker variable (default is False)
            reverse_time (bool): Whether to model the reverse continuity equation (default is False)
            force_automatic_div (bool): Whether to skip self.compute_div (default is False)
        """

        super(ModuleAndDivergence, self).__init__()
        self.module = module
        self.data_shape = data_shape
        self.data_shape_ones = (1,) * len(data_shape)
        self.module_has_div = hasattr(self.module, 'compute_div') and not force_automatic_div
        self.approx = approx
        if not self.module_has_div:
            if self.approx:
                self.module_div_fn = compute_div_approx_fn(self.module_flatten, return_f_val=True)
            else:
                self.module_div_fn = compute_div_fn(self.module_flatten, return_f_val=True)
        self.e = None
        self.randemaker = randemaker

    def make_e(self, shape, device):
        """Generate e from x's shape"""

        if self.randemaker:
            self.e = torch.randint(low=0, high=2, size=shape).to(device) * 2 - 1
        else:
            self.e = torch.randn(shape, device=device)
        return self.e

    def module_flatten(self, t, x):
        batch_size = x.shape[0]
        ret = self.module(t.view((-1, *self.data_shape_ones)), x.view((batch_size, *self.data_shape)))
        return ret.view((batch_size, -1))

    def forward(self, t, states):
        """Forward call for the drift of the continuity equation  b(t, x)

        Args:
            t (float): time
            states (tuple of torch.Tensor) :
                * states[0] is the x of shape (batch_size, *data_shape)
                * states[1] is the difference between log p(t,x) and log p_0(x) (TBC.) which is of shape (batch_size,)
                See eq (4) of [1] (https://arxiv.org/abs/1810.01367)
        Outputs:
            drifts (tuple of torch.Tensor)
                * drifts[0] is the backward drift of the ODE (ie. b(t, x))
                * drifts[1] is -div(b)(t, x) for the integration of the continuity equation
        """

        # Set t to have the right shape
        t = t * torch.ones((states[0].shape[0], *(1,) * (len(states[0].shape)-1)), device=states[0].device)
        # Set e if not done yet
        if self.approx and self.e is None:
            self.make_e(shape=states[0].shape, device=states[0].device)
        # Compute the div
        if self.module_has_div:
            if self.approx:
                f_val, div = self.module.compute_div(
                    t, states[0], return_f_val=True, e=self.e, make_e=lambda x: self.make_e(x))
            else:
                f_val, div = self.module.compute_div(t, states[0], return_f_val=True)
        else:
            x = states[0]
            batch_size = x.shape[0]
            if self.approx:
                f_val, div = self.module_div_fn(
                    t.view((batch_size, 1)),
                    x.view((batch_size, -1)),
                    self.e.view((batch_size, -1))
                )
            else:
                f_val, div = self.module_div_fn(
                    t.view((batch_size, 1)),
                    x.view((batch_size, -1))
                )
            f_val = f_val.view((batch_size, *self.data_shape))
        # Return the divergence
        return f_val, -div
