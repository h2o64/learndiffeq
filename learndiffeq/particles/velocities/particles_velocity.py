# Make a custom velocity networks for particle systems

# Librairies
from ...common.utils import compute_div_fn, compute_div_fn, compute_div_approx_fn, compute_div_with_a_fn, compute_div_approx_with_a_fn
from ...velocities.simple import make_mlp, make_simple_velocity
from .egnn import EGNN_dynamics
from .kernel import KernelDynamics
from ..utils import move_to_torus, make_mask, compute_combinations, reshape_tri
import torch

# Use the right vmap function depending on Pytorch's version
batch_trace = torch.vmap(torch.trace)


def make_particles_velocity(data_shape, name, hidden_layers, **kwargs):
    """Simple function to build one of the particle velocity field

    Args:
        data_shape (tuple of int): Shape of the data (likely (n_particles, dim_phys))
        name (str): Short name for the class to use (either of 'topk','equivariant','potential','pis','score' or 'default')
        hidden_layers (tuple or list of int): Hidden layers for the inner MLP

    Returns:
        b (torch.nn.Module): Velocity field
    """

    # Build a velocity field
    if 'equivariant' in name:
        return ParticlesVelocityEquivariant(data_shape=data_shape, **kwargs)
    elif 'kernel' in name:
        return ParticlesVelocityKernel(data_shape=data_shape, **kwargs)
    else:
        raise ValueError('Velocity type {} not found.'.format(name))

class ParticlesVelocityEquivariant(torch.nn.Module):
    """Equivariant architecture for the velocity field"""

    def __init__(self, data_shape, **kwargs):
        """Constructor

        Args:
            data_shape (tuple of int): Shape of the data (n_particles, dim_phys)
            kwargs (dict): Arguments for EGNN_dynamics
        """

        # Call the parent constructor
        super(ParticlesVelocityEquivariant, self).__init__()
        # Build the EGNN
        self.n_particles = data_shape[0]
        self.dim_phys = data_shape[1]
        self.egnn = EGNN_dynamics(n_particles=data_shape[0], n_dimension=data_shape[1], **kwargs)
        # Build the divergence
        self.flat_div_fn = compute_div_fn(self.forward_flat, return_f_val=True)
        self.flat_div_approx_fn = compute_div_approx_fn(self.forward_flat, return_f_val=True)
        self.flat_with_a_div_fn = compute_div_with_a_fn(self.forward_flat, return_f_val=True)
        self.flat_with_a_div_approx_fn = compute_div_approx_with_a_fn(self.forward_flat, return_f_val=True)

    def forward(self, t, X, a=None):
        """Compute b(t, X)

        Args:
            t (torch.Tensor of shape (batch_size, 1, 1)): Times
            X (torch.Tensor of shape (batch_size, n_particles, dim_phys)): Particles
            a (torch.Tensor of shape (batch_size, n_particles)): Species (default is None)

        Returns:
            Y (torch.Tensor of shape (batch_size, n_particles, dim_phys)): New state
        """

        return self.egnn(t, X, a=a)

    def forward_flat(self, t, X, a=None):
        """Compute b(t, X) where data_shape = (dim,) = (n_particles * dim_phys)

        Args:
            t (torch.Tensor of shape (batch_size, 1)): Times
            X (torch.Tensor of shape (batch_size, dim)): Particles
            a (torch.Tensor of shape (batch_size, n_particles)): Species (default is None)

        Returns:
            Y (torch.Tensor of shape (batch_size, dim)): New state
        """

        batch_size = X.shape[0]
        ret = self.forward(
            t.view((batch_size, 1, 1)),
            X.view((batch_size, self.n_particles, self.dim_phys)),
            a=a
        )
        return ret.view((batch_size, -1))

    def compute_div(self, t, X, a=None, return_f_val=False, e=None, make_e=None):
        """Compute div(b)(t, X)

        Args:
            t (torch.Tensor of shape (batch_size, 1, 1)): Times
            X (torch.Tensor of shape (batch_size, n_particles, dim_phys)): Particles
            a (torch.Tensor of shape (batch_size, n_particles)): Species (default is None)
            return_f_val (bool): Return the value of b(t, X) (default is False)
            e (torch.Tensor of shape (batch_size, n_particles, dim_phys)): Noise for Hutchison's estimator
            make_e (function taking a shape and device as input): Make the e vector from X's shape

        Returns:
            if return_f_val:
                Y (torch.Tensor of shape (batch_size, n_particles, dim_phys)): New state
                div (torch.Tensor of shape (batch_size,)): Divergence value
            otherwise
                div (torch.Tensor of shape (batch_size,)): Divergence value
        """

        # Reshape t and X
        if len(t.shape) == 0:
            t = t.repeat((X.shape[0]))
        t = t.view((X.shape[0], 1))
        X = X.view((X.shape[0], -1))
        # Evaluate the low-dim velocity field
        if e is not None and e.shape != X.shape:
            e = make_e(X.shape, X.device)
        # Compute the divergence
        if return_f_val:
            # Compute the low-dim divergence
            if e is not None:
                if a is None:
                    f_val, div = self.flat_div_approx_fn(t, X, e)
                else:
                    f_val, div = self.flat_with_a_div_approx_fn(t, X, a, e)
            else:
                if a is None:
                    f_val, div = self.flat_div_fn(t, X)
                else:
                    f_val, div = self.flat_with_a_div_fn(t, X, a)
            return f_val.view((X.shape[0], self.n_particles, self.dim_phys)), div
        else:
            # Compute the low-dim divergence
            if e is not None:
                if a is None:
                    div = self.flat_div_approx_fn(t, X, e)[1]
                else:
                    div = self.flat_with_a_div_approx_fn(t, X, a, e)[1]
            else:
                if a is None:
                    div = self.flat_div_fn(t, X)[1]
                else:
                    div = self.flat_with_a_div_fn(t, X, a)[1]
            # Compute the divergence
            return div


class ParticlesVelocityKernel(torch.nn.Module):
    """Kernel architecture for the velocity field"""

    def __init__(self, data_shape, n_species, d_max=8, n_rbfs=50, gamma_init=0.5,
                 n_rbfs_time=10, gamma_time_init=0.3, **kwargs):
        """Constructor

        Args:
            data_shape (tuple of int): Shape of the data (n_particles, dim_phys)
            d_max (int): Maximum value of mu (default is 8)
            n_rbfs (int): Number of kernels for the states (default is 50)
            gamma_init (float): Initial value of gamma for the states (default is 0.5)
            n_rbfs_time (int): Number of kernels for the time (default is 10)
            gamma_time_init (float): Initial value of gamma for the time (default is 0.3)
            kwargs (dict): Arguments for KernelDynamics
        """

        # Call the parent constructor
        super(ParticlesVelocityKernel, self).__init__()
        # Build the kernel velocity
        self.n_particles = data_shape[0]
        self.dim_phys = data_shape[1]
        self.n_species = n_species
        mus = torch.linspace(0, d_max, n_rbfs)
        gammas = gamma_init * torch.ones(n_rbfs)
        mus_time = torch.linspace(0, 1, n_rbfs_time)
        gammas_time = gamma_time_init * torch.ones(n_rbfs_time)
        self.model = KernelDynamics(
            n_particles=self.n_particles,
            n_dimensions=self.dim_phys,
            n_species=self.n_species,
            mus=mus,
            gammas=gammas,
            mus_time=mus_time,
            gammas_time=gammas_time,
            optimize_d_gammas=True,
            optimize_t_gammas=True,
            **kwargs
        )

    def forward(self, t, X, a):
        """Compute b(t, X)

        Args:
            t (torch.Tensor of shape (batch_size, 1, 1)): Times
            X (torch.Tensor of shape (batch_size, n_particles, dim_phys)): Particles
            a (torch.Tensor of shape (batch_size, n_particles)): Species

        Returns:
            Y (torch.Tensor of shape (batch_size, n_particles, dim_phys)): New state
        """

        return self.model(t, X, a, compute_divergence=False)

    def compute_div(self, t, X, a, return_f_val=False, **kwargs):
        """Compute div(b)(t, X)

        Args:
            t (torch.Tensor of shape (batch_size, 1, 1)): Times
            X (torch.Tensor of shape (batch_size, n_particles, dim_phys)): Particles
            a (torch.Tensor of shape (batch_size, n_particles)): Species
            return_f_val (bool): Return the value of b(t, X) (default is False)

        Returns:
            if return_f_val:
                Y (torch.Tensor of shape (batch_size, n_particles, dim_phys)): New state
                div (torch.Tensor of shape (batch_size,)): Divergence value
            otherwise
                div (torch.Tensor of shape (batch_size,)): Divergence value
        """

        # Reshape t and X
        if len(t.shape) == 0:
            t = t.repeat((X.shape[0])).view((-1, 1, 1))
        # Compute the divergence
        f_val, neg_div = self.model(t, X, a, compute_divergence=True)
        if return_f_val:
            return f_val, -neg_div.squeeze(-1)
        else:
            return -neg_div.squeeze(-1)