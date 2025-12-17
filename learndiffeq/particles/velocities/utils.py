# Helper functions

# Libraries
import torch
from ..distributions.lennard_jones import LennardJones, ModifiedLennardJones, ShiftedLennardJones
from ..distributions.harmonic import Harmonic
from ..distributions.utils import SumDistsWithGrad
from ..utils import clip_function, clip_function_and_log_jacobian


def compute_grad_U(U, x, *args, **kwargs):
    """Compute the gradient of U automatically"""
    requires_grad = x.requires_grad
    with torch.set_grad_enabled(True):
        x.requires_grad_(True)
        outputs = U(x, *args, **kwargs)
        gradient = torch.autograd.grad(outputs.sum(), x, create_graph=True, retain_graph=True)[0]
    x.requires_grad_(requires_grad)
    return gradient


def make_target_dist_lj(n_particles, dim_phys, use_modified_lj, use_harmonic_pot, r_min_sq=0.0, harm_coef=1.0, device=None, fn=None):
    """Make the target Lennard-Jones distribution

    Args:
            n_particles (int): Number of particles
            dim_phys (int): Dimension of a single particle
            use_modified_lj (bool): Whether to use modified Lennard-Jones
            use_harmonic_pot (bool): Whether to use the harmonic potential
            r_min_sq (float): Minimum square radius (default is 0.0)
            harm_coef (float): Coefficient in front of the harmonic potential (default is 1.0)
            device (torch.Device): Device to use for computations (default is None)
            fn (function): Function to apply on the distribution as nn.Module (default is None)

    Returns:
            target (Distribution): The target distribution
    """

    # Make the target distribution
    if use_modified_lj:
        target_lj = ModifiedLennardJones(n_particles=n_particles, dim_phys=dim_phys, r_min_sq=r_min_sq)
    else:
        target_lj = LennardJones(n_particles=n_particles, dim_phys=dim_phys, r_min_sq=r_min_sq)
    if use_harmonic_pot:
        target = SumDistsWithGrad(target_lj, Harmonic(n_particles=n_particles, dim_phys=dim_phys, k=harm_coef))
    else:
        target = target_lj
    if device is not None:
        target = target.to(device)
    if fn is not None:
        target = target._apply(fn)
    return target