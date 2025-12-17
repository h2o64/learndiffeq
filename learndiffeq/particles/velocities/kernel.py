# Taken from https://raw.githubusercontent.com/noegroup/bgflow/master/bgflow/nn/flow/dynamics/kernel_dynamic.py
import torch
import numpy as np
from ..utils import gram_species, gram_species_torus


def distances_from_vectors(r, eps=1e-6):
    """
    Computes the all-distance matrix from given distance vectors.

    Parameters
    ----------
    r : torch.Tensor
        Matrix of all distance vectors r.
        Tensor of shape `[n_batch, n_particles, n_other_particles, n_dimensions]`
    eps : Small real number.
        Regularizer to avoid division by zero.

    Returns
    -------
    d : torch.Tensor
        All-distance matrix d.
        Tensor of shape `[n_batch, n_particles, n_other_particles]`.
    """
    return (r.pow(2).sum(dim=-1) + eps).sqrt()


def rbf_kernels(d: torch.Tensor, mu, neg_log_gamma, derivative=False) -> torch.Tensor:
    """
    Takes a distance matrix `d` of shape

        `[n_batch, n_particles, n_particles, 1]`

    and maps it onto a normalized radial basis function (RBF) kernel
    representation of shape

        `[n_batch, n_particles, n_particles, n_kernels]`

    via

        `d_{ij} -> f_{ij}

    where

        `f_{ij} = (g_{ij1}, ..., g_{ijK}) / sum_{k} g_{ijk}

    and

        `g_{ijk} = exp(-(d_{ij} - mu_{k})^{2} / gamma^{2})`.

    Parameters
    ----------
    d: PyTorch tensor
        distance matrix of shape `[n_batch, n_particles, n_particles, 1]`
    mu: PyTorch tensor / scalar
        Means of RBF kernels. Either of shape `[1, 1, 1, n_kernels]` or
        scalar
    neg_log_gamma: PyTorch tensor / scalar
        Negative logarithm of bandwidth of RBF kernels. Either same shape as `mu` or scalar.
    derivative: boolean
        Whether the derivative of the rbf kernels is computed.

    Returns
    -------
    kernels: PyTorch tensor
        RBF representation of distance matrix of shape
        `[n_batch, n_particles, n_particles, n_kernels]`
    dkernels: PyTorch tensor
        Corresponding derivatives of shape
        `[n_batch, n_particles, n_particles, n_kernels]`
    """
    inv_gamma = torch.exp(neg_log_gamma)
    rbfs = torch.exp(-(d - mu).pow(2) * inv_gamma.pow(2))
    srbfs = rbfs.sum(dim=-1, keepdim=True)
    kernels = rbfs / (1e-6 + srbfs)
    if derivative:
        drbfs = -2 * (d - mu) * inv_gamma.pow(2) * rbfs
        sdrbfs = drbfs.sum(dim=-1, keepdim=True)
        dkernels = drbfs / (1e-6 + srbfs) - rbfs * sdrbfs / (1e-6 + srbfs ** 2)
    else:
        dkernels = None
    return kernels, dkernels


class KernelDynamics(torch.nn.Module):
    """
    Equivariant dynamics functions.
    Equivariant dynamics functions that allows an efficient
    and exact divergence computation :footcite:`Khler2020EquivariantFE`.

    References
    ----------
    .. footbibliography::

    """

    def __init__(self, n_particles, n_dimensions, n_species,
                 mus, gammas, L=None,
                 mus_time=None, gammas_time=None,
                 optimize_d_gammas=False,
                 optimize_t_gammas=False):
        super().__init__()
        self.L = L
        self._n_particles = n_particles
        self._n_dimensions = n_dimensions

        self._n_kernels = mus.shape[0]
        self.register_buffer('_mus', mus, persistent=False)
        self._neg_log_gammas = -torch.log(gammas)
        if optimize_d_gammas:
            self._neg_log_gammas = torch.nn.Parameter(self._neg_log_gammas)

        self.register_buffer('_mus_time', mus_time, persistent=False)
        self._neg_log_gammas_time = -torch.log(gammas_time)
        if optimize_t_gammas:
            self._neg_log_gammas_time = torch.nn.Parameter(self._neg_log_gammas_time)

        if self._mus_time is None:
            self._n_out = 1
        else:
            assert self._neg_log_gammas_time is not None and self._neg_log_gammas_time.shape[0] == self._mus_time.shape[
                0]
            self._n_out = self._mus_time.shape[0]

        self.n_species_couples = int(n_species * (n_species + 1) / 2)
        self.register_buffer(
            'species_couples_map', torch.zeros((n_species, n_species)).long(), persistent=False
        )
        c = 0
        for i in range(n_species):
            for j in range(n_species):
                if i <= j:
                    self.species_couples_map[i, j] = c
                    self.species_couples_map[j, i] = c
                    c += 1

        self._weights = torch.nn.Parameter(
            torch.randn((self.n_species_couples, self._n_kernels, self._n_out)) * np.sqrt(1. / self._n_kernels)
        )
        self._bias = torch.nn.Parameter(
            torch.zeros((self.n_species_couples, self._n_out))
        )

        self._importance = torch.nn.Parameter(
            torch.zeros((self._n_kernels,))
        )

    def _force_mag(self, t, a, d, derivative=False):

        importance = self._importance

        rbfs, d_rbfs = rbf_kernels(d, self._mus, self._neg_log_gammas, derivative=derivative)

        couple_id = self.species_couples_map[a[..., 0], a[..., 1]]
        force_mag = rbfs + importance.pow(2).view(1, 1, 1, -1)
        force_mag = torch.matmul(force_mag.unsqueeze(-2), self._weights[couple_id]).squeeze(-2)
        force_mag += self._bias[couple_id]
        if derivative:
            d_force_mag = torch.matmul(d_rbfs.unsqueeze(-2), self._weights[couple_id]).squeeze(-2)
        else:
            d_force_mag = None
        if self._mus_time is not None:
            trbfs, _ = rbf_kernels(t, self._mus_time, self._neg_log_gammas_time)
            force_mag = (force_mag * trbfs).sum(dim=-1, keepdim=True)
            if derivative:
                d_force_mag = (d_force_mag * trbfs).sum(dim=-1, keepdim=True)
        return force_mag, d_force_mag

    def forward(self, t, x, a, compute_divergence=True):
        """
        Computes the change of the system `dxs` at state `x` and
        time `t` due to the kernel dynamic. Furthermore, can also compute the exact change of log density
        which is equal to the divergence of the change.

        Parameters
        ----------
        t : PyTorch tensor
            The current time
        x : PyTorch tensor
            The current configuration of the system
        a : Pytorch tensor
            The current species within the configuration
        compute_divergence : boolean
            Whether the divergence is computed

        Returns
        -------
        forces, -divergence: PyTorch tensors
            The combined state update of shape `[n_batch, n_dimensions]`
            containing the state update of the system state `dx/dt`
            (`forces`) and the negative exact update of the log density (`-divergence`)

        """
        n_batch = x.shape[0]

        x = x.view(n_batch, self._n_particles, self._n_dimensions)
        # r = distance_vectors(x)
        # TODO: This can be optimized
        if self.L is not None:
            r, r_a = gram_species_torus(a, x, self.L)
        else:
            r, r_a = gram_species(a, x)
        r = r[:, torch.eye(self._n_particles, self._n_particles) == 0].view(
            -1, self._n_particles, self._n_particles - 1, self._n_dimensions
        )
        r_a = r_a[:, torch.eye(self._n_particles, self._n_particles) == 0].view(
            -1, self._n_particles, self._n_particles - 1, 2
        )

        d = distances_from_vectors(r).unsqueeze(-1)

        force_mag, d_force_mag = self._force_mag(t.unsqueeze(-1), r_a, d, derivative=compute_divergence)
        forces = (r * force_mag).sum(dim=-2)
        # forces = forces.view(n_batch, -1)

        if compute_divergence:
            divergence = (d * d_force_mag + self._n_dimensions * force_mag).view(n_batch, -1).sum(dim=-1)
            divergence = divergence.unsqueeze(-1)
            return forces, -divergence
        else:
            return forces
