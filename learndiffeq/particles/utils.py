# Utilities for working on particles

# Libraries
from itertools import count
from math import exp
import torch
import ot
try:
    from torch_linear_assignment import batch_linear_assignment
    linear_assignment_exists = True
except ImportError:
    linear_assignment_exists = False


def compute_pairwise_distances(X, Y):
    """Compute pairwise distances between two distinct tensors

    Args:
        X (torch.Tensor of shape (batch_size, n_particles, dim_phys)): Tensor of particles
        Y (torch.Tensor of shape (batch_size, n_particles, dim_phys)): Tensor of particles

    Returns:
        ret (torch.Tensor of shape (batch_size, n_particles, n_particles, dim_phys)) which
            verifies that ret[batch_id,i,j] = x_i - y_j
    """

    return X.unsqueeze(1) - Y.unsqueeze(2)


def compute_pairwise_distances_same(X, only_upper_tri=False):
    """Compute pairwise distances between items of a same tensor

    Args:
        X (torch.Tensor of shape (batch_size, n_particles, dim_phys)): Tensor of particles
        only_upper_tri (bool): Whether to return only the upper triangular part of the pairwise difference matrix
            (default is False)

    Returns:
        if only_upper_tri
            ret (torch.Tensor of shape (batch_size, n_particles * (n_particles-1) / 2, dim_phys)) which verifies that
                ret[batch_id,k] = x_i - x_j for some i,j where i > j
            Note that the indices i and j can be recovered using torch.triu_indices(X.shape[1], X.shape[1], offset=1, device=X.device)
        else
            ret (torch.Tensor of shape (batch_size, n_particles, n_particles, dim_phys)) which verifies that
                ret[batch_id,i,j] = x_i - x_j
    """

    # Generate indices for the upper triangular part
    row_indices, col_indices = torch.triu_indices(X.shape[1], X.shape[1], offset=1, device=X.device)
    # Compute the pairwise difference only for the upper triangular indices
    upper_tri = X[:, row_indices] - X[:, col_indices]
    if only_upper_tri:
        return upper_tri
    # Reshape the difference tensor into a matrix
    ret = torch.zeros((X.shape[0], X.shape[1], *X.shape[1:]), device=X.device)
    ret[:, row_indices, col_indices] = upper_tri
    # Reflect the upper triangular part to obtain the complete difference matrix
    ret = ret - ret.transpose(1, 2)
    return ret


def compute_pairwise_distances_species_same(a, X, only_upper_tri=False):
    """Compute pairwise distances between items of a same tensor and accounting for the particles species

    Args:
        a (torch.Tensor of shape (batch_size, n_particles)): Species of each particle
        X (torch.Tensor of shape (batch_size, n_particles, dim_phys)): Tensor of particles
        only_upper_tri (bool): Whether to return only the upper triangular part of the pairwise difference matrix
            (default is False)

    Returns:
        if only_upper_tri
            ret (torch.Tensor of shape (batch_size, n_particles * (n_particles-1) / 2, dim_phys)) which verifies that
                ret[batch_id,k] = x_i - x_j for some i,j where i > j
            ret_species (torch.Tensor of shape (batch_size, n_particles * (n_particles-1) / 2, 2)) which verifies that
                ret_species[batch_id,k] = (a[i], a[j]) for some i,j where i > j
            Note that the indices i and j can be recovered using torch.triu_indices(X.shape[1], X.shape[1], offset=1, device=X.device)
        else
            ret (torch.Tensor of shape (batch_size, n_particles, n_particles, dim_phys)) which verifies that
                ret[batch_id,i,j] = x_i - x_j
            ret_species (torch.Tensor of shape (batch_size, n_particles, n_particles, 2)) which verifies that
                ret_species[batch_id,i,j] = (a[i], a[j])
    """
    # Compute the pairwise difference
    ret = compute_pairwise_distances_same(X, only_upper_tri=only_upper_tri)
    # Return the species combinations
    if only_upper_tri:
        row_indices, col_indices = torch.triu_indices(X.shape[1], X.shape[1], offset=1, device=X.device)
        return ret, torch.stack([a[:, row_indices], a[:, col_indices]], dim=-1)
    else:
        return ret, torch.stack([
            torch.stack(torch.meshgrid([a[i], a[i]], indexing='xy'), dim=-1) for i in range(X.shape[0])
        ], dim=0)


def move_to_torus(G, L):
    """Move a difference matrix in a torus in a n-D torus (ie. [-width, width]^d).

    Args:
        G (torch.Tensor of any shape): Tensor with particles
        L (float): Length of the box
    Outputs:
        - ret (torch.Tensor of shape the same shape as G) which verifies that each element is in a torus
    """

    return G - L * torch.round(G / L)


def gram_torus(X, L, only_upper_tri=False):
    """Compute the pairwise difference matrix (ie. x_i - x_j) in a n-D torus (ie. [-width, width]^d).

    Args:
        X (torch.Tensor of shape (batch_size, n_particles, dim_phys)): Tensor of particles
        L (float): Length of the box
        only_upper_tri (bool): Whether to return only the upper triangular part of the pairwise difference matrix
            (default is False)

    Returns:
        if only_upper_tri
            ret (torch.Tensor of shape (batch_size, n_particles * (n_particles-1) / 2, dim_phys)) which verifies that
                ret[batch_id,k] = x_i - x_j for some i,j where i > j
            Note that the indices i and j can be recovered using torch.triu_indices(X.shape[1], X.shape[1], offset=1, device=X.device)
        else:
            ret (torch.Tensor of shape (batch_size, n_particles, n_particles, dim_phys)) which verifies that
                ret[batch_id,i,j] = x_i - x_j
    """

    G = compute_pairwise_distances_same(X, only_upper_tri=only_upper_tri)
    ret = move_to_torus(G, L)
    return ret


gram = compute_pairwise_distances_same


def gram_species_torus(a, X, L, only_upper_tri=False):
    """Compute the pairwise difference matrix (ie. x_i - x_j) in a n-D torus (ie. [-width, width]^d)  and accounting for the particles species.

    Args:
        a (torch.Tensor of shape (batch_size, n_particles)): Species of each particle
        X (torch.Tensor of shape (batch_size, n_particles, dim_phys)): Tensor of particles
        only_upper_tri (bool): Whether to return only the upper triangular part of the pairwise difference matrix
            (default is False)

    Returns:
        if only_upper_tri
            ret (torch.Tensor of shape (batch_size, n_particles * (n_particles-1) / 2, dim_phys)) which verifies that
                ret[batch_id,k] = x_i - x_j for some i,j where i > j
            ret_species (torch.Tensor of shape (batch_size, n_particles * (n_particles-1) / 2, 2)) which verifies that
                ret_species[batch_id,k] = (a[i], a[j]) for some i,j where i > j
            Note that the indices i and j can be recovered using torch.triu_indices(X.shape[1], X.shape[1], offset=1, device=X.device)
        else
            ret (torch.Tensor of shape (batch_size, n_particles, n_particles, dim_phys)) which verifies that
                ret[batch_id,i,j] = x_i - x_j
            ret_species (torch.Tensor of shape (batch_size, n_particles, n_particles, 2)) which verifies that
                ret_species[batch_id,i,j] = (a[i], a[j])
    """

    G, species = compute_pairwise_distances_species_same(a, X, only_upper_tri=only_upper_tri)
    ret = move_to_torus(G, L)
    return ret, species


gram_species = compute_pairwise_distances_species_same


def ot_permutation(X0, X1, L=None):
    """Use OT to get a permutation between two steps of particles

    Inputs:
        - X0 (torch.Tensor of shape (batch_size, n_particles, dim_phys)): samples from rho0
        - X1 (torch.Tensor of shape (batch_size, n_particles, dim_phys)): samples from rho1
        - L (float): size of the box (use None to disable boxed metric)

    Outputs:
        - X0, X1 (tuple of torch.Tensor of shape (batch_size, n_particles, dim_phys)): with X0 permuted
        to minimize the EMD distance against X1 (for each batch of X0 and X1)
    """
    # Compute the weights
    n_particles = X0.shape[-2]
    weights = torch.ones((n_particles,)) / n_particles
    # Compute the cost matrix
    G = compute_pairwise_distances(X0, X1)
    if L is not None:
        G = move_to_torus(G, L)
    M = torch.linalg.norm(G, dim=-1)
    # Create the pairings
    for batch_id in range(X0.shape[0]):
        # Compute the OT matrix
        G = ot.emd(weights, weights, M[batch_id])
        G /= G.max()
        G = G.bool()
        # Compute the pairing
        pairing = torch.arange(n_particles).unsqueeze(
            0).expand((n_particles, n_particles))
        pairing = pairing[G]
        X0[batch_id] = X0[batch_id][pairing]
    return X0, X1


def ot_permutation_batch_level(X0, X1, L=None):
    """Use OT to get a permutation between two steps of particles at batch level

    Inputs:
        - X0 (torch.Tensor of shape (batch_size, n_particles, dim_phys)): samples from rho0
        - X1 (torch.Tensor of shape (batch_size, n_particles, dim_phys)): samples from rho1
        - L (float): size of the box (use None to disable boxed metric)

    Outputs:
        - X0, X1 (tuple of torch.Tensor of shape (batch_size, n_particles, dim_phys)): with X0 permuted
        to minimize the EMD distance against X1 (for each batch of X0 and X1)
    """
    # Get the parameters
    batch_size = X0.shape[0]
    n_particles = X0.shape[-2]
    dim_phys = X0.shape[-1]
    data_shape = (batch_size, n_particles, dim_phys)
    # Reshape the data
    X0 = X0.view((1, -1, dim_phys))
    X1 = X1.view((1, -1, dim_phys))
    # Compute the weights
    num_samples = X0.shape[1]
    weights = torch.ones((num_samples,), device=X0.device) / num_samples
    # Compute the cost matrix
    G = compute_pairwise_distances(X0, X1)
    if L is not None:
        G = move_to_torus(G, L)
    M = torch.linalg.norm(G, dim=-1)
    # Compute the OT matrix
    G = ot.emd(weights, weights, M)
    G /= G.max()
    G = G.bool()
    # Compute the pairing
    pairing = torch.arange(num_samples).unsqueeze(
        0).expand((num_samples, num_samples))
    pairing = pairing[G]
    # Reshape the pairing
    return X0.squeeze(0)[pairing].view(data_shape), X1.view(data_shape)


def linear_assignment_permutation(X0, X1, L=None):
    """Use the hungarian algorithm to get a permutation between two steps of particles at batch level

    Inputs:
        - X0 (torch.Tensor of shape (batch_size, n_particles, dim_phys)): samples from rho0
        - X1 (torch.Tensor of shape (batch_size, n_particles, dim_phys)): samples from rho1
        - L (float): size of the box (use None to disable boxed metric)

    Outputs:
        - X0, X1 (tuple of torch.Tensor of shape (batch_size, n_particles, dim_phys)): with X0 permuted
        to minimize the EMD distance against X1 (for each batch of X0 and X1)
    """

    if not linear_assignment_exists:
        return ot_permutation(X0, X1, L=L)
    # Compute the cost matrix
    G = compute_pairwise_distances(X0, X1)
    if L is not None:
        G = move_to_torus(G, L)
    M = torch.sum(torch.square(G), dim=-1)
    # Compute the linear assignment
    idxs = batch_linear_assignment(M).unsqueeze(-1).expand((-1, -1, X0.shape[-1]))
    return torch.gather(X0, 1, idxs), X1


def superpose_points(points1, points2):
    """Apply the optimal rotation/translation to points1 to match points2

    Args:
        points1 (torch.Tensor of shape (batch_size, n_particles, dim_phys)): Points 1
        points2 (torch.Tensor of shape (batch_size, n_particles, dim_phys)): Points 2

    Returns:
        new_points1 (torch.Tensor of shape (batch_size, n_particles, dim_phys)): Modified points 1
    """

    # Compute optimal rotation and translation
    M = torch.matmul(points1.transpose(-2, -1), points2)
    U, S, V = torch.svd(M)
    R = torch.matmul(U, V.transpose(-2, -1))
    # Apply rotation and translation to superpose points1 onto points2
    return torch.matmul(points1, R), points2


def box_particles(x_prop, box_width):
    return torch.remainder(x_prop, box_width)


def make_mask(n, dim_phys, return_only_idx_mask=False):
    """Make a mask to help unravel triangular matrices

    See reshape_tri for direct application (and the topk velocity for return_only_idx_mask).

    Args:
        n (int): Number of particles
        dim_phys (int): Dimension of the particles
        return_only_idx_mask (bool): Whether to return the destination indexes of the tri-shaped tensor

    Returns:
        if return_only_idx_mask:
            indexes (torch.Tensor of shape (n * (n-1))): Contains the index of particles (i,j) in the tri-shaped tensor
            sign_mask (torch.Tensor of shape (n, n)): Binary mask explaining where to put a minus sign after unraveling
                the tri-shaped tensor
        otherwise
            mask (torch.Tensor of shape (n, 2, tri_size, dim_phys)): Binary mask explaining where to put the values during
                unraveling
    """

    # Compute the upper indexes
    row_indices, col_indices = torch.triu_indices(n, n, offset=1)
    # Make the indexes for the upper part
    indexes = torch.zeros((n, n), dtype=int)
    indexes[row_indices, col_indices] = torch.arange(row_indices.shape[0])
    indexes = indexes + indexes.transpose(0, 1)
    indexes = indexes[~torch.eye(n, dtype=bool)].view((n, n - 1))  # remove the diagonal
    # Make a mask for a negative sign
    sign_mask = torch.ones((n, n), dtype=bool)
    sign_mask[row_indices, col_indices] = False
    sign_mask = sign_mask[~torch.eye(n, dtype=bool)].view((n, n - 1))  # remove the diagonal
    if return_only_idx_mask:
        return indexes, sign_mask
    else:
        # Make the mask
        tri_size = int(n * (n - 1) / 2)
        mask = torch.zeros((n, 2, tri_size, dim_phys), dtype=bool)
        for i in range(n):
            mask[i, 0][indexes[i][sign_mask[i]]] = True
            mask[i, 1][indexes[i][~sign_mask[i]]] = True
        return mask


def compute_combinations(batch_size, n_particles, dim_phys, t, X, k=None):
    """Compute the upper-tri difference matrix of X and adapt t accordingly

    Args:
        batch_size (int): Size of the batch
        n_particles (int): Number of particles
        dim_phys (int): Dimension of particles
        t (torch.Tensor of shape (batch_size,1,1) or float): Times
        X (torch.Tensor of shape (batch_size, *data_shape)): Particles
        k (int or None): Reshape the time to n_particles * k

    Returns:
        xs_tri (torch.Tensor of shape (batch_size * n_particles * (n_particles - 1) / 2, dim_phys)) : the upper-tri difference matrix
        if k is not None
            ts_tri (torch.Tensor of shape (batch_size * n_particles * k), 1): the reshaped time
        otherwise
            ts_tri (torch.Tensor of shape (batch_size * n_particles * (n_particles - 1) / 2), 1): the reshaped time
        tri_size (float) : n_particles * (n_particles - 1) / 2
    """

    # Make the gram matrix
    xs_tri = gram(X, only_upper_tri=True)
    tri_size = xs_tri.shape[1]
    # Make t a tensor when its scalar
    if len(t.shape) == 0:
        t = t.repeat((X.shape[0]))
    if k is not None:
        ts_tri = t.flatten().view((-1, 1, 1)).repeat((1, n_particles * k, 1)).view((-1, 1))
    else:
        ts_tri = t.flatten().view((-1, 1, 1)).repeat((1, tri_size, 1)).view((-1, 1))
        xs_tri = xs_tri.view((-1, dim_phys))
    return xs_tri, ts_tri, tri_size


def reshape_tri(a, b, batch_size, n_particles, mask, expand=False):
    """Unravels tri-shaped tensors to full tensors

    Args:
        a (torch.Tensor of shape (batch_size, tri_size, *data_shape)): Tri-shaped tensor
        b (torch.Tensor of shape (batch_size, tri_size, *data_shape)): Tri-shaped tensor
        batch_size (int): Size of the batch
        n_particles (int): Number of particles
        mask (torch.Tensor): Output of make_mask(n_particles, dim_phys)
        expand (bool): Whether to use expand when possible. This must be used when a == b.
            (Default is false)

    Returns:
        ret (torch.Tensor of shape (batch_size, n_particles, n_particles-1, *data_shape)): Output unraveled tensor
    """

    batch_size = a.shape[0]
    data_shape = a.shape[2:]
    if expand:
        ret = a.unsqueeze(1).expand((a.shape[0], 2, *a.shape[1:]))
    else:
        ret = torch.stack([a, b], dim=1)
    ret = ret.unsqueeze(1).expand((batch_size, *mask.shape[:-1], *data_shape))
    ret = ret[:, mask[..., :a.shape[-1]]].view((batch_size, n_particles, n_particles - 1, *data_shape))
    return ret


def clip_function(g, x, clip_value):
    """Clip the value of the function g"""
    return torch.clamp(g(x.clone()), -clip_value, clip_value)


def clip_function_and_log_jacobian(g_and_jacobian, x, clip_value):
    """Clip the value of the function g and return its jacobian"""
    g_x, jac_g_x = g_and_jacobian(x.clone())
    mask = (g_x <= -clip_value) | (g_x >= clip_value)
    jac_g_x[mask] = 0.
    g_x = torch.clamp(g_x, -clip_value, clip_value)
    return g_x, jac_g_x


def separate_by_species(a, n_species, dim_phys):
    batch_size = a.shape[0]
    return [
        torch.nonzero(a == i)[:, 1].view((batch_size, -1, 1)).expand((-1, -1, dim_phys)) for i in range(n_species)
    ]


def rebuild_from_index(i, indexes, X):
    return torch.gather(X, 1, indexes[i])
