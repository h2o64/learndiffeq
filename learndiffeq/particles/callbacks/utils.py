# Utilify functions for callbacks
import torch
from scipy.special import gamma
from ..utils import gram, gram_torus


def _volume_sphere(radius, dim):
    """Volume of a sphere"""

    c = torch.pi**(dim / 2) / gamma(1 + dim / 2)  # Volume of unit sphere.
    return c * radius**dim


def radial_distribution_function(coordinates_, L=None, num_bins=512, max_box_size=10.0):
    """Estimate the radial distribution

    Stolen from https://github.com/google-deepmind/flows_for_atomic_solids/blob/main/utils/observable_utils.py#L140

    Args:
        coordinates (torch.Tensor of shape (batch_size, n_particles, dim_phys)): Coordinates of the particles
        num_bins (int): Number of bins in histograms computations (default is 512)
        L (float): Length of the box (default is None)
        max_box_size (float): Threshold for the distances (default is 10.0)

    Returns
        gr (torch.Tensor of shape (num_bins, 2)): Estimated histogram (r is at gr[:,0] and the corresponding density is at gr[:,1])
    """

    # Parse the shape
    batch_size, num_particles, dim = coordinates_.shape
    if L is None:
        # Restrict the box
        coordinates = coordinates_ - coordinates_.mean(dim=1, keepdim=True)
        box_mask = ((coordinates < -max_box_size) | (coordinates > max_box_size)
                    | torch.isnan(coordinates)).sum((1, 2)) > 0
        coordinates = coordinates[~box_mask]
        # Compute the pairwise distances
        dr = gram(coordinates, only_upper_tri=True).cpu()
        # Center the particles
        min_positions = coordinates.view((-1, dim)).min(dim=0).values
        max_positions = coordinates.view((-1, dim)).max(dim=0).values
        center = (max_positions + min_positions) / 2.
        coordinates = coordinates - center
        # Recompute the box size
        min_positions = coordinates.view((-1, dim)).min(dim=0).values.cpu()
        max_positions = coordinates.view((-1, dim)).max(dim=0).values.cpu()
        box_length = max_positions - min_positions
        min_box_length = torch.min(box_length)
        # Compute the volume of the box
        box_volume = torch.prod(box_length)
        # Compute the histogram
        hist, bins = torch.histogram(dr, bins=num_bins, range=(0., min_box_length / 2.))
    else:
        # Compute the pairwise distances
        dr = gram_torus(coordinates_, L, only_upper_tri=True).cpu()
        # Compute the volume of the box
        box_volume = L**dim
        # Compute the histogram
        hist, bins = torch.histogram(dr, bins=num_bins, range=(0., L))
    # Normaliser for histogram so that `g(r)` approaches unity for large `r`.
    density = num_particles / box_volume
    volume_shell = _volume_sphere(bins[1:], dim) - _volume_sphere(bins[:-1], dim)
    normaliser = volume_shell * density * batch_size * (num_particles - 1) / 2
    gr = torch.column_stack(((bins[:-1] + bins[1:]) / 2, hist / normaliser))
    return gr


def compute_intersection(h1, h2):
    """Intersection between two histograms

    Args:
            h1 (torch.Tensor of shape (bins,)): First histogram
            h2 (torch.Tensor of shape (bins,)): Second histogram

    Returns:
            inter (float): Metric
    """

    return torch.sum(torch.minimum(h1, h2))


def compute_correlation(h1, h2):
    """Correlation between two histograms

    Args:
            h1 (torch.Tensor of shape (bins,)): First histogram
            h2 (torch.Tensor of shape (bins,)): Second histogram

    Returns:
            corr (float): Metric
    """

    h1_norm = h1 - h1.mean()
    h2_norm = h2 - h2.mean()
    return torch.sum(h1_norm * h2_norm) / torch.sqrt(torch.sum(torch.square(h1_norm))
                                                     * torch.sum(torch.square(h2_norm)))


def compute_chi_squared(h1, h2):
    """Chi-squared distance between two histograms

    Args:
            h1 (torch.Tensor of shape (bins,)): First histogram
            h2 (torch.Tensor of shape (bins,)): Second histogram

    Returns:
            chi_squared (float): Metric

    """

    mask = h1 > 0
    return torch.sum(torch.square(h1[mask] - h2[mask]) / h1[mask])
