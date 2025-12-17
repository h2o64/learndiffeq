# Soft sphere (BHHP) potential

# Libraries
import torch
from .kob_andersen import KobAndersen
from ..utils import make_mask


class SoftSphere(KobAndersen):

    def __init__(self, n_particles, dim_phys, L, sigma_matrix=None, epsilon_matrix=None, rcut_matrix=None):
        """Constructor

        Args:
            n_particles (int): Number of particles
            dim_phys (int): Dimension of the particles
            L (float): Length of the box
            sigma_matrix (None or torch.Tensor): Value of the sigma matrix
            epsilon_matrix (None or torch.Tensor): Value of the epsilon matrix
            rcut_matrix (None or torch.Tensor): Value of the cutting radiuses matrix
        """

        # Call the parent constructor
        super(KobAndersen, self).__init__()
        # Store the parameters
        if sigma_matrix is None:
            sigma_matrix = torch.FloatTensor([
                [1.0, 1.2],
                [1.2, 1.4]
            ])
        self.register_buffer('sigma_matrix', sigma_matrix, persistent=False)
        if epsilon_matrix is None:
            epsilon_matrix = torch.ones((2, 2))
        self.register_buffer('epsilon_matrix', epsilon_matrix, persistent=False)
        if rcut_matrix is None:
            rcut_matrix = 2.5 * sigma_matrix
        self.register_buffer('rcut_sq_matrix', torch.square(rcut_matrix), persistent=False)
        self.L = L
        # Make the masks
        self.mask = make_mask(n_particles, dim_phys)
        self.mask_local = torch.zeros((n_particles, n_particles, dim_phys, dim_phys), dtype=bool)
        arr = torch.arange(n_particles)
        self.mask_local[arr, arr] = True
        row_indices, col_indices = torch.triu_indices(n_particles, n_particles, offset=1)
        self.register_buffer('row_indices', row_indices, persistent=False)
        self.register_buffer('col_indices', col_indices, persistent=False)

    def Wpp(self, r_sq, r_cut_sq, epsilon, sigma):
        """BHHP potential at the particle level

        Args:
            r_sq (torch.Tensor): Squared distance between two particles
            r_cut_sq (torch.Tensor): Value of the cutting distance
                epsilon (torch.Tensor): Value of epsilon
                sigma (torch.Tensor): Value of sigma

        Returns:
            energies (torch.Tensor with the same shape as r_sq) : Energy associated to each pair of particles
        """

        idr2_cut = sigma * sigma / r_cut_sq
        idr2 = sigma * sigma / r_sq
        idr12_cut = torch.pow(idr2_cut, 6)
        idr12 = torch.pow(idr2, 6)
        return torch.where(r_sq < r_cut_sq, epsilon * (idr12 - idr12_cut), torch.zeros_like(r_sq))

    def grad_Wpp(self, r_sq, r_cut_sq, epsilon, sigma):
        """First derivative of BHHP potential at the particle level

        Args:
            r_sq (torch.Tensor): Squared distance between two particles
            r_cut_sq (torch.Tensor): Value of the cutting distance
                epsilon (torch.Tensor): Value of epsilon
                sigma (torch.Tensor): Value of sigma

        Returns:
            grad_energies (torch.Tensor with the same shape as r_sq) : First derivative of the energy associated
                to each pair of particles
        """

        sigma_sq = sigma * sigma
        idr2 = sigma_sq / r_sq
        idr12 = torch.pow(idr2, 6)
        return torch.where(
            r_sq < r_cut_sq,
            -epsilon * 6. * idr12 / r_sq,
            torch.zeros_like(r_sq)
        )

    def grad_grad_Wpp(self, r_sq, r_cut_sq, epsilon, sigma):
        """Second derivative of the BHHP potential at the particle level

        Args:
            r_sq (torch.Tensor): Squared distance between two particles
            r_cut_sq (torch.Tensor): Value of the cutting distance
                epsilon (torch.Tensor): Value of epsilon
                sigma (torch.Tensor): Value of sigma

        Returns:
    grad_grad_energies (torch.Tensor with the same shape as r_sq) : Second derivative of the energy associated
        to each pair of particles
        """

        sigma_sq = sigma * sigma
        idr2 = sigma_sq / r_sq
        idr12 = torch.pow(idr2, 6)
        return torch.where(
            r_sq < r_cut_sq,
            epsilon * 42. * idr12 / torch.square(r_sq),
            torch.zeros_like(r_sq)
        )
