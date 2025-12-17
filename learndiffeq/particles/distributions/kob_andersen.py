# Kob-Andersen potential

# Libraries
import torch
from ..utils import gram_torus, make_mask, reshape_tri


class KobAndersen(torch.nn.Module):

    def __init__(self, n_particles, dim_phys, L, sigma_matrix=None, epsilon_matrix=None, rcut_matrix=None,
                 C0=0.04049023795, C2=-0.00970155098, C4=0.00062012616):
        """Constructor

        Args:
            n_particles (int): Number of particles
            dim_phys (int): Dimension of the particles
            L (float): Length of the box
            sigma_matrix (None or torch.Tensor): Value of the sigma matrix
            epsilon_matrix (None or torch.Tensor): Value of the epsilon matrix
            rcut_matrix (None or torch.Tensor): Value of the cutting radiuses matrix
            C0 (float): Value of C0
            C2 (float): Value of C2
            C4 (float): Value of C4
        """

        # Call the parent constructor
        super(KobAndersen, self).__init__()
        # Store the parameters
        if sigma_matrix is None:
            sigma_matrix = torch.FloatTensor([
                [1.0, 0.8, 0.9],
                [0.8, 0.88, 0.8],
                [0.9, 0.8, 0.94]
            ])
        self.register_buffer('sigma_matrix', sigma_matrix, persistent=False)
        if epsilon_matrix is None:
            epsilon_matrix = torch.FloatTensor([
                [1.0, 1.5, 0.75],
                [1.5, 0.5, 1.5],
                [0.75, 1.5, 0.75]
            ])
        self.register_buffer('epsilon_matrix', epsilon_matrix, persistent=False)
        if rcut_matrix is None:
            rcut_matrix = 2.5 * sigma_matrix
        self.register_buffer('rcut_sq_matrix', torch.square(rcut_matrix), persistent=False)
        self.C0 = C0
        self.C2 = C2
        self.C4 = C4
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
        """Kob-Andersen potential at the particle level

        Args:
            r_sq (torch.Tensor): Squared distance between two particles
            r_cut_sq (torch.Tensor): Value of the cutting distance
                epsilon (torch.Tensor): Value of epsilon
                sigma (torch.Tensor): Value of sigma

        Returns:
            energies (torch.Tensor with the same shape as r_sq) : Energy associated to each pair of particles
        """

        idr2 = sigma * sigma / r_sq
        idr4 = idr2 * idr2
        idr6 = idr4 * idr2
        idr12 = idr6 * idr6
        return torch.where(
            r_sq < r_cut_sq,
            4. * epsilon * (idr12 - idr6 + self.C0 + (self.C2 / idr2) + (self.C4 / idr4)),
            torch.zeros_like(r_sq)
        )

    def grad_Wpp(self, r_sq, r_cut_sq, epsilon, sigma):
        """First derivative of the Kob-Andersen potential at the particle level

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
        idr4 = idr2 * idr2
        idr6 = idr4 * idr2
        idr12 = idr6 * idr6
        return torch.where(
            r_sq < r_cut_sq,
            -4. * epsilon * (6. * idr12 - 3. * idr6 + self.C2 * idr2 + 2. * self.C4 * idr4) / r_sq,
            torch.zeros_like(r_sq)
        )

    def grad_grad_Wpp(self, r_sq, r_cut_sq, epsilon, sigma):
        """Second derivative of the Kob-Andersen potential at the particle level

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
        r_cu = torch.square(r_sq)
        idr2 = sigma_sq / r_sq
        idr4 = idr2 * idr2
        idr6 = idr4 * idr2
        idr12 = idr6 * idr6
        return torch.where(
            r_sq < r_cut_sq,
            8. * epsilon * (21. * idr12 - 6. * idr6 + self.C2 * idr2 + self.C4 * idr4) / r_cu,
            torch.zeros_like(r_sq)
        )

    def get_parameters(self, a):
        """Compute r_cut, epsilon and sigma from a"""
        r_cut_sq = self.rcut_sq_matrix[a[:, self.row_indices], a[:, self.col_indices]]
        epsilon = self.epsilon_matrix[a[:, self.row_indices], a[:, self.col_indices]]
        sigma = self.sigma_matrix[a[:, self.row_indices], a[:, self.col_indices]]
        return r_cut_sq, epsilon, sigma

    def U(self, a, x):
        """Kob-Andersen potential at the system level

        The computations are optimized for the tri-diagonal pairwise structure of the potential

        Args:
            a (torch.Tensor of shape (batch_size, n_particles)): Species
            x (torch.Tensor of shape (batch_size, n_particles, dim_phys)): Particles

        Returns:
            U (torch.Tensor of shape (batch_size,)) : Potential of the system
        """

        # Compute the gram matrix
        y = gram_torus(x, L=self.L, only_upper_tri=True)
        # Compute the squared norm
        y_norms_sq = torch.sum(torch.square(y), dim=-1)
        # Compute the potential
        U_pp = self.Wpp(y_norms_sq, *self.get_parameters(a))
        # return U_pp.sum(1) / x.shape[1] # WARNING: This is the potential per particle, not the total potential!
        return U_pp.sum(1) # This is the total potential!

    def grad_U(self, a, x, return_U=False):
        """Gradient of the Kob-Andersen potential at the system level

        The computations are optimized for the tri-diagonal pairwise structure of the potential

        Args:
            a (torch.Tensor of shape (batch_size, n_particles)): Species
            x (torch.Tensor of shape (batch_size, n_particles, dim_phys)): Particles
            return_U (bool): Whether to compute U(x) at the same time (default is False)

        Returns:
            if return_U
                U (torch.Tensor of shape (batch_size,)) : Potential of the system
                grad_U (torch.Tensor of shape (batch_size, n_particles, dim_phys)) : Gradient of the potential of the system
            otherwise
                grad_U (torch.Tensor of shape (batch_size, n_particles, dim_phys)) : Gradient of the potential of the system
        """

        # Parse the shape
        batch_size, n_particles, dim_phys = x.shape
        # Compute the gram matrix
        y = gram_torus(x, L=self.L, only_upper_tri=True)
        # Compute the squared norm
        y_norms_sq = torch.sum(torch.square(y), dim=-1, keepdim=True)
        # Compute the gradient of the potential
        a_params = self.get_parameters(a)
        grad_U_pp = self.grad_Wpp(y_norms_sq.squeeze(-1), *a_params).unsqueeze(-1)
        # Reshape things
        y_full = reshape_tri(y, -y, batch_size, n_particles, self.mask)
        grad_U_pp_full = reshape_tri(grad_U_pp, grad_U_pp, batch_size, n_particles, self.mask, expand=True)
        # Compute the gradient
        grad_U = -2. * (y_full * grad_U_pp_full).sum(dim=2) / n_particles
        # Return everything
        if return_U:
            U = self.Wpp(y_norms_sq.squeeze(-1), *a_params).sum(1) / n_particles
            return U, grad_U
        else:
            return grad_U

    def hessian_U(self, a, x, return_grad_U=False):
        """Hessian of the Kob-Andersen potential at the system level

        The computations are optimized for the tri-diagonal pairwise structure of the potential

        Args:
            a (torch.Tensor of shape (batch_size, n_particles)): Species
            x (torch.Tensor of shape (batch_size, n_particles, dim_phys)): Particles
            return_grad_U (bool): Whether to compute \nabla U(x) at the same time (default is False)
            kwargs (dict): Arguments to pass to Wpp

        Returns:
            if return_U
                grad_U (torch.Tensor of shape (batch_size, n_particles, dim_phys)) : Gradient of the potential of the system
                hessian_U (torch.Tensor of shape (batch_size, n_particles, dim_phys, n_particles, dim_phys)) : Gradient of the potential of the system
            otherwise
                hessian_U (torch.Tensor of shape (batch_size, n_particles, dim_phys, n_particles, dim_phys)) : Gradient of the potential of the system
        """

        # Parse the shape
        batch_size, n_particles, dim_phys = x.shape
        # Compute the gram matrix
        y = gram_torus(x, L=self.L, only_upper_tri=True)
        # Compute the squared norm
        y_norms_sq = torch.sum(torch.square(y), dim=-1, keepdim=True)
        # Compute grad_Wpp and grad_grad_Wpp
        a_params = self.get_parameters(a)
        grad_Wpp = self.grad_Wpp(y_norms_sq.squeeze(-1), *a_params).unsqueeze(-1)
        grad_grad_Wpp = self.grad_grad_Wpp(y_norms_sq.squeeze(-1), *a_params).unsqueeze(-1)
        # Compute the matrix A
        A = 2 * grad_grad_Wpp.unsqueeze(-1) * torch.matmul(y.unsqueeze(-1), y.unsqueeze(-2))
        A += grad_Wpp.unsqueeze(-1) * torch.eye(dim_phys, device=x.device)
        A = reshape_tri(A, A, batch_size, n_particles, self.mask, expand=True)
        # Compute the jacobian
        jac = torch.empty((batch_size, n_particles, n_particles, dim_phys, dim_phys), device=x.device)
        mask_local_ = self.mask_local.unsqueeze(0).expand((batch_size, -1, -1, -1, -1))
        jac[mask_local_] = A.sum(dim=2).flatten()
        jac[~mask_local_] = -1 * A.flatten()
        jac /= n_particles
        # Return the jacobian
        if return_grad_U:
            y_full = reshape_tri(y, -y, batch_size, n_particles, self.mask)
            grad_U_pp_full = reshape_tri(grad_Wpp, grad_Wpp, batch_size, n_particles, self.mask, expand=True)
            grad_U = -2. * (y_full * grad_U_pp_full).sum(dim=2)
            grad_U /= n_particles
            return grad_U, 2. * jac.transpose(2, 3)
        else:
            return 2. * jac.transpose(2, 3)
