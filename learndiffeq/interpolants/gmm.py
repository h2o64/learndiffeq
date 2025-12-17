# Play with GMM in the special case of a one-sided interpolant
# This is based on the content of appendix A

# Libraries
import torch
import math


class IntermediateGaussians:
    """Implement equations Eq. (A.3) in the special case of the one-sided interpolant"""

    def __init__(self, weights, means, covs, interpolant):
        """Constructor

        Args:
            weights (torch.Tensor of shape (n_modes,)): weights for each mode of the target
            means (torch.Tensor of shape (n_modes, dim)): means for each mode of the target
            covs (torch.Tensor of shape (n_modes, dim, dim)): covariance matrices for each mode of the target
            interpolant (learndiffeq.interpolants.OneSidedInterpolant): one-sided interpolant
        """

        self.N = weights.shape[0]
        self.weights = weights
        self.means = means
        self.covs = covs
        self.int = interpolant
        self.dim = self.means.shape[-1]
        self.dim_log_2_pi = self.dim * math.log(2. * torch.pi)

    def m(self, t):
        """Implement m_{ij}(t) from Eq. (A.3)

        Args:
            t (torch.Tensor of shape (batch_size, 1)): Times

        Returns:
            m (torch.Tensor of shape (batch_size, n_modes, dim)): Means
        """

        return self.int.beta(t).unsqueeze(-1) * self.means

    def m_dot(self, t):
        """Implement \\dot{m}_{ij}(t) from Eq. (A.3)

        Args:
            t (torch.Tensor of shape (batch_size, 1)): Times

        Returns:
            m (torch.Tensor of shape (batch_size, n_modes, dim)): Means
        """

        return self.int.beta_dot(t).unsqueeze(-1) * self.means

    def c(self, t):
        """Implement C_{ij}(t) from Eq. (A.3)

        Args:
            t (torch.Tensor of shape (batch_size, 1)): Times

        Returns:
            covs (torch.Tensor of shape (batch_size, n_modes, dim, dims)): Covariance matrices
        """

        ret = torch.square(self.int.alpha(t)).unsqueeze(-1) * torch.eye(self.dim, device=t.device)
        ret = ret.unsqueeze(1).repeat((1, self.N, 1, 1))
        ret += torch.square(self.int.beta(t)).unsqueeze(-1).unsqueeze(-1) * self.covs
        return ret

    def c_dot(self, t):
        """Implement \\dot{C}_{ij}(t) from Eq. (A.3)

        Args:
            t (torch.Tensor of shape (batch_size, 1)): Times

        Returns:
            covs (torch.Tensor of shape (batch_size, n_modes, dim, dims)): Covariance matrices
        """

        ret = self.int.alpha_squared_dot(t).unsqueeze(-1) * torch.eye(self.dim, device=t.device)
        ret = ret.unsqueeze(1).repeat((1, self.N, 1, 1))
        ret += self.int.beta_squared_dot(t).unsqueeze(-1).unsqueeze(-1) * self.covs
        return 2. * ret

    def log_prob(self, t, x):
        """Compute intermediate Gaussian log-likelihoods in Eq. (A.4)

        Args:
            t (torch.Tensor of shape (batch_size, 1)): Times
            x (torch.Tensor of shape (batch_size, dim)): States

        Returns:
            log_prob (torch.Tensor of shape (batch_size, n_modes)): Log-likelihood for each mode
        """

        # Compute mean and cov
        ms = self.m(t)
        covs = self.c(t)
        # Compute the log_prob
        diff = x.unsqueeze(1) - ms.unsqueeze(0)
        ret = -torch.matmul(diff.unsqueeze(-2), torch.linalg.solve(covs.unsqueeze(0), diff.unsqueeze(-1)))
        ret = ret.squeeze(-1).squeeze(-1)
        ret -= (self.dim_log_2_pi + torch.logdet(covs))
        return 0.5 * ret.squeeze(0)

    def prob(self, t, x):
        """Compute \rho(t,x) from Eq. (A.4) (without summing)

        Args:
            t (torch.Tensor of shape (batch_size, 1)): Times
            x (torch.Tensor of shape (batch_size, dim)): States

        Returns:
            prob (torch.Tensor of shape (batch_size, n_modes)): Weighted likelihood for each mode
        """
        return self.weights * torch.exp(self.log_prob(t, x))


class OptimalVelocityField(torch.nn.Module):
    """Optimal velocity field described in Eq. (A.5)"""

    def __init__(self, intermediate_gaussians):
        """Constructor

        Args:
            intermediate_gaussians (learndiffeq.interpolants.gmm.IntermediateGaussians): Intermediate object
        """

        super().__init__()
        self.int_gauss = intermediate_gaussians

    def forward(self, t, x):
        """Evaluate the velocity field

        Args:
            t (torch.Tensor of shape (batch_size, 1)): Times
            x (torch.Tensor of shape (batch_size, dim)): States

        Returns:
            b (torch.Tensor of shape (batch_size, dim)): Evaluation of b(t,x)
        """

        # Reshape t
        if len(t.shape) == 0:
            t = t.expand((x.shape[0], 1))
        # Get the important quantities
        c = self.int_gauss.c(t)
        c_dot = self.int_gauss.c_dot(t)
        m = self.int_gauss.m(t)
        m_dot = self.int_gauss.m_dot(t)
        prob = self.int_gauss.prob(t, x)
        # Build the velocity field
        x_ = x.unsqueeze(1).expand((-1, self.int_gauss.N, -1))
        factor = m_dot
        factor += 0.5 * torch.matmul(c_dot, torch.linalg.solve(c, (x_ - m).unsqueeze(-1))).squeeze(-1)
        # Compute the velocity field
        ret = (factor * prob.unsqueeze(-1)).sum(dim=1)
        ret /= prob.sum(dim=-1, keepdim=True)
        # Return the velocity field
        return ret


class OptimalScore(torch.nn.Module):
    """Optimal velocity field described in Eq. (A.6)"""

    def __init__(self, intermediate_gaussians):
        """Constructor

        Args:
            intermediate_gaussians (learndiffeq.interpolants.gmm.IntermediateGaussians): Intermediate object
        """

        super().__init__()
        self.int_gauss = intermediate_gaussians

    def forward(self, t, x):
        """Evaluate the score field

        Args:
            t (torch.Tensor of shape (batch_size, 1)): Times
            x (torch.Tensor of shape (batch_size, dim)): States

        Returns:
            s (torch.Tensor of shape (batch_size, dim)): Evaluation of s(t,x)
        """

        # Reshape t
        if len(t.shape) == 0:
            t = t.expand((x.shape[0], 1))
        # Get the important quantities
        c = self.int_gauss.c(t)
        m = self.int_gauss.m(t)
        prob = self.int_gauss.prob(t, x)
        # Build the velocity field
        x_ = x.unsqueeze(1).expand((-1, self.int_gauss.N, -1))
        factor = torch.linalg.solve(c, (x_ - m).unsqueeze(-1)).squeeze(-1)
        # Compute the velocity field
        ret = (factor * prob.unsqueeze(-1)).sum(dim=1)
        ret /= prob.sum(dim=-1, keepdim=True)
        # Return the velocity field
        return -ret


class VelocityFromScore(torch.nn.Module):
    """Obtain velocity field from the score using b(t,x) = coef1(t) * x + coef2(t) * s(t,x)"""

    def __init__(self, score, mean_rho1):
        """Constructor

        Args:
            score (function): Function taking t and x as arguments (see bellow for the shape of t and x)
            mean_rho1 (torch.Tensor of shape (dim,)) : Mean of the target mixture
        """

        super().__init__()
        self.s = score
        self.int = score.int_gauss.int
        self.mean_rho1 = mean_rho1

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
        coef2 = coef1 * torch.square(self.int.alpha(t)) - self.int.alpha_squared_dot(t)
        # Return both coefficients
        return coef1, coef2

    def forward(self, t, x):
        """Evaluate the velocity field

        Args:
            t (torch.Tensor of shape (batch_size, 1)): Times
            x (torch.Tensor of shape (batch_size, dim)): States

        Returns:
            b (torch.Tensor of shape (batch_size, dim)): Evaluation of b(t,x)
        """

        # Reshape t
        if len(t.shape) == 0:
            t = t.expand((x.shape[0], 1))
        # Get the coefficients
        coef1, coef2 = self.compute_coefs(t)
        # Compute the velocity field
        zero_t = torch.zeros_like(t)
        one_t = torch.ones_like(t)
        val = coef1 * x + coef2 * self.s(t, x)
        val = torch.where(t == 0.0, self.int.alpha_dot(zero_t) * x + self.int.beta_dot(zero_t)
                          * self.mean_rho1.unsqueeze(0).expand((x.shape[0], -1)), val)
        val = torch.where(t == 1.0, self.int.beta_dot(one_t) * x, val)
        return val
