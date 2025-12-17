# Uniform distribution using the species

# Libraries
import torch


class UniformParticles(torch.nn.Module):

    def __init__(self, species, n_particles, dim_phys, L, return_sorted=True):
        """Constructor of UniformParticles

        Args:
            species (torch.Tensor of shape (n_species,)): Composition of the particles
            n_particles (int): Number of particles
            dim_phys (int): Dimension of the particles
            L (float): Size of the box in each dimension.
            return_sorted (bool): Whether to only return sorted species (default is True)
        """

        # Call the parent constructor
        super().__init__()
        # Get the system characterics
        self.register_buffer('species', torch.sort(species).values, persistent=False)
        self.register_buffer('L', torch.tensor(L), persistent=False)
        self.n_particles = n_particles
        self.data_shape = (n_particles, dim_phys)
        self.return_sorted = return_sorted
        self.build_dist()

    def build_dist(self):
        # Build the distributions
        self.x_dist = torch.distributions.Independent(
            base_distribution=torch.distributions.Uniform(
                low=torch.zeros(*self.data_shape, device=self.L.device),
                high=self.L * torch.ones(*self.data_shape, device=self.L.device),
            ),
            reinterpreted_batch_ndims=len(self.data_shape)
        )

    def sample(self, sample_shape):
        """Sample the distribution

        Args:
            sample_shape (tuple of int): Shape of the samples

        Returns:
            a (torch.Tensor of shape (*sample_shape, n_particles)): Species of each particles
            X (torch.Tensor of shape (*sample_shape, n_particles, 2)): Particles
        """
        # Return everything
        if self.return_sorted:
            # We can just copy the reference species as positions and species are independent
            a = self.species.unsqueeze(0).repeat((sample_shape[0], 1))
        else:
            a = torch.stack([
                self.species[torch.randperm(self.n_particles, device=self.L.device)]
                for _ in range(sample_shape[0])
            ], dim=0)
        X = torch.remainder(self.x_dist.sample(sample_shape), self.L)
        return a, X

    def log_prob(self, a, X):
        """Evaluate the log-likelihood of the distribution

        Args:
            * a (torch.Tensor of shape (batch_size, n_particles)): Species of each particles
            * X (torch.Tensor of shape (batch_size, n_particles, 2)): Particles

        Returns:
            * log_prob (torch.Tensor of shape (batch_size,)): Log-likelihood
        """

        # Sort a (we don't care about sorting X because the uniform is dependent)
        a_ = torch.gather(a, 1, a.argsort(dim=-1))
        # Check the composition
        batch_size = a.shape[0]
        mask = (a_ != self.species.unsqueeze(0).expand((batch_size, -1)))
        mask = mask.float().sum(dim=-1) > 0
        # Note that to make things normalized we should add a -math.log(math.factorial(self.n_particles))
        return torch.where(mask, -float('inf') * torch.ones((batch_size,), device=X.device),
                           self.x_dist.log_prob(X))

    def _apply(self, fn):
        """Apply the fn function on the distribution

        Args:
            fn (function): Function to apply on tensors
        """

        new_self = super(UniformParticles, self)._apply(fn)
        new_self.build_dist()
        return new_self
