# Libraries
import matplotlib.pyplot as plt
import torch
from pytorch_lightning.callbacks import Callback
from ...callbacks.utils import plot_to_tensorboard, set_axis_white
from ..utils import gram, gram_torus
from .utils import radial_distribution_function, compute_intersection, compute_correlation, compute_chi_squared


class MarginalCallback(Callback):
    """Callback to plot the marginal distributions of the particles"""

    def __init__(
            self,
            log_prob_fn,
            dim_phys,
            L=None,
            every_n_epoch=1,
            batch_size=8192,
            n_steps=256,
            bins_1d=64,
            solver_args=None):
        """Constructor

        Args:
            log_prob_fn (function): Target log-probability
            dim_phys (int): Dimension of the particles
            L (float): Length of the box (default is None)
            every_n_epoch (int): How frequent should the callback be called (default is 1)
            batch_size (int): Number of samples to use (default is 8192)
            n_steps (int): Number of steps during integration (default is 256)
            bins_1d (int): Resolution for 1D histograms (default is 64)
        """

        # Call the Callback's constructor
        super().__init__()
        # Parameters for integration
        self.batch_size = batch_size
        self.n_steps = n_steps
        self.bins_1d = bins_1d
        # Parameters of the problem
        self.dim_phys = dim_phys
        self.log_prob_fn = log_prob_fn
        self.L = L
        # When to run the callback
        self.every_n_epoch = every_n_epoch
        # Build the solver arguments
        if solver_args is None:
            self.solver_args = {
                'method': 'euler',
                'n_steps': n_steps
            }
        else:
            self.solver_args = solver_args

    def on_fit_start(self, trainer, pl_module, *args, **kwargs):
        """Compute all the neeeded quantities on the dataset"""

        # Get data samples
        dataset = trainer.datamodule.ds_train.tensors
        # Compute the log prob of the dataset
        self.log_prob_target = self.log_prob_fn(dataset[0]).detach().cpu()
        # Compute the pairwise distances of the dataset
        self.n_particles = dataset[0].shape[-2]
        if self.L is not None:
            self.pairwise_distances_target = gram_torus(dataset[0], self.L, only_upper_tri=True).cpu()
        else:
            self.pairwise_distances_target = gram(dataset[0], only_upper_tri=True).cpu()
        self.pairwise_distances_target = self.pairwise_distances_target.view((-1, self.dim_phys))
        self.pairwise_distances_target = torch.linalg.norm(self.pairwise_distances_target, dim=-1)
        # Compute the radial distribution
        self.gr_target = radial_distribution_function(dataset[0], self.L)

    def on_validation_epoch_start(self, trainer, pl_module, *args, **kwargs):

        # Skip sometimes
        if (trainer.current_epoch+1) % self.every_n_epoch != 0:
            return

        # Set model to evaluation mode
        pl_module.eval()

        # Sample the model
        with torch.no_grad():
            base_samples = pl_module.rho0.sample(
                sample_shape=(self.batch_size,)).detach()
            model_samples = pl_module.sample(
                base_samples,
                keep_intermediates=False,
                **self.solver_args
            )
            model_samples = model_samples.detach().cpu()

        # Compute the pairwise distances
        with torch.no_grad():
            if self.L is not None:
                pairwise_distances = gram_torus(model_samples, self.L, only_upper_tri=True).cpu()
            else:
                pairwise_distances = gram(model_samples, only_upper_tri=True).cpu()
            pairwise_distances = torch.linalg.norm(pairwise_distances, dim=-1)

        # Compute the log-probs
        log_prob_model = self.log_prob_fn(model_samples).detach().cpu()

        # Filter the log-probs
        mask = torch.logical_or(
            torch.isneginf(log_prob_model), torch.isinf(log_prob_model)
        )
        perc_wierd_energies = mask.float().mean()
        log_prob_model = log_prob_model[~mask]
        trainer.logger.experiment.add_scalar('perc_wierd_energies', perc_wierd_energies.item(),
                                             global_step=trainer.current_epoch)

        # Compute the log-probs histogram
        fig = plt.figure(figsize=(5, 5), facecolor='#303030')
        fig.patch.set_facecolor('#303030')
        # Compute the range
        log_prob_min = float(self.log_prob_target.min())
        log_prob_max = float(self.log_prob_target.max())
        # Compute the histograms
        log_prob_linspace = torch.linspace(log_prob_min, log_prob_max, self.bins_1d)
        model_hist, _ = torch.histogram(log_prob_model, range=(
            log_prob_min, log_prob_max), density=True, bins=self.bins_1d)
        target_hist, _ = torch.histogram(self.log_prob_target, range=(
            log_prob_min, log_prob_max), density=True, bins=self.bins_1d)
        # Make the plot
        width = torch.min(torch.diff(log_prob_linspace))
        plt.bar(log_prob_linspace, model_hist, label='Model', align='edge', alpha=0.5, width=width)
        plt.bar(log_prob_linspace, target_hist, label='Target', align='edge', alpha=0.5, width=width)
        plt.xlabel('Log-prob. value')
        plt.ylabel('Density')
        plt.legend()
        # Set the fonts and background color
        ax = plt.gca()
        set_axis_white(ax)
        ax.set_facecolor('#303030')
        ax.grid(alpha=0.1)
        # Log the figure
        plt.tight_layout()
        plot_to_tensorboard(pl_module.logger.experiment, fig, 'log_prob_histograms', trainer.current_epoch)

        # Log the statistics of the energies
        target_en_mean = -self.log_prob_target.mean().item()
        target_en_var = (-self.log_prob_target).var().item()
        target_en_cv = (-torch.square(self.log_prob_target)).mean().item() - target_en_mean**2
        model_en_mean = -log_prob_model.mean().item()
        model_en_var = (-log_prob_model).var().item()
        model_en_cv = (-torch.square(log_prob_model)).mean().item() - model_en_var**2
        trainer.logger.experiment.add_scalar('en_mean_diff', model_en_mean - target_en_mean,
                                             global_step=trainer.current_epoch)
        trainer.logger.experiment.add_scalar('en_var_diff', model_en_var - target_en_var,
                                             global_step=trainer.current_epoch)
        trainer.logger.experiment.add_scalar('en_cv_diff', model_en_cv - target_en_cv,
                                             global_step=trainer.current_epoch)

        # Compute various metrics regarding the energy histogram
        model_energy_hist, _ = torch.histogram(
            log_prob_model, range=(
                float(
                    self.log_prob_target.min()), float(
                    self.log_prob_target.max())), density=True, bins=self.bins_1d)
        target_energy_hist, _ = torch.histogram(
            self.log_prob_target, range=(
                float(
                    self.log_prob_target.min()), float(
                    self.log_prob_target.max())), density=True, bins=self.bins_1d)
        trainer.logger.experiment.add_scalar(
            "energy_hist_intersection",
            float(
                compute_intersection(
                    target_energy_hist,
                    model_energy_hist)),
            global_step=trainer.current_epoch)
        trainer.logger.experiment.add_scalar(
            "energy_hist_correlation",
            float(
                compute_correlation(
                    target_energy_hist,
                    model_energy_hist)),
            global_step=trainer.current_epoch)
        trainer.logger.experiment.add_scalar(
            "energy_hist_chi_squared",
            float(
                compute_chi_squared(
                    target_energy_hist,
                    model_energy_hist)),
            global_step=trainer.current_epoch)

        # Compute the histogram of the pairwise distances
        fig = plt.figure(figsize=(5, 5), facecolor='#303030')
        fig.patch.set_facecolor('#303030')
        # Compute the range
        dist_min = float(self.pairwise_distances_target.min())
        dist_max = float(self.pairwise_distances_target.max())
        # Compute the histograms
        pairwise_hist_target, _ = torch.histogram(self.pairwise_distances_target,
                                                  bins=self.bins_1d, range=(dist_min, dist_max), density=True)
        model_pairwise_hist, _ = torch.histogram(pairwise_distances.flatten().cpu().detach(),
                                                 bins=self.bins_1d, range=(dist_min, dist_max), density=True)
        # Make the plot
        hist_pairwise_linespace = torch.linspace(dist_min, dist_max, self.bins_1d)
        width = torch.min(torch.diff(hist_pairwise_linespace))
        plt.bar(hist_pairwise_linespace, model_pairwise_hist, label='Model', align='edge', alpha=0.5, width=width)
        plt.bar(hist_pairwise_linespace, pairwise_hist_target, label='Target', align='edge', alpha=0.5, width=width)
        plt.ylabel('Density')
        plt.xlabel('Distance between particles')
        plt.legend()
        # Set the fonts and background color
        ax = plt.gca()
        set_axis_white(ax)
        ax.set_facecolor('#303030')
        ax.grid(alpha=0.1)
        # Log the figure
        plt.tight_layout()
        plot_to_tensorboard(pl_module.logger.experiment, fig, 'pairwise_distances', trainer.current_epoch)

        # Track the overlap percentage
        min_dist = self.pairwise_distances_target.min()
        perc_overlap = torch.mean((
            torch.sum(pairwise_distances.view((self.batch_size, -1)) < min_dist, dim=-1) > 0
        ).float())
        trainer.logger.experiment.add_scalar(
            "perc_overlap", 100. * float(perc_overlap), global_step=trainer.current_epoch)

        # Compute the radial distribution
        fig = plt.figure(figsize=(5, 5), facecolor='#303030')
        fig.patch.set_facecolor('#303030')
        # Compute the radial distribution of the model
        gr_model = radial_distribution_function(model_samples, self.L)
        # Plot everything
        plt.plot(gr_model[:, 0], gr_model[:, 1], linewidth=2, label='Model')
        plt.plot(self.gr_target[:, 0], self.gr_target[:, 1], linewidth=2, label='Data')
        plt.ylabel(r'$g(r)$')
        plt.yscale('log')
        plt.xlabel(r'$r / \sigma$')
        plt.legend()
        # Set the fonts and background color
        ax = plt.gca()
        set_axis_white(ax)
        ax.set_facecolor('#303030')
        ax.grid(alpha=0.1)
        # Log the figure
        plt.tight_layout()
        plot_to_tensorboard(pl_module.logger.experiment, fig, 'radial_distribution', trainer.current_epoch)

        # Set model to training model
        pl_module.train()
