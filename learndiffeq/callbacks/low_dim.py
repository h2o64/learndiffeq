# Libraries
import matplotlib.pyplot as plt
import torch
from pytorch_lightning.callbacks import Callback
from .utils import plot_to_tensorboard, set_axis_white
from learndiffeq.common.plot_utils import plot_trajectories_2d, plot_samples_2d, plot_forward_backward_ode


def callback_ax(ax):
    """Quick function to apply dark-mode transformations to a matplotlib.axes.Axes"""
    ax = plt.gca()
    set_axis_white(ax)
    ax.set_facecolor('#303030')
    ax.grid(alpha=0.1)


class DistributionCallback(Callback):
    """Callback to plot the samples, trajectories and densities when working on 2D distributions"""

    def __init__(
            self,
            rho1,
            log_prob_fn,
            every_n_epoch=1,
            batch_size=512,
            n_steps=128,
            grid_size=128,
            skip_density=True):
        """Constructor

        Args:
            rho1 (distribution from learndiffeq.distributions or torch.distributions.Distribution
                providing x_min/x_max/y_min/y_max): Target distribution
            log_prob_fn (function): Log-likelihood function of the target
            every_n_epoch (int): How frequent should the callback be called (default is 1)
            batch_size (int): Number of samples to use (default is 512)
            n_steps (int): Number of steps during integration (default is 128)
            grid_size (int): Resolution for density plots (default is 128)
            skip_density (bool): Whether to not compute densities during the callback (default is True)
        """

        # Call the Callback's constructor
        super().__init__()
        # Parameters for integration
        self.batch_size = batch_size
        self.n_steps = n_steps
        # Parameters of the problem
        self.log_prob_fn = log_prob_fn
        # rho1 distribution
        self.rho1 = rho1
        # Plotting limits
        self.x_lims = (rho1.x_min, rho1.x_max)
        self.y_lims = (rho1.y_min, rho1.y_max)
        self.grid_size = grid_size
        self.rho0_grids, self.rho1_grids = None, None
        # When to run the callback
        self.every_n_epoch = every_n_epoch
        self.skip_density = skip_density

    def make_grid(self, device):
        """Make meshgrids for plt.contourf"""
        # Make the edge linspaces
        x_linspace = torch.linspace(*self.x_lims, self.grid_size)
        y_linspace = torch.linspace(*self.y_lims, self.grid_size)
        # Make the grid for rho0 (assumed normal)
        linspace_latent = torch.linspace(-3, 3, self.grid_size)
        xx_latent, yy_latent = torch.meshgrid(linspace_latent.clone(), linspace_latent.clone(), indexing='xy')
        zz_latent = torch.cat([xx_latent.unsqueeze(2), yy_latent.unsqueeze(2)], 2).view(-1, 2)
        zz_latent = zz_latent.float()
        # Make the grid for rho1
        xx_data, yy_data = torch.meshgrid(x_linspace, y_linspace, indexing='xy')
        zz_data = torch.cat([xx_data.unsqueeze(2), yy_data.unsqueeze(2)], 2).view(-1, 2)
        zz_data = zz_data.float()
        # Return the grids
        return (xx_latent, yy_latent, zz_latent.to(device)), (xx_data, yy_data, zz_data.to(device))

    def on_validation_epoch_start(self, trainer, pl_module, *args, **kwargs):

        # Skip sometimes
        if (trainer.current_epoch+1) % self.every_n_epoch != 0:
            return

        # Set model to evaluation mode
        pl_module.eval()

        # Sample the base and target
        dist0_samples = pl_module.rho0.sample(sample_shape=(self.batch_size,))
        dist1_samples = self.rho1.sample(sample_shape=(self.batch_size,))
        device = dist1_samples.device

        # Make the grids if needed
        if self.rho0_grids is None or self.rho1_grids is None:
            self.rho0_grids, self.rho1_grids = self.make_grid(device)

        # Sample the model
        with torch.no_grad():
            intermediates = pl_module.sample(dist0_samples, n_steps=self.n_steps).detach().cpu()
        intermediates = intermediates.detach().cpu()
        dist0_samples = dist0_samples.detach().cpu()
        dist1_samples = dist1_samples.detach().cpu()

        # Plot the samples
        fig = plt.figure(figsize=(5, 5), facecolor='#303030')
        fig.patch.set_facecolor('#303030')
        plot_samples_2d(intermediates, target_samples=dist1_samples, base_samples=dist0_samples)
        plt.xlim(*self.x_lims)
        plt.ylim(*self.y_lims)
        # Set the fonts and background color
        ax = plt.gca()
        set_axis_white(ax)
        ax.set_facecolor('#303030')
        ax.grid(alpha=0.1)
        # Log the figure
        plt.tight_layout()
        plot_to_tensorboard(pl_module.logger.experiment, fig, 'samples', trainer.current_epoch)

        # Plot the trajectories
        fig = plt.figure(figsize=(5, 5), facecolor='#303030')
        fig.patch.set_facecolor('#303030')
        plot_trajectories_2d(intermediates, base_samples=dist0_samples, color='white', alpha=0.1)
        plt.xlim(*self.x_lims)
        plt.ylim(*self.y_lims)
        # Set the fonts and background color
        ax = plt.gca()
        set_axis_white(ax)
        ax.set_facecolor('#303030')
        ax.grid(alpha=0.1)
        # Log the figure
        plt.tight_layout()
        plot_to_tensorboard(pl_module.logger.experiment, fig, 'trajectories', trainer.current_epoch)

        # Plot the density
        if not self.skip_density:
            fig = plt.figure(figsize=(10, 5), facecolor='#303030')
            fig.patch.set_facecolor('#303030')
            plot_forward_backward_ode(
                pl_module.rho0,
                self.rho1,
                pl_module,
                *self.rho1_grids,
                *self.rho0_grids,
                self.grid_size,
                device,
                n_steps=self.n_steps,
                callback_ax=callback_ax,
                dont_show=True)
            # Log the figure
            plt.tight_layout()
            plot_to_tensorboard(pl_module.logger.experiment, fig, 'density', trainer.current_epoch)

        # Set model to training model
        pl_module.train()
