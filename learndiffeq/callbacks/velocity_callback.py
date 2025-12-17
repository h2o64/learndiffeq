# Libraries
import torch
import matplotlib.pyplot as plt
from pytorch_lightning.callbacks import Callback
from .utils import plot_to_tensorboard, set_axis_white


def make_grid(x_min, x_max, y_min, y_max, res):
    """Make a meshgrid

    Used for plt.contour(X, Y, Z)

    Args:
        x_min (float): Minimum value along x-axis
        x_max (float): Maximum value along y-axis
        y_min (float): Minimum value along x-axis
        y_max (float): Maximum value along y-axis
        res (int): Resolution of the meshgrid

    Returns:
        X (torch.tensor for shape (res, res))
        Y (torch.tensor for shape (res, res))
        Z (torch.tensor for shape (res * res, 2))
    """

    x = torch.linspace(x_min, x_max, res)
    y = torch.linspace(y_min, y_max, res)
    X, Y = torch.meshgrid([x, y], indexing='xy')
    Z = torch.stack([X, Y], dim=-1).view((-1, 2))
    return X, Y, Z


class VelocityCallback(Callback):
    """Callback to display velocity fields"""

    def __init__(self, target, res=256, every_n_epoch=1):
        """Constructor

        Args:
            target (distribution from learndiffeq.distributions or torch.distributions.Distribution
                providing x_min/x_max/y_min/y_max): Target distribution
            res (int): Resolution of the plot (default is 256)
            every_n_epoch (int): How frequent should the callback be called (default is 1)
        """

        # Call the constructor's callback
        super().__init__()
        # Parameters for display
        self.x_min, self.x_max = target.x_min, target.x_max
        self.y_min, self.y_max = target.y_min, target.y_max
        self.res = res
        self.X, self.Y, self.Z = None, None, None
        # When to run the callback
        self.every_n_epoch = every_n_epoch

    def plot_velocity_field(self, v, device):
        """Plot a velocity field

        Using plt.streamplot

        Args:
            v (function taking t of shape (batch_size, 1) and x of shape (batch_size, 2) as inputs and returning a single
                array of shape (batch_size, 2)): Velocity field
            device (torch.device): Device used for the computations
        """

        t = torch.ones((self.Z.shape[0], 1), device=device)
        Z_b = v(t, self.Z).view((self.res, self.res, 2)).detach().cpu().numpy()
        plt.streamplot(self.X, self.Y, Z_b[..., 0], Z_b[..., 1], density=2.0, linewidth=0.5, color='red')
        plt.xlim(self.x_min, self.x_max)
        plt.ylim(self.y_min, self.y_max)

    def on_validation_epoch_start(self, trainer, pl_module, *args, **kwargs):

        # Make the grid
        if self.Z is None:
            self.X, self.Y, self.Z = make_grid(self.x_min, self.x_max, self.y_min, self.y_max, self.res)
            self.Z = self.Z.to(pl_module.device)
            self.X = self.X.numpy()
            self.Y = self.Y.numpy()

        # Skip sometimes
        if (trainer.current_epoch+1) % self.every_n_epoch != 0:
            return

        # Set model to evaluation mode
        pl_module.eval()

        # Browse all the velocity fields
        for velocity_field_name in ['b', 'v', 's']:
            # Check if the model has this velocity field
            if hasattr(pl_module, velocity_field_name):
                # Compute the score
                fig = plt.figure(figsize=(5, 5), facecolor='#303030')
                fig.patch.set_facecolor('#303030')
                # Make the plot
                self.plot_velocity_field(getattr(pl_module, velocity_field_name), pl_module.device)
                # Set the fonts and background color
                ax = plt.gca()
                set_axis_white(ax)
                ax.set_facecolor('#303030')
                ax.grid(alpha=0.1)
                # Log the figure
                plt.tight_layout()
                plot_to_tensorboard(pl_module.logger.experiment, fig, velocity_field_name, trainer.current_epoch)

        # Set model to training model
        pl_module.train()
