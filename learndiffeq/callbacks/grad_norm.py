# Track grad norms

# Libraries
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities import grad_norm


class TrackGradNorm(Callback):
    """Track the norm of the gradients"""

    def __init__(self, network_names):
        super().__init__()
        self.network_names = network_names
        self.count = 0

    def on_before_optimizer_step(self, trainer, pl_module, *args, **kwargs):
        # Browse all the networks
        for network_name in self.network_names:
            # Compute the norms
            grad_norm_dict = grad_norm(getattr(pl_module, network_name), 2.0, group_separator='/')
            # Compute the total norm
            total_norm = grad_norm_dict['grad_2.0_norm_total']
            # Log the total norm
            trainer.logger.experiment.add_scalar("grad_norm_" + network_name, total_norm, global_step=self.count)
        self.count += 1
