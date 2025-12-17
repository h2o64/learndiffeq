# Callback to compute the ESS

# Libraries
import torch
import numpy as np
import matplotlib.pyplot as plt
from pytorch_lightning.callbacks import Callback
from .utils import plot_to_tensorboard, set_axis_white


class ESSCallback(Callback):
    """Callback to compute the ESS of the generative model"""

    def __init__(self, target_log_prob_fn, every_n_epoch=1, n_samples=1024, method='euler', bins_1d=64, n_steps=128,
                 approx=False, do_the_log_prob_log_prob_plot=True, solver_kwargs=None):
        """Constructor

        Args:
            target_log_prob_fn (function): Log-likelihood function of the target
            every_n_epoch (int): How frequent should the callback be called (default is 1)
            n_samples (int): Number of samples to use in the ESS computation (default is 1024)
            method (str): Method for integration (default is euler)
            n_steps (int): Number of steps during integration (default is 128)
            approx (bool): Whether to use approximate divergence computations (default is False)
            do_the_log_prob_log_prob_plot (bool): Display the log-probs of the model and target (default is True)
            solver_kwargs (dict): Arguments to pass to the solver
        """

        super().__init__()
        self.target_log_prob_fn = target_log_prob_fn
        self.every_n_epoch = every_n_epoch
        self.n_samples = n_samples
        self.method = method
        self.n_steps = n_steps
        self.approx = approx
        self.bins_1d = bins_1d
        self.do_the_log_prob_log_prob_plot = do_the_log_prob_log_prob_plot
        if solver_kwargs is None:
            self.solver_kwargs = {}
        else:
            self.solver_kwargs = solver_kwargs

    def on_validation_epoch_start(self, trainer, pl_module, *args, **kwargs):
        if (trainer.current_epoch+1) % self.every_n_epoch == 0:
            # Disable gradient with respect to the parameters
            parameters_states = []
            for p in pl_module.b.parameters():
                parameters_states.append(p.requires_grad)
                p.requires_grad = False
            # Sample the ODE
            proposal_samples, log_prob_proposal = pl_module.sample_and_log_prob(
                (self.n_samples,), method=self.method, n_steps=self.n_steps, approx=self.approx, **self.solver_kwargs)
            proposal_samples = proposal_samples.cpu()
            log_prob_proposal = log_prob_proposal.cpu()
            # Disable gradients
            with torch.no_grad():
                # Compute the log-likelihood of the target
                log_prob_target = self.target_log_prob_fn(proposal_samples)
                # Compute the importance weights
                log_weights = log_prob_target - log_prob_proposal
                log_weights = torch.where(torch.isnan(log_weights),
                                          -float('inf') * torch.ones_like(log_weights), log_weights)
                # Compute the ESS
                ess = torch.exp(2*torch.logsumexp(log_weights, dim=-1) \
                    - torch.logsumexp(2*log_weights, dim=-1))
                ess = ess.mean().cpu()
                # Log it
                trainer.logger.experiment.add_scalar(
                    "ess", ess / self.n_samples, global_step=trainer.current_epoch)
            # Do the log-prob / log-prob plot
            if self.do_the_log_prob_log_prob_plot:
                # Compute the histogram of the pairwise distances
                fig = plt.figure(figsize=(5, 5), facecolor='#303030')
                fig.patch.set_facecolor('#303030')
                # Flatten the log_probs
                log_prob_proposal = log_prob_proposal.flatten().cpu()
                log_prob_target = log_prob_target.flatten().cpu()
                mask_bad_proposal = torch.logical_or(
                    torch.isnan(log_prob_proposal), torch.isneginf(log_prob_proposal)
                )
                mask_bad_target = torch.logical_or(
                    torch.isnan(log_prob_target), torch.isneginf(log_prob_target)
                )
                mask = torch.logical_and(~mask_bad_proposal, ~mask_bad_target)
                if int(mask.int().sum()) > 0:
                    # Display the points
                    plt.scatter(log_prob_proposal[mask], log_prob_target[mask], color='white', alpha=0.3)
                    # Perform a linear regression
                    a, b = np.polyfit(log_prob_proposal[mask].numpy(), log_prob_target[mask].numpy(), 1)
                    x = np.linspace(log_prob_proposal[mask].min().item(), log_prob_proposal[mask].max().item(), 512)
                    plt.plot(x, a*x+b, color='white')
                    plt.title('Linear regression (alpha = {:.2f})'.format(a))
                    plt.xlabel('Log-prob. model')
                    plt.ylabel('Log-prob. target')
                    # Set the fonts and background color
                    ax = plt.gca()
                    set_axis_white(ax)
                    ax.set_facecolor('#303030')
                    ax.grid(alpha=0.1)
                    # Log the figure
                    plt.tight_layout()
                    plot_to_tensorboard(pl_module.logger.experiment, fig, 'model_vs_target', trainer.current_epoch)
                    trainer.logger.experiment.add_scalar("alpha_model_vs_target", a, global_step=trainer.current_epoch)

                    fig_hist = plt.figure(figsize=(5, 5), facecolor='#303030')
                ax = fig_hist.add_subplot(111)

                log_prob_min = min(log_prob_proposal.min(), log_prob_target.min()).item()
                log_prob_max = max(log_prob_proposal.max(), log_prob_target.max()).item()
                log_prob_linspace = torch.linspace(log_prob_min, log_prob_max, steps=self.bins_1d)
                model_hist, _ = torch.histogram(log_prob_proposal, range=(log_prob_min, log_prob_max),
                                                density=True, bins=self.bins_1d)
                target_hist, _ = torch.histogram(log_prob_target, range=(log_prob_min, log_prob_max),
                                                density=True, bins=self.bins_1d)
                width = torch.min(torch.diff(log_prob_linspace)).item()
                ax.bar(log_prob_linspace.numpy(), model_hist.numpy(), label='Model', align='edge', alpha=0.5, width=width)
                ax.bar(log_prob_linspace.numpy(), target_hist.numpy(), label='Target', align='edge', alpha=0.5, width=width)
                ax.set_xlabel('log p(x)')
                ax.set_ylabel('Density')
                ax.set_yscale('log')
                ax.legend()
                ax.set_facecolor('#303030')
                set_axis_white(ax)
                ax.grid(alpha=0.1)
                plt.tight_layout()
                plot_to_tensorboard(pl_module.logger.experiment, fig_hist, 'log_density_model_vs_target', trainer.current_epoch)

            # Disable gradient with respect to the parameters
            for i, p in enumerate(pl_module.b.parameters()):
                p.requires_grad_(parameters_states[i])
