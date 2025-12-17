# Libraries
import matplotlib.pyplot as plt
import torch
import math
from pytorch_lightning.callbacks import Callback
from ...callbacks.utils import plot_to_tensorboard, set_axis_white
from ..utils import gram_species_torus
from .utils import _volume_sphere, compute_intersection, compute_correlation, compute_chi_squared


class MarginalCallbackSpecies(Callback):
    """Callback to plot the marginal distributions of the particles with different species"""

    def __init__(
            self,
            log_prob_fn,
            dim_phys,
            ref_species,
            n_species,
            L,
            T,
            every_n_epoch=1,
            batch_size=8192,
            n_steps=256,
            bins_1d=64,
            solver_args=None):
        """Constructor

        Args:
            log_prob_fn (function): Target log-probability
            dim_phys (int): Dimension of the particles
            ref_species (torch.Tensor of shape (n_particles,)): Species
            n_species (int): Number of species
            L (float): Length of the box
            T (float): Target temperature
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
        self.T = T
        # Reference specifies
        self.n_species = n_species
        self.ref_species = ref_species
        n_species_mesh = int(0.5 * self.n_species * (self.n_species + 1))
        self.n_species_rows = int(math.sqrt(n_species_mesh))
        self.n_species_cols = int(n_species_mesh / self.n_species_rows)
        if n_species_mesh % self.n_species_cols != 0:
            self.n_species_rows += 1
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

    def _run_callback(self, trainer, pl_module, *args, **kwargs):
        # Set model to evaluation mode
        pl_module.eval()

        # Sample the model
        with torch.no_grad():
            base_a, base_x = pl_module.rho0.sample(sample_shape=(self.batch_size,))
            model_a, model_samples = pl_module.sample(
                a=base_a,
                values=base_x,
                keep_intermediates=False,
                **self.solver_args
            )
            model_a = model_a.detach().cpu()
            model_samples = model_samples.detach().cpu()

        # Compute the pairwise distances
        with torch.no_grad():
            pairwise_distances_model, pairwise_species_model = gram_species_torus(
                model_a, model_samples, self.L, only_upper_tri=True)
            pairwise_distances_norm_model = torch.linalg.norm(pairwise_distances_model, dim=-1)

        # Compute the log-probs
        log_prob_model = self.log_prob_fn(model_a, model_samples).detach().cpu()

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
        plt.xlabel('-E/T')
        plt.ylabel('Density')
        plt.legend()
        # Set the fonts and background color
        ax = plt.gca()
        ax.set_yscale('log')
        set_axis_white(ax)
        ax.set_facecolor('#303030')
        ax.grid(alpha=0.1)
        # Log the figure
        plt.tight_layout()
        plot_to_tensorboard(pl_module.logger.experiment, fig, 'log_prob_histograms', trainer.current_epoch)

        # # Compute the histogram of the raw probabilities
        # fig = plt.figure(figsize=(5, 5), facecolor='#303030')
        # fig.patch.set_facecolor('#303030')
        # # Compute probabilities
        # prob_model = torch.exp(log_prob_model)
        # prob_target = torch.exp(self.log_prob_target)
        # # Compute the range
        # prob_min = float(torch.min(prob_target.min(), prob_model.min()))
        # prob_max = float(torch.max(prob_target.max(), prob_model.max()))
        # # Compute histograms
        # prob_linspace = torch.linspace(prob_min, prob_max, self.bins_1d)
        # model_hist, _ = torch.histogram(prob_model, range=(prob_min, prob_max), density=True, bins=self.bins_1d)
        # target_hist, _ = torch.histogram(prob_target, range=(prob_min, prob_max), density=True, bins=self.bins_1d)
        # # Plot
        # width = torch.min(torch.diff(prob_linspace))
        # plt.bar(prob_linspace, model_hist, label='Model', align='edge', alpha=0.5, width=width)
        # plt.bar(prob_linspace, target_hist, label='Target', align='edge', alpha=0.5, width=width)
        # plt.xlabel('exp(-E/T)')
        # plt.ylabel('Density')
        # plt.legend()
        # # Set the fonts and background color
        # ax = plt.gca()
        # ax.set_yscale('log')
        # set_axis_white(ax)
        # ax.set_facecolor('#303030')
        # ax.grid(alpha=0.1)
        # # Log to TensorBoard
        # plt.tight_layout()
        # plot_to_tensorboard(pl_module.logger.experiment, fig, 'prob_histograms', trainer.current_epoch)

        # Compute raw energies
        U_model = -self.T * log_prob_model
        U_target = -self.T * self.log_prob_target

        # Histogram of energies p(E)
        fig = plt.figure(figsize=(5, 5), facecolor='#303030')
        fig.patch.set_facecolor('#303030')
        energy_min = min(U_model.min().item(), U_target.min().item())
        energy_max = max(U_model.max().item(), U_target.max().item())
        energy_linspace = torch.linspace(energy_min, energy_max, self.bins_1d)

        pU_model, _ = torch.histogram(U_model, range=(energy_min, energy_max), density=True, bins=self.bins_1d)
        pU_target, _ = torch.histogram(U_target, range=(energy_min, energy_max), density=True, bins=self.bins_1d)

        width = torch.min(torch.diff(energy_linspace))
        plt.bar(energy_linspace, pU_model, label='Model', align='edge', alpha=0.5, width=width)
        plt.bar(energy_linspace, pU_target, label='Target', align='edge', alpha=0.5, width=width)
        plt.xlabel('E')
        plt.ylabel('Density')
        plt.legend()
        ax = plt.gca()
        ax.set_yscale('log')
        set_axis_white(ax)
        ax.set_facecolor('#303030')
        ax.grid(alpha=0.1)
        plt.tight_layout()
        plot_to_tensorboard(pl_module.logger.experiment, fig, 'energy_histogram', trainer.current_epoch)

        # Estimate density of states: g(E) = p(E) * exp(E/T)
        gU_model = pU_model * torch.exp(energy_linspace / self.T)
        gU_target = pU_target * torch.exp(energy_linspace / self.T)

        # Plot
        width = torch.min(torch.diff(energy_linspace))
        plt.bar(energy_linspace, gU_model, label='Model', align='edge', alpha=0.5, width=width)
        plt.bar(energy_linspace, gU_target, label='Target', align='edge', alpha=0.5, width=width)
        plt.xlabel('E')
        plt.ylabel('g(E)')
        plt.yscale('log') 
        plt.legend()

        ax = plt.gca()
        set_axis_white(ax)
        ax.set_facecolor('#303030')
        ax.grid(alpha=0.1)

        plt.tight_layout()
        plot_to_tensorboard(pl_module.logger.experiment, fig, 'density_of_states', trainer.current_epoch)

        # Log the statistics of the energies
        target_en_mean = -self.log_prob_target.mean().item() * self.T
        target_en_cv=((torch.square(self.log_prob_target)).mean().item() - torch.square(self.log_prob_target.mean()).item()) / self.n_particles
        model_en_mean = -log_prob_model.mean().item() * self.T
        model_en_cv=((torch.square(log_prob_model)).mean().item() - torch.square(log_prob_model.mean()).item()) / self.n_particles
        trainer.logger.experiment.add_scalar('en_mean_diff', model_en_mean - target_en_mean,
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
        fig = plt.figure(figsize=(self.n_species_cols * 5, self.n_species_rows * 5), facecolor='#303030')
        fig.patch.set_facecolor('#303030')
        # Browse all the species
        a_i, a_j = torch.triu_indices(self.n_species, self.n_species, offset=0)
        for idx, (a, b) in enumerate(zip(a_i, a_j)):
            # Select the data
            pairwise_distances_norm_model_ab = pairwise_distances_norm_model[
                torch.logical_or(
                    torch.logical_and(pairwise_species_model[..., 0] == a, pairwise_species_model[..., 1] == b),
                    torch.logical_and(pairwise_species_model[..., 0] == b, pairwise_species_model[..., 1] == a)
                )
            ]
            pairwise_distances_norm_target_ab = self.pairwise_distances_norm_target[
                torch.logical_or(
                    torch.logical_and(self.pairwise_species_target[..., 0]
                                      == a, self.pairwise_species_target[..., 1] == b),
                    torch.logical_and(self.pairwise_species_target[..., 0]
                                      == b, self.pairwise_species_target[..., 1] == a)
                )
            ]
            # Compute the range
            dist_min = pairwise_distances_norm_target_ab.min().item()
            dist_max = pairwise_distances_norm_target_ab.max().item()
            # Compute the histograms
            pairwise_hist_target, _ = torch.histogram(pairwise_distances_norm_target_ab,
                                                      bins=self.bins_1d, range=(dist_min, dist_max), density=True)
            pairwise_hist_model, _ = torch.histogram(pairwise_distances_norm_model_ab.flatten().cpu().detach(),
                                                     bins=self.bins_1d, range=(dist_min, dist_max), density=True)
            # Make the plot
            ax = plt.subplot(self.n_species_rows, self.n_species_cols, idx+1)
            hist_pairwise_linespace = torch.linspace(dist_min, dist_max, self.bins_1d)
            width = torch.min(torch.diff(hist_pairwise_linespace))
            plt.bar(hist_pairwise_linespace, pairwise_hist_model, label='Model', align='edge', alpha=0.5, width=width)
            plt.bar(hist_pairwise_linespace, pairwise_hist_target, label='Target', align='edge', alpha=0.5, width=width)
            plt.ylabel('Density')
            plt.xlabel('Distance between particles')
            plt.title('{} | {}'.format(a, b))
            plt.legend()
            # Set the fonts and background color
            set_axis_white(ax)
            ax.set_facecolor('#303030')
            ax.grid(alpha=0.1)
        # Log the figure
        plt.tight_layout()
        plot_to_tensorboard(pl_module.logger.experiment, fig, 'pairwise_distances', trainer.current_epoch)

        # Compute the radial distribution
        fig = plt.figure(figsize=(self.n_species_cols * 5, self.n_species_rows * 5), facecolor='#303030')
        fig.patch.set_facecolor('#303030')
        # Browse all the species
        box_volume = self.L**self.dim_phys
        density = self.n_particles / box_volume
        species_unique, counts = torch.unique(self.ref_species, return_counts=True)
        concentration=counts / self.n_particles
        concentration = concentration.cpu()
        a_i, a_j = torch.triu_indices(self.n_species, self.n_species, offset=0)
        for idx, (a, b) in enumerate(zip(a_i, a_j)):
            # Select the data
            pairwise_distances_norm_model_ab = pairwise_distances_norm_model[
                torch.logical_or(
                    torch.logical_and(pairwise_species_model[..., 0] == a, pairwise_species_model[..., 1] == b),
                    torch.logical_and(pairwise_species_model[..., 0] == b, pairwise_species_model[..., 1] == a)
                )
            ]
            pairwise_distances_norm_target_ab = self.pairwise_distances_norm_target[
                torch.logical_or(
                    torch.logical_and(self.pairwise_species_target[..., 0]
                                      == a, self.pairwise_species_target[..., 1] == b),
                    torch.logical_and(self.pairwise_species_target[..., 0]
                                      == b, self.pairwise_species_target[..., 1] == a)
                )
            ]
            # Compute the histograms
            pairwise_hist_target, bins = torch.histogram(pairwise_distances_norm_target_ab, bins=512,
                                                         range=(0., self.L/2))
            pairwise_hist_model, _ = torch.histogram(pairwise_distances_norm_model_ab.flatten().cpu().detach(),
                                                     bins=512, range=(0., self.L/2))
            # Compute the volume of the shell
            volume_shell = _volume_sphere(bins[1:], self.dim_phys) - _volume_sphere(bins[:-1], self.dim_phys)
            # Compute the normalizer
            # normaliser = volume_shell * density * self.batch_size * (self.n_particles - 1) / 2
            n_samples_target=self.pairwise_distances_norm_target.cpu().shape[0]
            n_samples_model=pairwise_distances_norm_model.cpu().shape[0]
            normaliser = self.n_particles * volume_shell * density * concentration[a] * concentration[b] / 2
            # Make the plot
            ax = plt.subplot(self.n_species_rows, self.n_species_cols, idx+1)
            plt.plot((bins[:-1] + bins[1:]) / 2, pairwise_hist_model / normaliser / n_samples_model, linewidth=2, label='Model')
            plt.plot((bins[:-1] + bins[1:]) / 2, pairwise_hist_target / normaliser / n_samples_target, linewidth=2, label='Target')
            plt.ylabel(r'$g(r)$')
            # plt.yscale('log')
            plt.xlabel(r'$r / \sigma$')
            plt.title('{} | {}'.format(a, b))
            plt.legend()
            # Set the fonts and background color
            set_axis_white(ax)
            ax.set_facecolor('#303030')
            ax.grid(alpha=0.1)
        # Log the figure
        plt.tight_layout()
        plot_to_tensorboard(pl_module.logger.experiment, fig, 'radial_distribution', trainer.current_epoch)

        # Compute the norm of the velocity field
        with torch.no_grad():
            device = pl_module.device
            t = torch.rand((self.batch_size, *pl_module.data_shape_ones))
            velocity = pl_module.b(t.to(device), base_x.to(device), a=base_a.to(device))
        trainer.logger.experiment.add_scalar('verlocity_norm', velocity.norm().item(), global_step=trainer.current_epoch)

        # Set model to training model
        pl_module.train()

    def on_fit_start(self, trainer, pl_module, *args, **kwargs):
        """Compute all the neeeded quantities on the dataset"""

        # Get data samples
        dataset = trainer.datamodule.ds_train.tensors
        # Compute the log prob of the dataset
        if len(dataset) == 4:
            species_idx, conf_idx = 3, 2
        else:
            species_idx, conf_idx = 1, 0
        self.log_prob_target = self.log_prob_fn(dataset[species_idx], dataset[conf_idx]).detach().cpu()
        # Compute the pairwise distances of the dataset
        self.n_particles = dataset[conf_idx].shape[-2]
        self.pairwise_distances_target, self.pairwise_species_target = gram_species_torus(
            dataset[species_idx], dataset[conf_idx], self.L, only_upper_tri=True)
        self.pairwise_distances_target = self.pairwise_distances_target.cpu()
        self.pairwise_species_target = self.pairwise_species_target.cpu()
        self.pairwise_distances_norm_target = torch.linalg.norm(self.pairwise_distances_target, dim=-1)

        # Make sure to run callbacks at time 0
        self._run_callback(trainer, pl_module, *args, **kwargs)

    def on_validation_epoch_start(self, trainer, pl_module, *args, **kwargs):

        # Skip sometimes
        if (trainer.current_epoch+1) % self.every_n_epoch != 0:
            return
        
        self._run_callback(trainer, pl_module, *args, **kwargs)
        


       