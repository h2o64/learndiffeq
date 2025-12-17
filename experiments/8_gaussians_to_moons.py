# Libraries
from learndiffeq.common.datamodules import FiniteDistributionDataModule
from learndiffeq.distributions.circle_mixture import CircularMixture
from learndiffeq.distributions.two_moons_sklearn import TwoMoonsSklearn
from learndiffeq.interpolants.ode import InterpolantODE
from learndiffeq.interpolants.sde import InterpolantSDE
from learndiffeq.flow_matching.flow_matching import FlowMatching
from learndiffeq.common.plot_utils import plot_trajectories_2d, plot_samples_2d
import math
import matplotlib.pyplot as plt
import pytorch_lightning as pl

# Main function


def main(
        device,
        eq_type,
        results_path,
        n_samples=1024,
        batch_size=2048,
        max_epochs=50,
        load_model=False,
        n_steps=256,
        display_samples=False,
        use_ot_coupling=False):
    """Main function

    Args:
        device (torch.device): Device to use for computations
        eq_type (str): Which type of algorithm to use (either 'ode' or 'sde')
        results_path (str): Path to save the results
        n_samples (int): Number of samples to display (default is 1024)
        batch_size (int): Batch size during training (default is 2048)
        max_epochs (int): Number of epochs for training (default is 50)
        load_model (bool): Whether to load the model from the save in results_path/ (default is False)
        n_steps (int): Number of integration steps (default is 256)
        display_samples (bool): Whether to display the samples after training (default is False)
        use_ot_coupling (bool): Whether to use an OT coupling
    """

    # Make the distributions
    dist1 = CircularMixture(device=device, radius=5.0, scale=math.sqrt(0.1))
    dist2 = TwoMoonsSklearn(device=device, affine=(3.0, -1.0), noise=0.2)

    # Load the data
    dm = FiniteDistributionDataModule(
        dist=dist2, batch_size=batch_size, dataset_length=512000, batched=True)

    # Make the trainer
    trainer = pl.Trainer(
        accelerator='auto',
        max_epochs=max_epochs,
        logger=pl.loggers.TensorBoardLogger('lightning_logs/'),
        callbacks=[pl.callbacks.progress.TQDMProgressBar(refresh_rate=10)]
    )

    # Make the parameters
    base_params = {
        'data_shape': (2,),
        'rho0': dist1,
        'use_ot_coupling': use_ot_coupling
    }
    if eq_type == 'fm':
        interpolant_params = {
            'interpolation_type': 'linear',
            'sigma': 1e-3,
        }
        networks_params = {
            'lr': 1e-3,
            'velocity_type': 'mlp,64,64,64'
        }
    else:
        interpolant_params = {
            'interpolant_type': 'linear',
            'gamma_type': 'brownian',
        }
        if eq_type == 'ode':
            networks_params = {
                'lr': 1e-4,
                'losses': ['b'],
                'velocity_types': {
                    'b': 'mlp,64,64,64'
                }
            }
        else:
            interpolant_params['epsilon'] = 0.5
            networks_params = {
                'lr': 1e-4,
                'losses': ['v', 's'],
                'velocity_types': {
                    'v': 'mlp,64,64,64',
                    's': 'mlp,64,64,64'
                }
            }

    # Build the model
    if eq_type == 'fm':
        model = FlowMatching(**base_params, **interpolant_params, **networks_params)
    elif eq_type == 'ode':
        model = InterpolantODE(**base_params, **interpolant_params, **networks_params)
    else:
        model = InterpolantSDE(**base_params, **interpolant_params, **networks_params)

    # Fit the model
    filepath = "{}/8_gaussians_to_moons_{}.ckpt".format(results_path, eq_type)
    if load_model:
        if eq_type == 'fm':
            model = FlowMatching.load_from_checkpoint(checkpoint_path=filepath,
                weights_only=False)
        elif eq_type == 'ode':
            model = InterpolantODE.load_from_checkpoint(checkpoint_path=filepath,
                weights_only=False)
        else:
            model = InterpolantSDE.load_from_checkpoint(checkpoint_path=filepath,
                weights_only=False)
    else:
        trainer.fit(model, dm)
        trainer.save_checkpoint(filepath)

    # Move model to the device and disable gradients
    model = model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    # Set the seed
    seed = 42
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.manual_seed_all(seed)

    # Skip if display_samples if False
    if display_samples:

        # Get intermediate samples
        dist1_samples = dist1.sample(sample_shape=(n_samples,))
        dist2_samples = dist2.sample(sample_shape=(n_samples,))
        intermediates = model.sample(dist1_samples, n_steps=n_steps)
        intermediates = intermediates.detach().cpu()
        dist1_samples = dist1_samples.detach().cpu()
        dist2_samples = dist2_samples.detach().cpu()

        # Plot everything
        plt.subplot(1, 2, 1)
        plot_samples_2d(intermediates, target_samples=dist2_samples,
                        base_samples=dist1_samples)
        plt.xlim(min(dist1.x_min, dist2.x_min), max(dist1.x_max, dist2.x_max))
        plt.ylim(min(dist1.y_min, dist2.y_min), max(dist1.y_max, dist2.y_max))
        plt.subplot(1, 2, 2)
        plot_trajectories_2d(intermediates, base_samples=dist1_samples)
        plt.xlim(min(dist1.x_min, dist2.x_min), max(dist1.x_max, dist2.x_max))
        plt.ylim(min(dist1.y_min, dist2.y_min), max(dist1.y_max, dist2.y_max))
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    # Libraries
    import torch
    import argparse
    # Parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("eq_type")
    parser.add_argument('--results_path', type=str)
    parser.add_argument('--n_samples', type=int, default=1024)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--max_epochs', type=int, default=50)
    parser.add_argument('--load_model', action=argparse.BooleanOptionalAction)
    parser.add_argument('--n_steps', type=int, default=256)
    parser.add_argument('--display_samples', action=argparse.BooleanOptionalAction)
    parser.add_argument('--use_ot_coupling', action=argparse.BooleanOptionalAction)
    args = parser.parse_args()
    # Get the Pytorch device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_float32_matmul_precision('medium')
    # Run the experiment
    main(
        device=device,
        eq_type=args.eq_type,
        results_path=args.results_path,
        n_samples=args.n_samples,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        load_model=args.load_model,
        n_steps=args.n_steps,
        display_samples=args.display_samples,
        use_ot_coupling=args.use_ot_coupling
    )
