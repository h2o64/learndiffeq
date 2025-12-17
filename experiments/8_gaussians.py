# Libraries
from learndiffeq.common.datamodules import FiniteDistributionDataModule
from learndiffeq.common.plot_utils import plot_trajectories_2d, plot_samples_2d
from learndiffeq.distributions.circle_mixture import CircularMixture
from learndiffeq.interpolants.ode import InterpolantODE
from learndiffeq.interpolants.sde import InterpolantSDE
from learndiffeq.flow_matching.flow_matching import FlowMatching
from learndiffeq.ml.maximum_likelihood import MaximumLikelihood
import matplotlib.pyplot as plt
import pytorch_lightning as pl


def main(
        device,
        eq_type,
        reverse_time=False,
        n_samples=512,
        batch_size=32,  # 4096,
        save_path=None,
        load_path=None):
    """Main function called when executing the script

    Args
        device (torch.device): Device used for computations
        eq_type (str): Which type of algorithm to use (either 'ode', 'sde', 'fm' or 'ml')
        reverse_time (bool): Whether to sample from rho1 and go to rho0
            (default is False)
        n_samples (int): Number of samples used in plots
            (default is 512)
        batch_size (int): Batch size when training
            (default is 4096)
        save_path (str): Path to store the data
            (default is None which means no saving)
        load_path (str): Path to load the data from
            (default is None which means no loading)
    """

    # Make the distributions
    dist1 = torch.distributions.MultivariateNormal(
        loc=torch.zeros((2,), device=device),
        covariance_matrix=torch.eye(2, device=device)
    )
    dist2 = CircularMixture(device=device)

    # Load the data
    dm = FiniteDistributionDataModule(
        dist=dist2, batch_size=batch_size, dataset_length=60000)

    # Make the trainer
    trainer = pl.Trainer(
        accelerator='auto',
        max_epochs=1 if eq_type != 'ml' else 1,
        logger=pl.loggers.TensorBoardLogger('lightning_logs/'),
        callbacks=[pl.callbacks.progress.TQDMProgressBar(refresh_rate=10)]
    )

    # Make the parameters
    base_params = {
        'data_shape': (2,),
        'rho0': dist1
    }
    if eq_type == 'fm':
        interpolant_params = {
            'interpolation_type': 'linear',
            'sigma': 1e-3,
        }
        networks_params = {
            'lr': 1e-3,
            'velocity_type': 'mlp,128,128'
        }
    elif eq_type == 'ml':
        base_params = {
            'solver_method': 'euler',
            'solver_n_steps': 64,
            'approx': False,
            **base_params
        }
        networks_params = {
            'lr': 1e-3,
            'velocity_type': 'mlp,16,16,16'
        }
    else:
        interpolant_params = {
            'interpolant_type': 'linear',
            'gamma_type': 'brownian',
        }
        if eq_type == 'ode':
            networks_params = {
                'lr': 1e-3,
                'losses': ['b'],
                'velocity_types': {
                    'b': 'mlp,128,128'
                }
            }
        else:
            interpolant_params['epsilon'] = 0.5
            networks_params = {
                'lr': 1e-3,
                'losses': ['v', 's'],
                'velocity_types': {
                    'v': 'mlp,128,128',
                    's': 'mlp,128,128'
                }
            }

    # Build the model
    if eq_type == 'fm':
        model = FlowMatching(**base_params, **interpolant_params, **networks_params)
    elif eq_type == 'ml':
        model = MaximumLikelihood(**base_params, **networks_params)
    elif eq_type == 'ode':
        model = InterpolantODE(**base_params, **interpolant_params, **networks_params)
    else:
        model = InterpolantSDE(**base_params, **interpolant_params, **networks_params)

    # Fit the model
    if load_path:
        # Load a save if possible
        filepath = "{}/8_gaussians_{}.ckpt".format(load_path, eq_type)
        if eq_type == 'fm':
            model = FlowMatching.load_from_checkpoint(checkpoint_path=filepath)
        elif eq_type == 'ml':
            model = MaximumLikelihood.load_from_checkpoint(checkpoint_path=filepath)
        elif eq_type == 'ode':
            model = InterpolantODE.load_from_checkpoint(checkpoint_path=filepath)
        else:
            model = InterpolantSDE.load_from_checkpoint(checkpoint_path=filepath)
    else:
        # Otherwise, train it
        filepath = "{}/8_gaussians_{}.ckpt".format(save_path, eq_type)
        trainer.fit(model, dm)
        if save_path:
            trainer.save_checkpoint(filepath)

    # Move model to the device and disable gradients
    model = model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    # Get intermediate samples
    dist1_samples = dist1.sample(sample_shape=(n_samples,))
    dist2_samples = dist2.sample(sample_shape=(n_samples,))
    if reverse_time:
        dist1_samples, dist2_samples = dist2_samples, dist1_samples
        intermediates = model.sample(
            dist1_samples, n_steps=250, reverse_time=True)
    else:
        intermediates = model.sample(dist1_samples, n_steps=250)
    intermediates = intermediates.detach().cpu()
    dist1_samples = dist1_samples.detach().cpu()
    dist2_samples = dist2_samples.detach().cpu()

    # Plot everything
    plt.subplot(1, 2, 1)
    plot_samples_2d(intermediates, target_samples=dist2_samples,
                    base_samples=dist1_samples)
    plt.xlim(dist2.x_min, dist2.x_max)
    plt.ylim(dist2.y_min, dist2.y_max)
    plt.subplot(1, 2, 2)
    plot_trajectories_2d(intermediates, base_samples=dist1_samples)
    plt.xlim(dist2.x_min, dist2.x_max)
    plt.ylim(dist2.y_min, dist2.y_max)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Libraries
    import torch
    import argparse
    # Parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("eq_type")
    parser.add_argument('--save_path', type=str, default='')
    parser.add_argument('--load_path', type=str, default='')
    parser.add_argument('--reverse_time', action=argparse.BooleanOptionalAction)
    args = parser.parse_args()
    # Get the Pytorch device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_float32_matmul_precision('medium')
    # Run the experiment
    save_path = None if len(args.save_path) == 0 else args.save_path
    load_path = None if len(args.load_path) == 0 else args.load_path
    main(device, args.eq_type, reverse_time=args.reverse_time,
         save_path=save_path, load_path=load_path)
