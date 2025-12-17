# Libraries
from matplotlib import use
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
from learndiffeq.callbacks.grad_norm import TrackGradNorm
from learndiffeq.common.datamodules import TensorDataModule
from learndiffeq.ml.maximum_likelihood import MaximumLikelihood, MaximumLikelihoodWithTwoDatasets
from learndiffeq.flow_matching.riemannian_fm import RiemannianFlowMatching, RiemannianFlowMatchingWithTwoDatasets
from learndiffeq.flow_matching.flow_matching_species import FlowMatchingWithSpecies
from learndiffeq.particles.callbacks.marginals_species import MarginalCallbackSpecies
from learndiffeq.particles.callbacks.ess import ESSCallbackSpecies
from learndiffeq.callbacks.weight_averaging import WeightAveraging
from learndiffeq.distributions.dataset import DatasetDistributionWrapper
from learndiffeq.particles.distributions.kob_andersen import KobAndersen
from learndiffeq.particles.distributions.soft_spheres import SoftSphere
from learndiffeq.particles.distributions.uniform import UniformParticles
from learndiffeq.particles.utils import separate_by_species, rebuild_from_index

class DataAugmentationCallback(Callback):
    def __init__(self, transform):
        super().__init__()
        self.transform = transform
    def on_train_epoch_start(self, trainer, pl_module, *args, **kwargs):
        trainer.datamodule.refresh_dataset(self.transform)

def randperm_batched(batch_size, N, device):
    """Sample a batch of permutations"""
    # return torch.stack([torch.randperm(N, device=device) for _ in range(batch_size)])
    return torch.argsort(torch.rand(batch_size, N, device=device))

def get_augmentation_parameters(batch_size, indexes_species, n_particles, dim_phys, L, device):
    """Build random augmentations"""
    # Make random within species
    perms_indexes_species = [
        randperm_batched(batch_size, indexes_species[i].shape[1], device).unsqueeze(-1).expand((-1, -1, dim_phys))
            for i in range(len(indexes_species))
    ]
    # Make random translations
    trans = L * torch.rand((batch_size, 1, dim_phys), device=device)
    # Make the random symmetries
    symmetries_perms = randperm_batched(batch_size, dim_phys, device).view((batch_size, 1, dim_phys))
    symmetries_signs = (2 * torch.randint(0, 2, (batch_size, 1, dim_phys), device=device) - 1)
    return perms_indexes_species, trans, symmetries_perms, symmetries_signs

def apply_augmentation_parameters(X, a, L, n_species):
    """Apply the augmentation"""
    if len(X.shape) == 2:
        n_particles, dim_phys = X.shape
        indexes_species = separate_by_species(a.unsqueeze(0), n_species, dim_phys)
        perms_indexes_species, trans, symmetries_perms, symmetries_signs = get_augmentation_parameters(
            1, indexes_species, n_particles, dim_phys, L, X.device)
        indexes_species_permed = [
            torch.gather(indexes_species[i], 1, perms_indexes_species[i]) for i in range(n_species)
        ]
        X = torch.cat([
            rebuild_from_index(i, indexes_species_permed, X.unsqueeze(0)) for i in range(n_species)
        ], dim=1).squeeze(0)
        X += trans[0]
        X = torch.gather(X, 1, symmetries_perms[0].expand((n_particles, -1)))
        X *= symmetries_signs[0]
    else:
        batch_size, n_particles, dim_phys = X.shape
        indexes_species = separate_by_species(a, n_species, dim_phys)
        perms_indexes_species, trans, symmetries_perms, symmetries_signs = get_augmentation_parameters(
            batch_size, indexes_species, n_particles, dim_phys, L, X.device)
        indexes_species_permed = [
            torch.gather(indexes_species[i], 1, perms_indexes_species[i]) for i in range(n_species)
        ]
        X = torch.cat([
            rebuild_from_index(i, indexes_species_permed, X) for i in range(n_species)
        ], dim=1)
        X += trans
        X = torch.gather(X, 2, symmetries_perms.expand((-1, n_particles, -1)))
        X *= symmetries_signs
    return torch.remainder(X, L), a

def make_dataset(dataset_filepath, species_filepath, dataset_size, batch_size):
    # Load the dataset
    dataset = torch.load(dataset_filepath, weights_only=False, map_location=torch.device('cpu')).detach()
    species = torch.load(species_filepath, weights_only=False, map_location=torch.device('cpu')).detach()
    # Build the data module
    perm = torch.randperm(dataset.shape[0])[:dataset_size]
    dataset, species = dataset[perm], species[perm]
    # Sort the species
    idx_sorted = species.argsort(dim=-1)
    species = torch.gather(species, 1, idx_sorted).int()
    dataset = torch.gather(dataset, 1, idx_sorted.unsqueeze(-1).expand((-1, -1, dataset.shape[-1])))
    # Build the dataloader
    dm = TensorDataModule(dataset, species, batch_size=batch_size, shuffle=True)
    return dataset.shape[-2], dataset.shape[-1], species[0], dm


def make_dataset_joint(base_dataset_filepath, base_species_filepath, target_dataset_filepath, target_species_filepath,
                       dataset_size, batch_size):
    # Load the dataset
    base_dataset = torch.load(base_dataset_filepath, weights_only=False,
        map_location=torch.device('cpu')).detach()
    base_species = torch.load(base_species_filepath, weights_only=False,
        map_location=torch.device('cpu')).detach().int()
    target_dataset = torch.load(target_dataset_filepath, weights_only=False,
        map_location=torch.device('cpu')).detach()
    target_species = torch.load(target_species_filepath, weights_only=False,
        map_location=torch.device('cpu')).detach().int()
    # Check the size
    if not ((base_dataset.shape == target_dataset.shape) and (base_species.shape == target_species.shape)):
        raise ValueError('Datasets should have the same size.')
    # Shuffle and cut the datasets
    perm = torch.randperm(base_dataset.shape[0])[:dataset_size]
    base_dataset, base_species = base_dataset[perm], base_species[perm]
    target_dataset, target_species = target_dataset[perm], target_species[perm]
    # Sort the species
    base_idx_sorted = base_species.argsort(dim=-1)
    base_species = torch.gather(base_species, 1, base_idx_sorted)
    base_dataset = torch.gather(base_dataset, 1, base_idx_sorted.unsqueeze(-1).expand((-1, -1, base_dataset.shape[-1])))
    target_idx_sorted = target_species.argsort(dim=-1)
    target_species = torch.gather(target_species, 1, target_idx_sorted)
    target_dataset = torch.gather(
        target_dataset, 1, target_idx_sorted.unsqueeze(-1).expand((-1, -1, target_dataset.shape[-1])))
    # Build the dataloader
    dm = TensorDataModule(base_dataset, base_species, target_dataset,
                          target_species, batch_size=batch_size, shuffle=True)
    return base_dataset.shape[-2], base_dataset.shape[-1], target_species[0], dm, base_species, base_dataset


def make_dist(dist_type, n_particles, dim_phys, n_species):
    if dist_type == 'kba':
        density=1.1920748468939728
        L=(n_particles / density) ** (1. / dim_phys)
        return KobAndersen(
            n_particles=n_particles,
            dim_phys=dim_phys,
            L=L
        ).cpu()
    elif dist_type == 'bhhp':
        density=0.5
        L=(n_particles / density) ** (1. / dim_phys)
        return SoftSphere(
            n_particles=n_particles,
            dim_phys=dim_phys,
            L=L
        ).cpu()
    else:
        raise ValueError('Distribution {} not found.'.format(dist_type))

def main(device, savename, temp, config):

    # Load the large joint dataset
    use_joint_dataset = all([
        len(config['dataset_filepath']) > 0,
        len(config['species_filepath']) > 0,
        len(config['base_dataset_filepath']) > 0,
        len(config['base_species_filepath']) > 0
    ])
    if use_joint_dataset:
        n_particles, dim_phys, ref_species, dm, base_species, base_dataset = make_dataset_joint(
            base_dataset_filepath=config['base_dataset_filepath'],
            base_species_filepath=config['base_species_filepath'],
            target_dataset_filepath=config['dataset_filepath'],
            target_species_filepath=config['species_filepath'],
            dataset_size=config['dataset_size'],
            batch_size=config['batch_size']
        )
    # Make the target distribution and dataset
    if config['target_type'] in ['kba', 'bhhp']:
        # Load the dataset
        if not use_joint_dataset:
            n_particles, dim_phys, ref_species, dm = make_dataset(
                dataset_filepath=config['dataset_filepath'],
                species_filepath=config['species_filepath'],
                dataset_size=config['dataset_size'],
                batch_size=config['batch_size']
            )
        # Make the target
        n_species = torch.unique(ref_species).shape[0]
        target = make_dist(config['target_type'], n_particles, dim_phys, n_species)
        def target_log_prob_fn(a, x): return -target.U(a, x) / temp
    else:
        print('Target type {} unsupported (only kba and bhhp are supported)'.format(
            config['target_type']))
        exit(0)

    # Make the base distribution
    if config['base_type'] == 'uniform':
        base = UniformParticles(
            species=ref_species,
            n_particles=n_particles,
            dim_phys=dim_phys,
            L=float(target.L)
        )
    else:
        base = make_dist(config['base_type'], n_particles, dim_phys, n_species)
    if use_joint_dataset:
        # Wrap the dataset around the distribution
        base = DatasetDistributionWrapper(base, base_species, base_dataset, temp=config['base_temp'])
    base = base.to(device)

    # Wrap things just in case
    if use_joint_dataset:
        base_dataset = torch.remainder(base_dataset, target.L)
    for i in range(len(dm.data_train)):
        if dm.data_train[i].shape[1:] == (n_particles, dim_phys):
            dm.data_train[i] = torch.remainder(dm.data_train[i], target.L)
            dm.data_val[i] = torch.remainder(dm.data_val[i], target.L)
    if isinstance(base, DatasetDistributionWrapper):
        for i in range(base.n_datasets):
            setattr(base, 'dataset_' + str(i),
                torch.remainder(getattr(base, 'dataset_' + str(i)), target.L))

    # Make the callbacks
    callbacks = [
        pl.callbacks.progress.TQDMProgressBar(refresh_rate=10),
        ModelCheckpoint(
            dirpath=config['results_path'] + '/checkpoints/' + savename + '/',
            filename=savename + 'model_{epoch}',
            every_n_epochs=1,
            save_top_k=1,
            save_last=True
        )
    ]
    # Add the data augmentation
    if config['use_data_aug']:
        # Build the transform
        if use_joint_dataset:
            transform = lambda X_0, a_0, X_1, a_1 : (X_0, a_0,
                *apply_augmentation_parameters(X_1, a_1, target.L, n_species))
        else:
            transform = lambda X_1, a_1 : apply_augmentation_parameters(X_1, a_1, target.L, n_species)
        callbacks.append(DataAugmentationCallback(transform))
    if config['use_ema']:
        adjust = config['dataset_size'] * config['batch_size'] * config['ema_steps'] / config['n_epochs']
        alpha = 1.0 - config['ema_decay']
        alpha = min(1.0, alpha * adjust)
        callbacks.append(WeightAveraging(
            device=device,
            avg_fn=torch.optim.swa_utils.get_ema_avg_fn(1. - alpha)
        ))
    if not config['disable_callbacks']:
        callbacks.append(TrackGradNorm(
            network_names=['b']
        ))
        callbacks.append(MarginalCallbackSpecies(
            log_prob_fn=target_log_prob_fn,
            dim_phys=dim_phys,
            L=float(target.L),
            T=temp,
            ref_species=ref_species.to(device),
            n_species=n_species,
            every_n_epoch=config['freq_marginal_callback'],
            bins_1d=128,
            batch_size=config['marginal_batch_size'],
            solver_args={
                'method': 'dopri5',
                'n_steps': 2,
                'atol': 1e-5,
                'rtol': 1e-5
            }
        ))
        if config['enable_ess_callback']:
            callbacks.append(ESSCallbackSpecies(
                target_log_prob_fn=target_log_prob_fn,
                L=float(target.L),
                every_n_epoch=config['freq_ess_callback'],
                n_samples=config['n_particles_ess'],
                method='dopri5',
                n_steps=2,
                approx=False,
                solver_kwargs={
                    'atol': 1e-5,
                    'rtol': 1e-5
                }
            ))

    
    # Network parameters
    networks_params = {
        'lr': config['lr'],
        'velocity_type': config['velocity_type'],
        'velocity_kwargs': {
            'L': None if config['training_algo'] == 'fm' else float(target.L),
            'n_species': n_species,
            }
    }

    if config['weight_decay'] > 0.0:
        networks_params['weight_decay'] = config['weight_decay']
        networks_params['optimizer_type'] = 'adamw'
    if 'particles_equivariant' in config['velocity_type']:
        equivariant_params = {
            'tanh': True,
            'attention': True,
            'hidden_nf': config['egnn_hidden_nf'],
            'n_layers': config['egnn_n_layers']
        }
        networks_params['velocity_kwargs'] = dict(**networks_params['velocity_kwargs'], **equivariant_params)
    if 'mlp_no_angle' in config['velocity_type']:
        networks_params['velocity_kwargs'] = dict(**networks_params['velocity_kwargs'],
            use_angle_encoding=False)

    # Build the trainer
    trainer = pl.Trainer(
        # accelerator='auto',
        accelerator='auto' if torch.cuda.is_available() else 'cpu',
        max_epochs=config['n_epochs'],
        logger=pl.loggers.TensorBoardLogger(config['results_path'] + '/', name=savename),
        callbacks=callbacks,
        gradient_clip_val=config['clipping'] if config['clipping'] > 0 else None,
        reload_dataloaders_every_n_epochs=1 if config['use_data_aug'] else 0
    )

    # Build model
    algo_params = {
        'rho0': base,
        'skip_sorting': True
    }
    if config['training_algo'] == 'ml':
        algo_params['L'] = float(target.L)
        algo_params['data_shape'] = (n_particles, dim_phys)
        algo_params['solver_approx'] = ('particles_equivariant' in config['velocity_type'])
        algo_params['solver_method'] = 'euler'
        algo_params['solver_n_steps'] = 128
        algo_params['force_automatic_div'] = config['force_automatic_div']
        if use_joint_dataset:
            model = MaximumLikelihoodWithTwoDatasets(**algo_params, **networks_params)
        else:
            model = MaximumLikelihood(**algo_params, **networks_params)
    elif config['training_algo'] == 'rfm':
        algo_params['L'] = float(target.L)
        algo_params['n_particles'] = n_particles
        algo_params['dim_phys'] = dim_phys
        algo_params['use_linear_assignment_particles'] = True
        algo_params['force_automatic_div'] = config['force_automatic_div']
        if use_joint_dataset:
            model = RiemannianFlowMatchingWithTwoDatasets(**algo_params, **networks_params)
        else:
            model = RiemannianFlowMatching(**algo_params, **networks_params)
    elif config['training_algo'] == 'fm':
        algo_params['n_particles'] = n_particles
        algo_params['dim_phys'] = dim_phys
        algo_params['use_linear_assignment_particles'] = True
        algo_params['force_automatic_div'] = config['force_automatic_div']
        if use_joint_dataset:
            ValueError(f"Cannot train fm with two datasets")
        else:
            model = FlowMatchingWithSpecies(**algo_params, **networks_params)
    else:
        raise ValueError(f"Unknown training algorithm: {config['training_algo']}")

    print()
    print(f'Target type: {target.__class__.__name__}')
    print(f'Base type: {model.rho0.__class__.__name__}')
    print(f'N: {n_particles}, dim: {dim_phys}, species: {n_species}, T: {config["temp"]}')
    print(f"Algorithm: {model.__class__.__name__}")
    print(f'Velocity type: {model.b.__class__.__name__}')
    print(*[f"\t'{k}': {v}" for k, v in networks_params['velocity_kwargs'].items()], sep="\n")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

    # Launch training
    ckpt_path = config['ckpt_path'] if len(config['ckpt_path']) > 0 else None
    trainer.fit(model, dm, ckpt_path=ckpt_path)


if __name__ == "__main__":

    # Libraries
    import argparse
    import random
    import numpy as np

    # Parse the arguments
    parser = argparse.ArgumentParser()
    # Tensorboard details
    parser.add_argument('--savename', type=str)
    parser.add_argument('--results_path', type=str)
    # Dataset details
    parser.add_argument('--target_type', type=str, default='kba')
    parser.add_argument('--base_type', type=str, default='uniform')
    parser.add_argument('--dataset_filepath', type=str, default='')
    parser.add_argument('--species_filepath', type=str, default='')
    parser.add_argument('--base_dataset_filepath', type=str, default='')
    parser.add_argument('--base_species_filepath', type=str, default='')
    parser.add_argument('--base_temp', type=float, default=1.0)
    parser.add_argument('--dataset_size', type=int, default=100000)
    # Target details
    parser.add_argument('--temp', type=float, default=0.5)
    # Velocity details details
    parser.add_argument('--velocity_type', type=str, default='particles_equivariant')
    parser.add_argument('--egnn_hidden_nf', type=int, default=32)
    parser.add_argument('--egnn_n_layers', type=int, default=3)
    parser.add_argument('--force_automatic_div', action=argparse.BooleanOptionalAction)
    # Callback details
    parser.add_argument('--disable_callbacks', action=argparse.BooleanOptionalAction)
    parser.add_argument('--enable_ess_callback', action=argparse.BooleanOptionalAction)
    parser.add_argument('--freq_ess_callback', type=int, default=1)
    parser.add_argument('--freq_marginal_callback', type=int, default=1)
    parser.add_argument('--marginal_batch_size', type=int, default=2048)
    parser.add_argument('--n_particles_ess', type=int, default=1024)
    # Training details
    parser.add_argument('--training_algo', type=str, default='rfm')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--clipping', type=float, default=5.0)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--n_epochs', type=int, default=1250)
    parser.add_argument('--ckpt_path', type=str, default='')
    parser.add_argument('--use_ema', action=argparse.BooleanOptionalAction)
    parser.add_argument('--ema_decay', type=float, default=0.99998)
    parser.add_argument('--ema_steps', type=int, default=32)
    parser.add_argument('--use_data_aug', action=argparse.BooleanOptionalAction)
    # Seed
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    config = vars(args)

    # Get the Pytorch device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    # torch.set_float32_matmul_precision('medium')

    # Set the seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Run the script
    main(device, args.savename, args.temp, config)
