# Libraries
from ..velocities.simple import make_simple_velocity
from ..velocities.potential_velocity import ScoreBasedVelocity
from ..particles.velocities.particles_velocity import make_particles_velocity


def make_velocity(data_shape, dim, velocity_type, **kwargs):
    """Instanciate a velocity field

    Args:
        data_shape (tuple of int): shape of the data
        velocity_type (str or dict): name of the velocity field to use and structure of the inner MLP (eg, "mlp,128,128,128")
            - name = 'particles_*' : Use velocity fields from the learndiffeq.velocities.particles_velocity
            - name = 'mlp_*'
                - 'mlp_score': Uses ScoreBasedVelocity from learndiffeq.velocities.potential_velocity
                - 'mlp': Uses make_simple_velocity from learndiffeq.velocities.simple
        kwargs (dict): Parameters to pass to the velocity constructor


    Returns:
        b (torch.nn.Module): a velocity field
    """

    # Make the velocity field
    if isinstance(velocity_type, str):
        name = velocity_type.split(',')[0]
        hidden_layers = list(map(int, velocity_type.split(',')[1:]))
        if 'particles' in name:
            b = make_particles_velocity(data_shape, name, hidden_layers, **kwargs)
        elif 'mlp' in name:
            if 'score' in name:
                b = ScoreBasedVelocity(
                    dim,
                    mean_rho1=kwargs.get('mean_rho1'),
                    f=kwargs.get('f'),
                    net_type=kwargs.get('net_type', 'score'),
                    hidden_layers=hidden_layers)
            else:
                b = make_simple_velocity(dim=dim, hidden_layers=hidden_layers)
        else:
            raise NotImplementedError('Velocity type {} not implemented.'.format(velocity_type))
    else:
        raise NotImplementedError('Velocity type {} not implemented.'.format(velocity_type))
    # Return the velocity type
    return b
