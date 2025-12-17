# Plots for interpolant models

# Libraries
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def plot_samples_2d(intermediates, target_samples=None, base_samples=None):
    """Plot the samples

    Args:
        intermediates (torch.Tensor of shape (n_steps, batch_size, dim)): Samples from the integration
        target_samples (torch.Tensor of shape (batch_size, dim)): Samples from the target
        base_samples (torch.Tensor of shape (batch_size, dim)): Samples from the base
    """

    handles = []
    if target_samples is not None:
        plt.scatter(target_samples[:, 0],
                    target_samples[:, 1], alpha=0.5, color='red')
        handles.append(mpatches.Patch(color='red', label='Target'))
    if base_samples is not None:
        plt.scatter(base_samples[:, 0],
                    base_samples[:, 1], alpha=0.5, color='blue')
        handles.append(mpatches.Patch(color='blue', label='Base'))
    plt.scatter(intermediates[-1, :, 0],
                intermediates[-1, :, 1], alpha=0.5, color='green')
    handles.append(mpatches.Patch(color='green', label='Model'))
    plt.legend(handles=handles)
    plt.title('Samples')


def plot_trajectories_2d(intermediates, base_samples=None, color='black', alpha=0.1):
    """Plot the samples

    Args:
        intermediates (torch.Tensor of shape (n_steps, batch_size, dim)): Samples from the integration
        base_samples (torch.Tensor of shape (batch_size, dim)): Samples from the base
        color (str): Color of the trajectories
        alpha (float): Alpha level for the trajectories
    """

    plot_samples_2d(intermediates, base_samples=base_samples)
    for traj_id in range(intermediates.shape[1]):
        plt.plot(intermediates[:, traj_id, 0],
                 intermediates[:, traj_id, 1], color=color, alpha=alpha)
    plt.title('Trajectories')


def plot_forward_backward_ode(
        base,
        target,
        model,
        xx_data,
        yy_data,
        zz_data,
        xx_latent,
        yy_latent,
        zz_latent,
        grid_size,
        device,
        n_steps=10,
        method='dopri5',
        approx_div=False,
        callback_ax=None,
        dont_show=False):
    """Plot push-forward and push-backward 2D for ODEs"""

    # Compute the push-forward
    zz_data_bwd = model.sample(zz_data, n_steps=n_steps, reverse_time=True, method=method, approx=approx_div)[-1]
    _, zz_data_log_jac = model.sample(zz_data_bwd, n_steps=n_steps,
                                      return_log_jac=True, method=method, approx=approx_div)
    log_prob_forward = base.log_prob(zz_data_bwd) + zz_data_log_jac[-1].flatten()
    # Compute the push-backward
    zz_latent_fwd = model.sample(zz_latent, n_steps=n_steps, method=method, approx=approx_div)[-1]
    _, zz_latent_log_jac = model.sample(zz_latent_fwd, n_steps=n_steps, reverse_time=True, return_log_jac=True,
                                        method=method, approx=approx_div)
    log_prob_backward = target.log_prob(zz_latent_fwd) + zz_latent_log_jac[-1].flatten()
    # Plot everything
    ax = plt.subplot(1, 2, 1)
    plt.contourf(xx_latent, yy_latent, log_prob_backward.exp(
    ).detach().cpu().view((grid_size, grid_size)), 20, cmap='RdPu')
    plt.contour(xx_latent, yy_latent, base.log_prob(zz_latent).exp().detach().cpu(
    ).view((grid_size, grid_size)), 10, colors='k', linestyles='--', alpha=0.3)
    plt.title('Push-backward')
    if callback_ax is not None:
        callback_ax(ax)
    ax = plt.subplot(1, 2, 2)
    plt.contourf(xx_data, yy_data, log_prob_forward.exp().detach(
    ).cpu().view((grid_size, grid_size)), 20, cmap='GnBu')
    plt.contour(xx_data, yy_data, target.log_prob(zz_data).exp().detach().cpu().view(
        (grid_size, grid_size)), 10, colors='k', linestyles='--', alpha=0.3)
    plt.title('Push-forward')
    if callback_ax is not None:
        callback_ax(ax)
    plt.tight_layout()
    if not dont_show:
        plt.show()


def plot_forward_backward_sde(
        base,
        target,
        model,
        xx_data,
        yy_data,
        zz_data,
        xx_latent,
        yy_latent,
        zz_latent,
        grid_size,
        device,
        n_steps=10,
        n_likelihood=10,
        approx_div=False,
        callback_ax=None,
        dont_show=False):
    """Plot push-forward and push-backward 2D for SDEs"""

    # Compute the push-forward
    zz_data_ = zz_data.clone()
    zz_data = zz_data.unsqueeze(1).repeat(1, n_likelihood, 1).view((-1, 2))
    x_ts = model.sample(zz_data, n_steps=n_steps, reverse_time=True).view(
        (n_steps, -1, n_likelihood, 2))
    log_prob_fwd = model.log_prob(
        x_ts, n_steps=n_steps, forward=True, approx=approx_div).squeeze(-1)
    base_x_ts = base.log_prob(x_ts[-1].view((-1, 2))).view(x_ts.shape[1:-1])
    log_prob_forward = torch.mean(base_x_ts - log_prob_fwd, dim=1)
    # Compute the push-backward
    zz_latent_ = zz_latent.clone()
    zz_latent = zz_latent.unsqueeze(1).repeat(1, n_likelihood, 1).view((-1, 2))
    x_ts = model.sample(zz_latent, n_steps=n_steps, reverse_time=False).view(
        (n_steps, -1, n_likelihood, 2))
    log_prob_bwd = model.log_prob(
        x_ts, n_steps=n_steps, forward=False, approx=approx_div).squeeze(-1)
    target_x_ts = target.log_prob(
        x_ts[-1].view((-1, 2))).view(x_ts.shape[1:-1])
    log_prob_backward = torch.mean(target_x_ts - log_prob_bwd, dim=1)
    # Plot everything
    ax = plt.subplot(1, 2, 1)
    plt.contourf(xx_latent, yy_latent, log_prob_backward.exp(
    ).detach().cpu().view((grid_size, grid_size)), 20, cmap='RdPu')
    plt.contour(xx_latent, yy_latent, base.log_prob(zz_latent_).exp().detach().cpu(
    ).view((grid_size, grid_size)), 10, colors='k', linestyles='--', alpha=0.3)
    plt.title('Push-backward')
    if callback_ax is not None:
        callback_ax(ax)
    ax = plt.subplot(1, 2, 2)
    plt.contourf(xx_data, yy_data, log_prob_forward.exp().detach(
    ).cpu().view((grid_size, grid_size)), 20, cmap='GnBu')
    plt.contour(xx_data, yy_data, target.log_prob(zz_data_).exp().detach().cpu(
    ).view((grid_size, grid_size)), 10, colors='k', linestyles='--', alpha=0.3)
    plt.title('Push-forward')
    if callback_ax is not None:
        callback_ax(ax)
    plt.tight_layout()
    if not dont_show:
        plt.show()
