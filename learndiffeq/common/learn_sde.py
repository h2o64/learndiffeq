# Learning an SDE

# Libraries
import torch
from torchsde import sdeint
from .learn_base import LearnBase


class LearnSDE(LearnBase):

    def sample(self, values, method='euler_heun', n_steps=1000, reverse_time=False):
        with torch.no_grad():
            t = torch.linspace(0.0, 1.0, n_steps).to(values.device)
            return sdeint(
                sde=self.sde_maker(reverse_time=reverse_time),
                y0=values,
                ts=t,
                method=method
            )

    def sample_ode(self, values, method='euler', n_steps=1000, reverse_time=False, approx=False, return_log_jac=False):
        # Sample the probability flow given by Eq 13 of 2011.13456
        raise NotImplementedError('`sample_ode` function is not yet implemented.')

    def configure_optimizers(self):
        raise NotImplementedError('`configure_optimizers` function is not yet implemented.')
