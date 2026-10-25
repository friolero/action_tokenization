import torch
from torch import nn
from torch.distributions.normal import Normal
from zuko.utils import odeint


class IndependentGaussianDist:
    """Factorized Gaussian prior; exposes a per-dimension inverse CDF for building Sobol codebooks."""

    def __init__(self, sample_dim, loc=None, std=None):
        self.sample_dim = sample_dim
        self._loc = torch.zeros(sample_dim) if loc is None else loc
        self._std = torch.ones(sample_dim) if std is None else std
        assert self._loc.shape[-1] == sample_dim, "Wrong loc dimension."
        assert self._std.shape[-1] == sample_dim, "Wrong std dimension."
        self._dist = [Normal(self._loc[i], self._std[i]) for i in range(self.sample_dim)]

    def icdf(self, cdfs):
        """Map (N, sample_dim) values in (0, 1) to prior samples, dimension by dimension."""
        return torch.cat([self._dist[i].icdf(cdfs[:, i]).reshape(-1, 1) for i in range(self.sample_dim)], dim=1)


class FlowMatchingLoss(nn.Module):
    """Conditional flow matching loss with a linear (rectified-flow) interpolation path."""

    def __init__(self, sigma_min=0.0):
        super().__init__()
        self.sigma_min = sigma_min

    def forward(self, flow, x, c=None):
        z = torch.randn_like(x)
        t = torch.rand(x.shape[0]).to(x.device)
        for _ in x.shape[1:]:
            t = t.unsqueeze(-1)
        psi_t = (1 - (1 - self.sigma_min) * t) * z + t * x
        d_psi = x - (1 - self.sigma_min) * z
        fml = (flow.net(psi_t, t.squeeze(), c=c) - d_psi.detach()).square()
        return fml.mean()


class ZukoPFlow(nn.Module):
    """Probability-flow ODE solved with zuko's adaptive ``odeint``. t=0 is the prior, t=1 the data."""

    def __init__(self, net, prior_dist, t_limit=None):
        super().__init__()
        self.prior_dist = prior_dist
        self.t_limit = t_limit
        self.net = net

    def forward(self, t, x):
        if len(t.shape) == 0:
            t = t.unsqueeze(0).repeat(x.shape[0]).to(x)
        if self.t_limit is not None:
            t = torch.clip(t, self.t_limit[0], self.t_limit[1])
        return self.net(x, t)

    def encode(self, tgt_x):
        """Data -> prior."""
        return odeint(self, tgt_x, 1.0, 0.0, phi=self.net.parameters())

    def decode(self, src_x):
        """Prior -> data."""
        return odeint(self, src_x, 0.0, 1.0, phi=self.net.parameters())
