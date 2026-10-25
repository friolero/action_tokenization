import math

import torch
from torch import nn


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=x.device) * -emb)
        emb = x[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class MLP(nn.Module):
    """Velocity field v(x, t[, c]) of the flow-matching model."""

    def __init__(self, x_dim, c_dim=None, h_dim=256, n_hidden=3):
        super().__init__()
        self.x_dim = x_dim
        self.c_dim = c_dim

        self.x_mlp = nn.Linear(x_dim, h_dim)
        self.t_enc = SinusoidalPosEmb(h_dim)
        if c_dim is not None:
            self.c_mlp = nn.Linear(c_dim, h_dim)

        self.mid_layer = nn.ModuleList()
        for _ in range(n_hidden):
            self.mid_layer.extend([nn.Linear(h_dim, h_dim), nn.ELU()])
        self.final_layer = nn.Linear(h_dim, self.x_dim)

    def forward(self, x, t, c=None):
        h = self.x_mlp(x) + self.t_enc(t)
        if c is not None:
            h += self.c_mlp(c)
        for layer in self.mid_layer:
            h = layer(h)
        return self.final_layer(h)
