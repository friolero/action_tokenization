import math

import torch
from einops import rearrange
from torch import nn
from torch.nn import functional as F

from cnf import FlowMatchingLoss, IndependentGaussianDist, ZukoPFlow
from flow_backbones import MLP


def find_nearest(query, dictionary, max_bs=8 * 16 * 16 * 1024 * 10):
    """Nearest-neighbour lookup of ``query`` (N, D) in ``dictionary`` (K, D), chunked to bound memory."""
    dict_size = dictionary.shape[0]
    n_batch = max(1, math.ceil(query.numel() * dict_size / max_bs))
    bs = math.ceil(query.shape[0] / n_batch)

    zq_inds, zqs = [], []
    for i in range(n_batch):
        q = query[i * bs : (i + 1) * bs]
        dists = ((q.unsqueeze(1) - dictionary.unsqueeze(0)) ** 2).sum(-1)
        min_inds = dists.min(1)[1]
        zq_inds.append(min_inds)
        zqs.append(dictionary[min_inds])
    return torch.cat(zq_inds, 0), torch.cat(zqs, 0)


class FlowQuantizer(nn.Module):
    """FlowVQ codebook.

    A flow-matching model maps a standard Gaussian onto the encoder latent distribution. The codebook
    is obtained by pushing a scrambled Sobol (quasi-random) grid of the Gaussian prior through the flow,
    optionally followed by a few Lloyd iterations on dense flow samples.
    """

    def __init__(self, k_dim, x_dim, h_dim=256, n_hiddens=3):
        super().__init__()

        self.k_dim = k_dim
        self.x_dim = x_dim

        self.prior_dist = IndependentGaussianDist(self.x_dim, loc=torch.zeros(self.x_dim), std=torch.ones(self.x_dim))
        net = MLP(x_dim=self.x_dim, h_dim=h_dim, n_hidden=n_hiddens)
        self.flow = ZukoPFlow(net, self.prior_dist, t_limit=(0, 1))

        self.embedding = nn.Embedding(self.k_dim, self.x_dim)

        # NOTE: the scrambled Sobol engine draws from the global torch RNG. Keep the draw order below
        # (dense samples first, then the prior codebook) to reproduce the paper's codebooks.
        self.sobol = torch.quasirandom.SobolEngine(dimension=self.prior_dist.sample_dim, scramble=True)
        eps = 5e-3
        self.sample_ratio = 50
        self.sample_cdfs = self.sobol.draw(self.sample_ratio * self.k_dim).clamp(eps, 1 - eps)
        cdf_vals = self.sobol.draw(self.k_dim)
        self.register_buffer("prior_codebook", self.prior_dist.icdf(cdf_vals))
        self.initialized = False

    @property
    def codebook(self):
        return self.embedding.weight.data

    @property
    def vocab_t(self):
        """The prior codebook pushed forward through the flow."""
        with torch.no_grad():
            return self.flow.decode(self.prior_codebook)

    def set_codebook(self, codebook_vectors):
        with torch.no_grad():
            self.embedding.weight.copy_(codebook_vectors)

    @torch.no_grad()
    def optim_centroids(self, centroids, optim_steps=0, batch_size=1024):
        """Run ``optim_steps`` Lloyd iterations of ``centroids`` against dense samples from the flow."""
        centroids = centroids.clone()
        if optim_steps == 0:
            return centroids

        n_batch = math.ceil(self.k_dim * self.sample_ratio / batch_size)
        sample_priors = self.prior_dist.icdf(self.sample_cdfs).to(centroids.device)
        samples = torch.cat(
            [self.flow.decode(sample_priors[i * batch_size : (i + 1) * batch_size]) for i in range(n_batch)], 0
        )
        for n_iter in range(optim_steps):
            zq_inds = torch.cat(
                [find_nearest(samples[i * batch_size : (i + 1) * batch_size], centroids)[0] for i in range(n_batch)],
                0,
            )
            for i in range(self.k_dim):
                if (zq_inds == i).sum() > 0:
                    centroids[i] = samples[zq_inds == i].mean(0)
                else:
                    print(f"==> Lloyd iter {n_iter}: code {i} has no samples, keeping it unchanged")
        return centroids

    def get_vq_loss(self, ze_batches):
        """Mean L1 quantization error of a stack of latent sequences (B, T, D)."""
        vq_losses = []
        for z_e in ze_batches:
            z_q = find_nearest(z_e, self.codebook)[1]
            vq_losses.append((z_q - z_e).abs())
        return torch.cat(vq_losses, 0).mean()

    def update_codebook(self, pushforward=True, optim_steps=0):
        """Reset the codebook to the flow push-forward of the prior codebook (+ Lloyd refinement)."""
        if pushforward:
            print("==> Update the codebook.")
            self.set_codebook(self.optim_centroids(self.vocab_t, optim_steps=optim_steps))
        self.initialized = True

    def forward(self, z):
        vocab = self.embedding.weight
        z_flattened = z.reshape(-1, self.x_dim)
        d = (
            torch.sum(z_flattened**2, dim=1, keepdim=True)
            + torch.sum(vocab**2, dim=1)
            - 2 * torch.einsum("bd,dn->bn", z_flattened, rearrange(vocab, "n d -> d n"))
        )
        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = vocab[min_encoding_indices].reshape(z.shape)
        return min_encoding_indices.reshape(z_q.shape[:-1]), z_q

    def quantize(self, z):
        """Quantize ``z`` (..., D). Before the codebook is initialized this is the identity (plain AE)."""
        if self.initialized:
            zq_inds, z_q = self(z.reshape(-1, z.shape[-1]))
            return zq_inds.reshape(z.shape[:-1]), z_q.reshape(z.shape)
        return torch.ones(z.shape[:-1]).long().to(z.device), z.clone()

    def loss(self, ze):
        flattened_ze = ze.reshape(-1, ze.shape[-1])
        assert flattened_ze.shape[1] == self.x_dim, "Wrong input dimension, should be (..., d)"
        loss = FlowMatchingLoss()(self.flow, flattened_ze)
        return loss, {"latent_fm_loss": loss}


class VectorQuantizer(nn.Module):
    """Plain VQ-VAE codebook (nearest neighbour lookup)."""

    def __init__(self, k_dim, x_dim):
        super().__init__()
        self.n_e = k_dim
        self.e_dim = x_dim
        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)
        self.initialized = True

    @property
    def codebook(self):
        return self.embedding.weight.data

    def forward(self, z):
        z_flattened = z.reshape(-1, self.e_dim)
        d = (
            torch.sum(z_flattened**2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight**2, dim=1)
            - 2 * torch.einsum("bd,dn->bn", z_flattened, rearrange(self.embedding.weight, "n d -> d n"))
        )
        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = self.embedding(min_encoding_indices).reshape(z.shape)
        return min_encoding_indices.reshape(z_q.shape[:-1]), z_q

    def quantize(self, z):
        zq_inds, z_q = self(z.reshape(-1, z.shape[-1]))
        return zq_inds.reshape(z.shape[:-1]), z_q.reshape(z.shape)
