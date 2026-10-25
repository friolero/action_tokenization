import math

import torch
import torch.nn as nn

from quantizer import FlowQuantizer, VectorQuantizer


def generate_causal_mask(seq_len):
    """Lower-triangular mask that prevents attending to future positions."""
    return torch.tril(torch.ones(seq_len, seq_len))


class MLPBlock(nn.Module):
    def __init__(self, in_dim, h_dim, out_dim, n_layers=2):
        assert n_layers >= 2
        super().__init__()
        self.map = nn.ModuleList([nn.Linear(in_dim, h_dim)])
        for _ in range(0, n_layers - 2):
            self.map.append(nn.Linear(h_dim, h_dim))
        self.map.append(nn.Linear(h_dim, out_dim))
        self.act = nn.ReLU()
        self.n_layers = n_layers

    def forward(self, x):
        for i in range(self.n_layers - 1):
            x = self.act(self.map[i](x))
        return self.map[-1](x)


class TransformerBlock(nn.Module):
    """Post-norm transformer block.

    NOTE: as in the paper experiments, attention scores are scaled by sqrt(sequence length) rather than
    sqrt(head dim), and the same LayerNorm is shared by both residual branches.
    """

    def __init__(self, dim, ffn_dim, n_attn_heads, dropout):
        assert dim % n_attn_heads == 0, f"dim {dim} cannot be divided by n_attn_heads {n_attn_heads}"
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.dim = dim
        self.n_attn_heads = n_attn_heads

        self.norm = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.softmax = nn.Softmax(dim=-1)
        self.linear = nn.Linear(dim, dim)
        self.ffn = MLPBlock(in_dim=dim, h_dim=ffn_dim, out_dim=dim)

    def forward(self, x, attn_mask=None):
        b, l, _ = x.shape
        head_dim = self.dim // self.n_attn_heads
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.reshape(b, l, self.n_attn_heads, head_dim).transpose(1, 2)
        k = k.reshape(b, l, self.n_attn_heads, head_dim).transpose(1, 2)
        v = v.reshape(b, l, self.n_attn_heads, head_dim).transpose(1, 2)
        attn_scores = q @ k.transpose(2, 3) / math.sqrt(l)
        if attn_mask is not None:
            attn_scores = attn_scores.masked_fill((attn_mask == 0).unsqueeze(1), -torch.inf)
        attn_weight = self.softmax(attn_scores).nan_to_num(0.0)
        attn_out = (attn_weight @ v).transpose(1, 2).reshape(b, l, self.dim)
        attn_out = self.linear(attn_out)

        x = self.norm(x + self.dropout(attn_out))
        x = self.norm(x + self.dropout(self.ffn(x)))
        return x


class SinusoidalEncoder(nn.Module):
    def __init__(self, max_len, embedding_dim, n=10000.0):
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * -(math.log(n) / embedding_dim))
        pe = torch.zeros(max_len, embedding_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        b, l, _ = x.shape
        return self.pe[:l].unsqueeze(0).expand(b, -1, -1)


class TransformerAE(nn.Module):
    """Transformer auto-encoder over action chunks.

    With ``n_reg == 0`` every timestep gets its own latent token (decoded by ``ar_decode``); with
    ``n_reg > 0`` the sequence is summarized into ``n_reg`` register tokens (decoded by ``masked_decode``).

    NOTE: both decoders run through ``enc_blocks``, i.e. encoder and decoder share their transformer
    weights. ``dec_proj``/``dec_blocks`` are allocated but unused; they are kept so that parameter
    initialization and checkpoint keys match the models reported in the paper.
    """

    def __init__(
        self,
        in_dim,
        z_dim,
        n_reg,
        enc_h_dim,
        enc_n_heads,
        enc_n_blocks,
        dec_h_dim,
        dec_n_heads,
        dec_n_blocks,
        dropout=0.1,
        causal_attn=False,
        max_seq_len=100,
    ):
        super().__init__()
        if isinstance(in_dim, int):
            in_dim = [in_dim]
        assert isinstance(in_dim, (list, tuple)), "Wrong in_dim type"
        self.causal_attn = causal_attn
        self.max_seq_len = max_seq_len

        self.n_reg = n_reg
        self.reg_emb = nn.Parameter(torch.zeros(1, self.n_reg, enc_h_dim), requires_grad=True)
        self.mask_emb = nn.Parameter(torch.zeros(1, 1, enc_h_dim), requires_grad=True)
        init_v = 1 / enc_h_dim**0.5
        nn.init.uniform_(self.reg_emb, a=-init_v, b=init_v)
        nn.init.uniform_(self.mask_emb, a=-init_v, b=init_v)

        self.enc_proj = nn.Linear(sum(in_dim), enc_h_dim)
        self.enc_pos_emb = SinusoidalEncoder(self.max_seq_len, enc_h_dim)
        self.enc_blocks = nn.ModuleList(
            [TransformerBlock(enc_h_dim, enc_h_dim, enc_n_heads, dropout) for _ in range(enc_n_blocks)]
        )
        self.z_dim = enc_h_dim if z_dim == 0 else z_dim
        self.latent_proj = nn.Linear(enc_h_dim, self.z_dim)
        self.latent_unproj = nn.Linear(self.z_dim, dec_h_dim)
        self.dec_proj = nn.Linear(sum(in_dim), enc_h_dim)  # unused, see class docstring
        self.dec_pos_emb = SinusoidalEncoder(self.max_seq_len, dec_h_dim)
        self.dec_blocks = nn.ModuleList(  # unused, see class docstring
            [TransformerBlock(dec_h_dim, dec_h_dim, dec_n_heads, dropout) for _ in range(dec_n_blocks)]
        )
        self.pred_heads = nn.ModuleList([MLPBlock(in_dim=dec_h_dim, h_dim=dec_h_dim, out_dim=d) for d in in_dim])
        self.dropout = nn.Dropout(dropout)

    def _attn_mask(self, batch_size, seq_len, device):
        if self.causal_attn:
            mask = generate_causal_mask(seq_len)
        else:
            mask = torch.ones(seq_len, seq_len)
        return mask.to(device).unsqueeze(0).repeat(batch_size, 1, 1)

    def _run_blocks(self, h):
        attn_mask = self._attn_mask(h.shape[0], h.shape[1], h.device)
        h = self.dropout(h)
        for layer in self.enc_blocks:
            h = layer(h, attn_mask=attn_mask)
        return torch.cat([pred_head(h) for pred_head in self.pred_heads], -1)

    def encode(self, x):
        x_emb = self.enc_proj(x)
        if self.n_reg > 0:
            x_emb = torch.cat([self.reg_emb.expand(x.shape[0], -1, -1), x_emb], 1)
        h = x_emb + self.enc_pos_emb(x_emb)

        attn_mask = self._attn_mask(h.shape[0], h.shape[1], h.device)
        h = self.dropout(h)
        for layer in self.enc_blocks:
            h = layer(h, attn_mask=attn_mask)
        if self.n_reg > 0:
            return self.latent_proj(h[:, : self.n_reg, :])
        return self.latent_proj(h)

    def masked_decode(self, latent, seq_len):
        latent_emb = self.latent_unproj(latent)
        mask_emb = self.mask_emb.expand(latent.shape[0], seq_len, -1)
        latent_emb = torch.cat([latent_emb, mask_emb], 1)
        out = self._run_blocks(latent_emb + self.dec_pos_emb(latent_emb))
        return out[:, -seq_len:]

    def ar_decode(self, latent):
        latent_emb = self.latent_unproj(latent)
        return self._run_blocks(latent_emb + self.dec_pos_emb(latent_emb))


class TransformerVQVAE(TransformerAE):
    def __init__(
        self,
        k_dim,
        z_dim,
        in_dim,
        n_reg,
        h_dim,
        n_heads,
        n_blocks,
        dropout=0.05,
        codebook_h_dim=256,
        codebook_n_hiddens=3,
        max_seq_len=100,
        quantizer_type="vqvae",
    ):
        if z_dim is None:
            z_dim = h_dim
        super().__init__(
            in_dim=in_dim,
            z_dim=z_dim,
            n_reg=n_reg,
            enc_h_dim=h_dim,
            enc_n_heads=n_heads,
            enc_n_blocks=n_blocks,
            dec_h_dim=h_dim,
            dec_n_heads=n_heads,
            dec_n_blocks=n_blocks,
            dropout=dropout,
            causal_attn=False,
            max_seq_len=max_seq_len,
        )
        self.k_dim = k_dim
        self.quantizer_type = quantizer_type
        if self.quantizer_type == "flow":
            self.codebook = FlowQuantizer(self.k_dim, self.z_dim, codebook_h_dim, codebook_n_hiddens)
        else:
            self.codebook = VectorQuantizer(self.k_dim, self.z_dim)

    def forward(self, x, decode=True):
        """Returns ``(z_e, z_q, zq_inds), recon`` if ``decode`` else ``(z_e, z_q, zq_inds)``."""
        L = x.shape[1]
        z_e = self.encode(x)
        zq_inds, z_q = self.codebook.quantize(z_e)
        if not decode:
            return z_e, z_q, zq_inds

        latent = z_e + (z_q - z_e).detach()  # straight-through estimator
        if self.n_reg == 0:
            out = self.ar_decode(latent)
        else:
            out = self.masked_decode(latent, seq_len=L)
        return (z_e, z_q, zq_inds), out
