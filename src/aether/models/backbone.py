"""Bidirectional transformer backbone with AdaLN timestep conditioning.

The denoiser reads the whole (partially masked) sequence at once, so attention is
bidirectional (no causal mask). Diffusion time ``t`` is injected via adaptive
layer norm (AdaLN-Zero, as in DiT): a timestep embedding produces per-block
shift/scale/gate parameters that modulate each sub-layer. AdaLN-Zero initializes
the modulation to identity so training starts from a stable near-residual network.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def timestep_embedding(t: Tensor, dim: int, max_period: float = 10_000.0) -> Tensor:
    """Sinusoidal embedding of continuous time ``t in [0, 1]`` -> ``(B, dim)``."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(half, dtype=torch.float32, device=t.device)
        / max(half, 1)
    )
    args = t.float()[:, None] * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


def modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    """AdaLN modulation: ``x * (1 + scale) + shift`` broadcast over sequence."""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbedder(nn.Module):
    """Embeds scalar diffusion time into a conditioning vector."""

    def __init__(self, hidden_size: int, freq_dim: int = 256) -> None:
        super().__init__()
        self.freq_dim = freq_dim
        self.mlp = nn.Sequential(
            nn.Linear(freq_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, t: Tensor) -> Tensor:
        out: Tensor = self.mlp(timestep_embedding(t, self.freq_dim))
        return out


class Attention(nn.Module):
    """Multi-head bidirectional self-attention (no causal mask)."""

    def __init__(self, dim: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        if dim % n_heads:
            raise ValueError(f"d_model {dim} not divisible by n_heads {n_heads}")
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.dropout = dropout
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: Tensor) -> Tensor:
        b, length, dim = x.shape
        qkv = self.qkv(x).reshape(b, length, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        dropout_p = self.dropout if self.training else 0.0
        attn = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p, is_causal=False)
        attn = attn.transpose(1, 2).reshape(b, length, dim)
        out: Tensor = self.proj(attn)
        return out


class FeedForward(nn.Module):
    """Position-wise MLP."""

    def __init__(self, dim: int, mult: int, dropout: float) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, mult * dim)
        self.fc2 = nn.Linear(mult * dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        out: Tensor = self.fc2(self.drop(F.gelu(self.fc1(x))))
        return out


class DiTBlock(nn.Module):
    """Transformer block with AdaLN-Zero conditioning on the time embedding."""

    def __init__(self, dim: int, n_heads: int, dropout: float, mlp_mult: int = 4) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(dim, n_heads, dropout)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.mlp = FeedForward(dim, mlp_mult, dropout)
        self.adaln = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))

    def forward(self, x: Tensor, c: Tensor) -> Tensor:
        mod: Tensor = self.adaln(c)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod.chunk(6, dim=-1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x
