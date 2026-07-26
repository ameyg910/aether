"""The Aether denoiser: a masked diffusion language model.

Maps a partially masked token sequence ``x_t`` and diffusion time ``t`` to logits
over the vocabulary. The absorbing ``[MASK]`` token is part of the input vocab;
the SUBS parameterization that forbids predicting it lives in the loss, so the
model itself is a plain bidirectional classifier over clean tokens.
"""

from __future__ import annotations

import structlog
import torch
from torch import Tensor, nn

from aether.config.schemas import ModelConfig
from aether.models.backbone import DiTBlock, TimestepEmbedder, modulate

logger = structlog.get_logger()


def _zero_linear(module: nn.Module) -> None:
    """Zero the weight and bias of a Linear layer (for AdaLN-Zero init)."""
    assert isinstance(module, nn.Linear)
    nn.init.zeros_(module.weight)
    nn.init.zeros_(module.bias)


class AetherModel(nn.Module):
    """Bidirectional, time-conditioned transformer denoiser."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, cfg.max_seq_len, cfg.d_model))
        self.t_embedder = TimestepEmbedder(cfg.d_model)
        self.blocks = nn.ModuleList(
            DiTBlock(cfg.d_model, cfg.n_heads, cfg.dropout) for _ in range(cfg.n_layers)
        )
        self.norm_final = nn.LayerNorm(cfg.d_model, elementwise_affine=False, eps=1e-6)
        self.adaln_final = nn.Sequential(nn.SiLU(), nn.Linear(cfg.d_model, 2 * cfg.d_model))
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size)
        self._init_weights()
        logger.info(
            "model_init",
            params=self.num_params,
            d_model=cfg.d_model,
            n_layers=cfg.n_layers,
            n_heads=cfg.n_heads,
            vocab_size=cfg.vocab_size,
            flops_per_token=self.flops_per_token,
        )

    def _init_weights(self) -> None:
        nn.init.normal_(self.tok_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb, std=0.02)
        nn.init.normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)
        # AdaLN-Zero: zero the modulation outputs so blocks start as identity.
        for block in self.blocks:
            assert isinstance(block, DiTBlock)
            _zero_linear(block.adaln[-1])
        _zero_linear(self.adaln_final[-1])

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @property
    def flops_per_token(self) -> int:
        # Rough forward-pass estimate: ~2 FLOPs per parameter per token.
        return 2 * self.num_params

    def forward(self, x_t: Tensor, t: Tensor) -> Tensor:
        """Args: ``x_t`` ids ``(B, L)``, ``t`` times ``(B,)``. Returns logits ``(B, L, V)``."""
        length = x_t.shape[1]
        h = self.tok_emb(x_t) + self.pos_emb[:, :length]
        c = self.t_embedder(t)
        for block in self.blocks:
            h = block(h, c)
        shift, scale = self.adaln_final(c).chunk(2, dim=-1)
        h = modulate(self.norm_final(h), shift, scale)
        logits: Tensor = self.head(h)
        return logits
