"""Exponential moving average of model parameters.

EMA maintains a smoothed copy of the weights (``shadow = decay*shadow +
(1-decay)*param``). For diffusion models the EMA weights consistently produce
better samples than the raw training weights, so sampling and evaluation use them.
The shadow is kept in fp32 for numerical stability regardless of training dtype.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn


class EMA:
    """Tracks an exponential moving average of a model's trainable parameters."""

    def __init__(self, model: nn.Module, decay: float) -> None:
        self.decay = decay
        self.shadow: dict[str, Tensor] = {
            name: p.detach().clone().float()
            for name, p in model.named_parameters()
            if p.requires_grad
        }
        self._backup: dict[str, Tensor] = {}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for name, p in model.named_parameters():
            if p.requires_grad and name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(p.detach().float(), alpha=1.0 - self.decay)

    @torch.no_grad()
    def copy_to(self, model: nn.Module) -> None:
        """Overwrite the model's parameters with the EMA weights (for sampling)."""
        for name, p in model.named_parameters():
            if name in self.shadow:
                p.data.copy_(self.shadow[name].to(p.dtype))

    @torch.no_grad()
    def store(self, model: nn.Module) -> None:
        self._backup = {
            name: p.detach().clone() for name, p in model.named_parameters() if name in self.shadow
        }

    @torch.no_grad()
    def restore(self, model: nn.Module) -> None:
        for name, p in model.named_parameters():
            if name in self._backup:
                p.data.copy_(self._backup[name])
        self._backup = {}

    def state_dict(self) -> dict[str, Any]:
        return {"decay": self.decay, "shadow": self.shadow}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.decay = state["decay"]
        self.shadow = {k: v.clone() for k, v in state["shadow"].items()}
