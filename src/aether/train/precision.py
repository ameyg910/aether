"""Device and mixed-precision resolution.

Extracted from the trainer (Engineering Review #1, refactor 1) so the training
loop, evaluation, and sampling all resolve devices and autocast the same way.

bf16 is the default on Ampere and newer: it has the same exponent range as fp32,
so it does not need loss scaling and cannot silently overflow the way fp16 does.
fp16 keeps more mantissa bits but a much smaller range, so it requires a
``GradScaler``; the scaler here is therefore enabled *only* for fp16.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

# Import from the concrete module rather than the ``torch.amp`` package: older
# torch releases do not re-export ``GradScaler`` in ``torch/amp/__init__.pyi``,
# so ``torch.amp.GradScaler`` fails type-checking there while this path works on
# every version that has AMP at all.
from torch.amp.grad_scaler import GradScaler

_AMP_DTYPE: dict[str, torch.dtype] = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}


def resolve_device(spec: str, local_rank: int = 0) -> torch.device:
    """Resolve a device spec to a concrete device.

    ``"auto"`` picks CUDA when available. Under distributed training each process
    owns exactly one GPU, selected by ``local_rank``.
    """
    if spec == "auto":
        if torch.cuda.is_available():
            return torch.device(f"cuda:{local_rank}")
        return torch.device("cpu")
    if spec == "cuda" and torch.cuda.is_available():
        return torch.device(f"cuda:{local_rank}")
    return torch.device(spec)


def amp_dtype(precision: str) -> torch.dtype:
    if precision not in _AMP_DTYPE:
        raise ValueError(f"Unknown precision {precision!r}; expected one of {sorted(_AMP_DTYPE)}")
    return _AMP_DTYPE[precision]


@dataclass(frozen=True)
class PrecisionPlan:
    """Everything the loop needs to run one precision consistently."""

    precision: str
    dtype: torch.dtype
    autocast_enabled: bool
    needs_scaler: bool

    @classmethod
    def from_spec(cls, precision: str) -> PrecisionPlan:
        dtype = amp_dtype(precision)
        return cls(
            precision=precision,
            dtype=dtype,
            autocast_enabled=precision in ("bf16", "fp16"),
            needs_scaler=precision == "fp16",
        )

    def scaler(self, device: torch.device) -> GradScaler:
        return GradScaler(device.type, enabled=self.needs_scaler)

    def autocast(self, device: torch.device) -> torch.autocast:
        return torch.autocast(
            device_type=device.type, dtype=self.dtype, enabled=self.autocast_enabled
        )
