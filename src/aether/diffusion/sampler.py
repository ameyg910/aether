"""Backwards-compatible re-exports; the implementations live in ``samplers``.

Week 3 shipped a single ancestral sampler here. Week 6 added a second strategy
and NFE accounting, which moved the implementations to
:mod:`aether.diffusion.samplers`. This module stays so existing imports keep
working.
"""

from __future__ import annotations

from aether.diffusion.samplers import (
    SamplerOutput,
    ancestral_sample,
    ancestral_sample_full,
    confidence_sample_full,
    sample,
)

__all__ = [
    "SamplerOutput",
    "ancestral_sample",
    "ancestral_sample_full",
    "confidence_sample_full",
    "sample",
]
