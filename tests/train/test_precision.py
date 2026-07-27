"""Device and precision resolution."""

from __future__ import annotations

import pytest
import torch

from aether.train.precision import PrecisionPlan, amp_dtype, resolve_device


def test_amp_dtype_mapping() -> None:
    assert amp_dtype("bf16") is torch.bfloat16
    assert amp_dtype("fp16") is torch.float16
    assert amp_dtype("fp32") is torch.float32
    with pytest.raises(ValueError, match="Unknown precision"):
        amp_dtype("int4")


def test_only_fp16_needs_a_gradient_scaler() -> None:
    # bf16 has fp32's exponent range, so gradients cannot underflow the way fp16's do.
    assert PrecisionPlan.from_spec("fp16").needs_scaler is True
    assert PrecisionPlan.from_spec("bf16").needs_scaler is False
    assert PrecisionPlan.from_spec("fp32").needs_scaler is False


def test_autocast_disabled_for_fp32() -> None:
    assert PrecisionPlan.from_spec("fp32").autocast_enabled is False
    assert PrecisionPlan.from_spec("bf16").autocast_enabled is True


def test_resolve_device_explicit_cpu() -> None:
    assert resolve_device("cpu").type == "cpu"


def test_resolve_device_auto_without_cuda() -> None:
    dev = resolve_device("auto", local_rank=0)
    assert dev.type in ("cpu", "cuda")
