"""FLOP accounting and MFU arithmetic."""

from __future__ import annotations

from aether.train.mfu import (
    ThroughputMeter,
    compute_mfu,
    lookup_peak_tflops,
    training_flops_per_token,
)


def test_flops_per_token_matches_formula() -> None:
    # 6N dominates; the attention term scales with sequence length.
    flops = training_flops_per_token(num_params=1_000_000, n_layers=4, d_model=256, seq_len=512)
    assert flops == 6 * 1_000_000 + 12 * 4 * 256 * 512


def test_flops_grow_with_sequence_length() -> None:
    short = training_flops_per_token(1_000_000, 4, 256, 128)
    long = training_flops_per_token(1_000_000, 4, 256, 1024)
    assert long > short


def test_lookup_prefers_the_most_specific_device() -> None:
    assert lookup_peak_tflops("NVIDIA H100 PCIe") == 756.0
    assert lookup_peak_tflops("NVIDIA RTX A6000") == 154.8
    assert lookup_peak_tflops("Some Unreleased GPU") is None


def test_compute_mfu_is_a_fraction_of_peak() -> None:
    # 1e12 FLOPs/s achieved against a 2 TFLOP/s peak == 50%.
    mfu = compute_mfu(flops_per_token=1_000, tokens_per_sec=1e9, peak_tflops=2.0)
    assert abs(mfu - 0.5) < 1e-9


def test_meter_reports_throughput_and_scales_peak_by_world_size() -> None:
    meter = ThroughputMeter(flops_per_token=1_000, peak_tflops=1.0, world_size=4)
    metrics = meter.metrics(tokens=2_000, seconds=2.0)
    assert metrics["tokens_per_sec"] == 1_000.0
    # peak is 4 x 1 TFLOP/s, so MFU = (1000 * 1000) / 4e12
    assert abs(metrics["mfu"] - (1_000 * 1_000) / 4e12) < 1e-15


def test_meter_omits_mfu_when_peak_unknown() -> None:
    meter = ThroughputMeter(flops_per_token=1_000, peak_tflops=None)
    metrics = meter.metrics(tokens=100, seconds=1.0)
    assert "tokens_per_sec" in metrics
    assert "mfu" not in metrics


def test_meter_handles_zero_elapsed_time() -> None:
    meter = ThroughputMeter(flops_per_token=10, peak_tflops=1.0)
    assert meter.metrics(tokens=10, seconds=0.0) == {}
