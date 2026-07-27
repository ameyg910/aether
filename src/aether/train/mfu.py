"""Model FLOPs Utilization (MFU) and throughput accounting.

MFU is the standard efficiency metric for training runs: the ratio of FLOPs the
model *usefully* performs to the FLOPs the hardware could theoretically retire in
the same wall-clock time. It answers "how much of the GPU am I actually using?"
in a way that is comparable across models and clusters, unlike raw tokens/sec.

The FLOP count follows the PaLM appendix / nanoGPT convention::

    flops_per_token = 6 * N + 12 * n_layers * d_model * seq_len

``6N`` covers the parameter matmuls: 2 FLOPs per parameter per token in the
forward pass, and roughly twice that again in the backward pass (one matmul for
the input gradient and one for the weight gradient). The second term is
attention, whose cost grows with sequence length rather than parameter count --
each layer forms an ``L x L`` score matrix and applies it to the values.
Aether's denoiser is bidirectional, so the full score matrix is computed (there
is no causal mask halving the work).

Peak numbers below are *dense* (non-sparse) bf16/fp16 tensor-core throughput, in
TFLOP/s per GPU, from the vendor datasheets. Reporting against the sparse figure
would roughly halve the apparent MFU for no good reason.
"""

from __future__ import annotations

# Dense bf16/fp16 tensor-core peak, TFLOP/s per GPU.
DEVICE_PEAK_TFLOPS: dict[str, float] = {
    "a6000": 154.8,
    "a100": 312.0,
    "h100 sxm": 989.0,
    "h100 pcie": 756.0,
    "h100 nvl": 835.0,
    "h100": 989.0,
    "l40s": 362.0,
    "v100": 125.0,
    "4090": 165.2,
}


def training_flops_per_token(num_params: int, n_layers: int, d_model: int, seq_len: int) -> int:
    """FLOPs performed per token for one forward+backward pass."""
    return 6 * num_params + 12 * n_layers * d_model * seq_len


def lookup_peak_tflops(device_name: str) -> float | None:
    """Best-effort match of a CUDA device name to a peak throughput figure.

    Returns ``None`` when the device is unknown, in which case MFU is not
    reported rather than being reported wrongly. Longer keys are tried first so
    that e.g. ``"h100 sxm"`` wins over the generic ``"h100"``.
    """
    name = device_name.lower()
    for key in sorted(DEVICE_PEAK_TFLOPS, key=len, reverse=True):
        if key in name:
            return DEVICE_PEAK_TFLOPS[key]
    return None


def compute_mfu(flops_per_token: int, tokens_per_sec: float, peak_tflops: float) -> float:
    """Fraction of theoretical peak FLOPs actually used (0.0-1.0)."""
    if peak_tflops <= 0:
        return 0.0
    achieved = flops_per_token * tokens_per_sec
    return achieved / (peak_tflops * 1e12)


class ThroughputMeter:
    """Tracks tokens/sec and MFU across a training run.

    ``world_size`` scales the peak so that MFU stays comparable as GPUs are added:
    a perfectly-scaling run holds MFU flat while tokens/sec rises linearly.
    """

    def __init__(
        self,
        flops_per_token: int,
        peak_tflops: float | None,
        world_size: int = 1,
    ) -> None:
        self.flops_per_token = flops_per_token
        self.peak_tflops = peak_tflops
        self.world_size = world_size

    def metrics(self, tokens: int, seconds: float) -> dict[str, float]:
        """Throughput metrics for ``tokens`` processed across all ranks."""
        if seconds <= 0:
            return {}
        tokens_per_sec = tokens / seconds
        out = {"tokens_per_sec": tokens_per_sec}
        if self.peak_tflops is not None:
            cluster_peak = self.peak_tflops * self.world_size
            out["mfu"] = compute_mfu(self.flops_per_token, tokens_per_sec, cluster_peak)
        return out
