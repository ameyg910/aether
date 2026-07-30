# Roadmap

Aether reached v1.0.0 after ten weeks: a trained model, an evaluation harness, a
served API, containers, Kubernetes, and a public release. This is what is
plausibly next, roughly in order of value per unit of effort.

Items tagged **`good first issue`** are self-contained, have a clear definition of
done, and do not require a GPU. See [CONTRIBUTING.md](CONTRIBUTING.md).

## Next

### Prompt-conditioned generation (infilling)
The model is currently **unconditional** — it generates from an all-`[MASK]`
sequence with no prompt input. But the architecture is *natively* an infiller:
seed some positions with real tokens, pin them so the sampler never overwrites
them, and the model fills the rest around them, bidirectionally. That is
something an autoregressive model structurally cannot do, and it is the single
biggest gap between what this demonstrates and what the architecture is for.
Roughly a 15-line change to `iter_denoise` plus a `prompt` field on
`GenerateRequest`.

### KV-cache reuse for block/semi-autoregressive decoding
Generate in blocks rather than over the whole sequence at once, so attention
state can be cached across denoising steps. Block Diffusion shows this enables
variable-length generation and large speedups. Currently every step recomputes
attention over the full sequence.

### Quantized serving
int8 weight-only quantization plus `torch.compile` on the served model, with a
before/after latency and quality table. The `log_softmax` over a 50k vocabulary
dominates the step; quantization should help throughput more than it hurts a
model this small.

### Multi-node training
FSDP works in principle but has never run on hardware — no CI coverage (it
cannot initialize on CPU) and no multi-GPU access during development. Validating
it on a real cluster and reporting a scaling curve to 8+ GPUs is the honest
completion of Week 5.

## Smaller, well-scoped

- **`good first issue`** — **Top-k / top-p sampling.** The samplers support
  temperature only. Add nucleus sampling to `_masked_probs` with tests covering
  the degenerate cases (`k=1`, `p=1.0`).
- **`good first issue`** — **`aether-eval` comparison mode.** Accept several
  checkpoints and emit one table, so runs can be compared without diffing JSON.
- **`good first issue`** — **Cosine schedule benchmark.** The cosine noise
  schedule is implemented and untested against linear on a real model. Run the
  NFE sweep for both and add the comparison to `docs/evaluation.md`.
- **`good first issue`** — **Dockerfile image-size pass.** The serve image is
  larger than it needs to be. Measure, trim, and document the result.
- **`good first issue`** — **Structured request IDs.** Thread a request id
  through the batcher into the logs so one slow request can be traced from
  arrival to response.

## Deliberately not planned

- **Scaling the model.** A larger model is a compute problem, not an engineering
  one, and this project is about the platform.
- **Instruction tuning / RLHF.** Out of scope for a base-model platform, and
  doing it responsibly needs evaluation infrastructure this project does not have.
- **Custom CUDA kernels.** Interesting, but MFU is 29.7% — the wins available
  from batching, caching, and quantization are larger and cheaper first.
