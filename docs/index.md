# Aether

A production platform for **masked (absorbing-state) diffusion language models** —
the MDLM/SUBS formulation that LLaDA and Dream scaled to challenge autoregressive
LLMs.

Built over ten weeks from a typed research skeleton into a trained, evaluated,
served, containerized, orchestrated, and released framework.

[Try the demo](https://huggingface.co/spaces/ameyg910/aether-demo){ .md-button .md-button--primary }
[GitHub](https://github.com/ameyg910/aether){ .md-button }

## What it does

Generation runs the masking process backwards: start from an all-`[MASK]`
sequence and unmask progressively. Unlike an autoregressive model, which must
spend one forward pass per token, a diffusion model **chooses** how much compute
to spend on a sequence of any length. That knob — NFE, the number of function
evaluations — is what makes these models interesting, and measuring it honestly
is a large part of what this project is about.

## The platform

| layer | what is there |
| --- | --- |
| [Data](data.md) | GPT-2 and byte tokenizers, packing, sharded `uint16` memmaps with a content fingerprint |
| [Training](training.md) | bf16 AMP, gradient accumulation, EMA, resumable checkpointing, experiment tracking |
| [Distributed](cluster.md) | DDP and FSDP behind one flag, world-size-independent checkpoints, MFU reporting |
| [Evaluation](evaluation.md) | NELBO / bits-per-dim, MAUVE, diversity, an NFE-quality sweep, a CI regression benchmark |
| [Serving](serving.md) | FastAPI with dynamic batching, SSE streaming, a versioned model registry, Prometheus metrics |
| [Deployment](deployment.md) | Multi-stage images, a Helm chart, autoscaling on queue depth, dashboards and alerts as code |

## Measured results

A 55.5M-parameter model trained on WikiText-103 for 30k steps on one RTX A6000:

| | |
| --- | --- |
| Likelihood bound | 7.14 nats/token, 10.30 bits-per-dim |
| Training efficiency | 29.7% MFU, single GPU |
| Serving throughput | ~15 req/s at 16 concurrent users, p50 660 ms |
| Batching gain | ~12x over serialized serving |

The model is small and undertrained by design — the point of the project is the
platform around it. See [evaluation](evaluation.md) for what those numbers mean
and why perplexity is subtler for a diffusion LM than for an AR model.

## Start here

- [Quickstart](quickstart.md) — running in five minutes
- [Architecture](architecture.md) — how the pieces fit together
- [Final review](reviews/review-final.md) — an honest account of what is solid,
  what is not, and what comes next

