"""Publish a trained checkpoint and its model card to the Hugging Face Hub.

    python scripts/publish_hf.py \
        --checkpoint runs/aether-55m/checkpoints/latest.pt \
        --repo ameyg910/aether-55m \
        --eval benchmarks/results/final-30k.json \
        --tag v1.0.0

The card is generated from the checkpoint and the evaluation report rather than
written by hand, so the published numbers cannot drift from the ones actually
measured. A card claiming a perplexity nobody can reproduce is worse than no card.

``--tag`` creates a Hub *revision*, which is immutable. That is what makes
``hf:owner/repo@v1.0.0`` a reproducible deploy target and a rollback that means
something -- pointing at ``main`` gives you whatever happens to be there today.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from aether.models.loading import build_model_from_checkpoint


def _fmt(value: Any, spec: str = ".4g") -> str:
    if value is None:
        return "n/a"
    if isinstance(value, int | float):
        return f"{value:{spec}}"
    return str(value)


def build_card(
    repo: str,
    checkpoint: Path,
    ckpt: dict[str, Any],
    config: Any,
    params: int,
    evaluation: dict[str, Any] | None,
    dataset: str,
    tag: str | None,
) -> str:
    """Render the model card from measured facts."""
    nll = (evaluation or {}).get("nll", {})
    diversity = (evaluation or {}).get("diversity", {})
    mauve = (evaluation or {}).get("mauve")
    eval_cfg = (evaluation or {}).get("config", {})
    step = ckpt.get("step")
    tokens = None
    if step:
        # Effective batch during the reference run; recorded here so the token
        # count is traceable rather than asserted.
        tokens = step * 128 * config.max_seq_len

    version = tag or "main"
    return f"""---
license: apache-2.0
library_name: aether
tags:
  - diffusion
  - masked-diffusion
  - language-model
  - mdlm
datasets:
  - {dataset}
pipeline_tag: text-generation
---

# {repo}

A masked (absorbing-state) diffusion language model trained with
[Aether](https://github.com/ameyg910/aether).

Unlike an autoregressive model, generation does not proceed left to right. The
model starts from an all-`[MASK]` sequence and unmasks progressively, so the
number of forward passes (**NFE**) is a knob rather than a function of sequence
length.

## Model details

| | |
| --- | --- |
| Architecture | bidirectional DiT denoiser, AdaLN-Zero time conditioning |
| Objective | MDLM / SUBS masked-diffusion loss |
| Parameters | {params:,} |
| Width / depth / heads | {config.d_model} / {config.n_layers} / {config.n_heads} |
| Context length | {config.max_seq_len} |
| Vocabulary | {config.vocab_size:,} (GPT-2 + `[MASK]`) |
| Training steps | {_fmt(step, ",")} |
| Tokens seen | {_fmt(tokens, ",")} |
| License | Apache-2.0 |

## Evaluation

Measured with `aether-eval`; see
[the evaluation protocol](https://github.com/ameyg910/aether/blob/main/docs/evaluation.md).

| metric | value |
| --- | --- |
| NELBO (nats/token) | {_fmt(nll.get("nats_per_token"))} |
| Bits per dim | {_fmt(nll.get("bits_per_dim"))} |
| Perplexity (upper bound) | {_fmt(nll.get("perplexity"))} |
| MAUVE | {_fmt(mauve)} |
| distinct-2 | {_fmt(diversity.get("distinct_2"))} |
| Sampler / steps | {eval_cfg.get("sampler", "n/a")} / {eval_cfg.get("steps", "n/a")} |

> **Perplexity here is an upper bound, not an exact likelihood.** A masked
> diffusion model has no exact factorization of `log p(x)`; what is reported is a
> Monte Carlo estimate of a variational bound. It is comparable to other diffusion
> models evaluated the same way, and **not** directly comparable to an
> autoregressive model's exact perplexity, which would flatter the AR model.

## Usage

Serve it:

```bash
pip install "aether-dlm[serve]"
aether-serve serve.model_version=hf:{repo}@{version}

curl -X POST localhost:8000/generate \\
  -H 'content-type: application/json' \\
  -d '{{"n_samples":2,"length":64,"steps":64,"sampler":"ancestral"}}'
```

Or load it directly:

```python
import torch
from huggingface_hub import hf_hub_download
from aether.models.loading import build_model_from_checkpoint
from aether.diffusion.samplers import sample

path = hf_hub_download("{repo}", "latest.pt", revision="{version}")
model, config = build_model_from_checkpoint(
    torch.load(path, map_location="cpu", weights_only=False)
)
out = sample(model.eval(), batch=2, length=64,
             mask_token_id=config.vocab_size - 1, steps=64)
print(out.tokens.shape, "NFE:", out.nfe)
```

## Intended use

Research and demonstration of masked diffusion language modelling: studying the
NFE-quality tradeoff, comparing sampling strategies, and as a fixture for
inference-serving work.

## Limitations

- **Small and undertrained.** {params:,} parameters and roughly
  {_fmt(tokens, ",")} tokens. It captures vocabulary and local phrasing, not
  long-range coherence or factual grounding.
- **Unconditional.** There is no prompt input; it generates from an all-`[MASK]`
  sequence. Prompt-conditioned infilling is a natural extension the architecture
  supports but this release does not implement.
- **No alignment of any kind.** No instruction tuning, no safety filtering, no
  RLHF. Output may be offensive, false, or nonsensical.
- **Inherits its corpus.** Trained on {dataset}, and reproduces the biases and
  errors in it.

Not suitable for production text generation, question answering, or any use where
output correctness matters.

## Citation

```bibtex
@software{{aether,
  author = {{Gupta, Amey}},
  title  = {{Aether: a production platform for masked diffusion language models}},
  year   = {{2026}},
  url    = {{https://github.com/ameyg910/aether}}
}}
```
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--repo", required=True, help="e.g. ameyg910/aether-55m")
    ap.add_argument("--eval", type=Path, default=None, help="aether-eval report JSON")
    ap.add_argument("--tag", default=None, help="create an immutable Hub revision")
    ap.add_argument("--dataset", default="wikitext")
    ap.add_argument("--filename", default="latest.pt", help="name on the Hub")
    ap.add_argument("--private", action="store_true")
    ap.add_argument(
        "--card-only",
        action="store_true",
        help="render the card to stdout without uploading (use to preview)",
    )
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model, config = build_model_from_checkpoint(ckpt)
    params = sum(p.numel() for p in model.parameters())

    evaluation = json.loads(args.eval.read_text()) if args.eval else None
    if evaluation is None:
        print("warning: no --eval report; the card's metric table will read 'n/a'")

    card = build_card(
        repo=args.repo,
        checkpoint=args.checkpoint,
        ckpt=ckpt,
        config=config,
        params=params,
        evaluation=evaluation,
        dataset=args.dataset,
        tag=args.tag,
    )

    if args.card_only:
        print(card)
        return

    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(args.repo, repo_type="model", private=args.private, exist_ok=True)

    card_path = Path("/tmp/README.md")
    card_path.write_text(card)
    api.upload_file(
        path_or_fileobj=str(card_path),
        path_in_repo="README.md",
        repo_id=args.repo,
        repo_type="model",
        commit_message="docs: model card",
    )
    api.upload_file(
        path_or_fileobj=str(args.checkpoint),
        path_in_repo=args.filename,
        repo_id=args.repo,
        repo_type="model",
        commit_message=f"model: {args.checkpoint.name}",
    )
    print(f"uploaded to https://huggingface.co/{args.repo}")

    if args.tag:
        # Immutable: this is what makes hf:repo@tag a reproducible deploy target.
        api.create_tag(args.repo, tag=args.tag, repo_type="model", exist_ok=True)
        print(f"tagged {args.tag} -> serve with hf:{args.repo}@{args.tag}")


if __name__ == "__main__":
    main()
