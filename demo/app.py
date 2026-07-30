"""Gradio demo for Aether — a masked diffusion language model.

Deployed as a Hugging Face Space. The interesting thing to show is not the text
(a 55M model trained for 30k steps produces vocabulary, not prose) but the
*process*: the sequence resolving out of noise, and the compute/quality knob that
autoregressive models do not have.

Run locally:

    pip install gradio
    python demo/app.py
"""

from __future__ import annotations

import os
import time

import gradio as gr
import torch

from aether.diffusion.samplers import iter_denoise
from aether.models.loading import build_model_from_checkpoint

REPO = os.environ.get("AETHER_REPO", "ameyg910/aether-55m")
REVISION = os.environ.get("AETHER_REVISION", "v1.0.0")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_model = None
_config = None
_tokenizer = None


def load() -> tuple[object, object, object]:
    """Load once, lazily, and cache. Spaces restart cold; this keeps it to one hit."""
    global _model, _config, _tokenizer
    if _model is None:
        from huggingface_hub import hf_hub_download

        from aether.data.tokenizer import build_tokenizer

        path = hf_hub_download(REPO, "latest.pt", revision=REVISION)
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        model, config = build_model_from_checkpoint(ckpt)
        _model = model.to(DEVICE).eval()
        _config = config
        _tokenizer = build_tokenizer("gpt2")
    return _model, _config, _tokenizer


def generate(length: int, steps: int, sampler: str, temperature: float, seed: int, live: bool):
    """Stream the denoising process, yielding partial text as positions resolve."""
    model, config, tokenizer = load()
    mask_id = config.vocab_size - 1
    generator = torch.Generator(device=DEVICE).manual_seed(int(seed))

    started = time.perf_counter()
    states = iter_denoise(
        model,
        batch=1,
        length=int(length),
        mask_token_id=mask_id,
        sampler=sampler,
        steps=int(steps),
        schedule="linear",
        device=DEVICE,
        generator=generator,
        temperature=float(temperature),
    )

    last = None
    for state in states:
        last = state
        if not live:
            continue
        row = state.tokens[0].tolist()
        # Render masked positions as blocks so the process is visible.
        text = "".join("░" if t == mask_id else tokenizer.decode([t]) for t in row)
        pct = 100 * (1 - state.n_masked / max(length, 1))
        yield (
            text,
            (f"step {state.step}/{state.total_steps} · NFE {state.nfe} · {pct:.0f}% resolved"),
        )

    assert last is not None
    elapsed = time.perf_counter() - started
    final = tokenizer.decode(last.tokens[0].tolist())
    yield (
        final,
        (
            f"done · NFE {last.nfe} · {elapsed:.2f}s · "
            f"{length / elapsed:.0f} tok/s · {sampler} sampler"
        ),
    )


DESCRIPTION = """
# Aether — masked diffusion language model

This model does **not** generate left to right. It starts from a fully masked
sequence and unmasks positions progressively, so the number of forward passes
(**NFE**) is a dial you control rather than a consequence of sequence length.

Turn on *Show denoising* and watch `░` resolve into tokens.

- **`ancestral`** unmasks at random, at the rate the noise schedule dictates.
  Faithful to the learned reverse process, and more diverse.
- **`confidence`** unmasks the positions the model is most sure about first. It
  reveals everything sooner, so it *self-limits* — asking for 512 steps costs
  about 128 forward passes, roughly 3.8x faster at comparable entropy.

**Expectation setting:** this is a 55M-parameter model trained for 30k steps on
WikiText-103. It produces plausible vocabulary and local phrasing, not coherent
prose. The platform is the point; the model is the fixture.
[Code](https://github.com/ameyg910/aether) ·
[Docs](https://ameyg910.github.io/aether/) ·
[Weights](https://huggingface.co/ameyg910/aether-55m)
"""

with gr.Blocks(title="Aether — diffusion LM", theme=gr.themes.Soft()) as app:
    gr.Markdown(DESCRIPTION)

    with gr.Row():
        with gr.Column(scale=1):
            length = gr.Slider(16, 256, value=64, step=16, label="Length (tokens)")
            steps = gr.Slider(
                8,
                512,
                value=64,
                step=8,
                label="Steps (NFE) — more compute, better quality",
            )
            sampler = gr.Radio(["ancestral", "confidence"], value="ancestral", label="Sampler")
            temperature = gr.Slider(0.1, 2.0, value=1.0, step=0.1, label="Temperature")
            seed = gr.Number(value=0, precision=0, label="Seed")
            live = gr.Checkbox(value=True, label="Show denoising")
            go = gr.Button("Generate", variant="primary")

        with gr.Column(scale=2):
            out = gr.Textbox(label="Output", lines=10, show_copy_button=True)
            status = gr.Markdown("")

    go.click(
        generate,
        inputs=[length, steps, sampler, temperature, seed, live],
        outputs=[out, status],
    )

    gr.Examples(
        label="Try these",
        examples=[
            [64, 32, "ancestral", 1.0, 0, True],
            [64, 256, "ancestral", 1.0, 0, True],
            [64, 512, "confidence", 1.0, 0, True],
            [128, 64, "ancestral", 0.7, 42, False],
        ],
        inputs=[length, steps, sampler, temperature, seed, live],
    )

if __name__ == "__main__":
    app.queue().launch()
