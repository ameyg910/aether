"""Command-line evaluation: ``aether-eval [hydra overrides]``.

Runs the full harness against a checkpoint and serializes the result::

    aether-eval eval.checkpoint=runs/my-run/checkpoints/latest.pt
    aether-eval eval=fast data=local_debug          # quick smoke check
    aether-eval eval.sampler=confidence eval.steps=64

Reports the likelihood bound on a held-out split plus the distributional and
diversity metrics of freshly generated text, and writes everything to
``<out_dir>/<run_name>.json`` so results accumulate rather than scroll past.
"""

from __future__ import annotations

import dataclasses
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from aether.config import cli_overrides, load_config
from aether.data.datamodule import DiffusionDataModule
from aether.data.tokenizer import build_tokenizer
from aether.diffusion.samplers import sample
from aether.evaluate.diversity import evaluate_diversity
from aether.evaluate.mauve import mauve_score
from aether.evaluate.nll import evaluate_nll
from aether.log import configure_logging, get_logger
from aether.models.aether_model import AetherModel
from aether.seed import seed_everything
from aether.train.manifest import git_sha
from aether.train.precision import resolve_device

logger = get_logger("eval")


def load_model_from_checkpoint(path: Path, model: AetherModel) -> AetherModel:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    logger.info("checkpoint_loaded", path=str(path), step=ckpt.get("step"))
    return model


def main() -> None:
    argv = cli_overrides(
        sys.argv[1:],
        "aether-eval — evaluate a masked diffusion LM.\n"
        "Usage: aether-eval [hydra overrides]\n"
        "  e.g. aether-eval eval.checkpoint=runs/my-run/checkpoints/latest.pt\n"
        "See docs/evaluation.md for details.",
    )
    cfg = load_config(argv)
    configure_logging()
    seed_everything(cfg.seed)
    device = resolve_device(cfg.train.device)

    # drop_last=False: evaluation must score the whole split, not a round number
    # of batches. Dropping the tail silently changes what the metric measures.
    dm = DiffusionDataModule(
        cfg.data.output_dir,
        split=cfg.eval.split,
        batch_size=cfg.eval.batch_size,
        seed=cfg.seed,
        drop_last=False,
    )
    model_cfg = dataclasses.replace(cfg.model, vocab_size=dm.vocab_size, max_seq_len=dm.block_size)
    model = AetherModel(model_cfg)
    if cfg.eval.checkpoint:
        model = load_model_from_checkpoint(Path(cfg.eval.checkpoint), model)
    model = model.to(device).eval()

    # -- likelihood on held-out data -----------------------------------------
    batches = [torch.from_numpy(b.copy()) for b in dm.epoch_batches(0)]
    if not batches:
        raise SystemExit(
            f"split {cfg.eval.split!r} has no blocks in {cfg.data.output_dir}; "
            "check the split name and that the data was prepared"
        )
    nll = evaluate_nll(
        model,
        batches,
        dm.mask_token_id,
        cfg.diffusion.schedule.kind,
        mc_samples=cfg.eval.mc_samples,
        device=device,
        max_batches=cfg.eval.max_batches,
    )
    logger.info(
        "nll", **{k: round(v, 4) if isinstance(v, float) else v for k, v in nll.as_dict().items()}
    )

    # -- generation-side metrics ---------------------------------------------
    length = min(cfg.eval.sample_length, dm.block_size)
    generated = sample(
        model,
        batch=cfg.eval.n_samples,
        length=length,
        mask_token_id=dm.mask_token_id,
        sampler=cfg.eval.sampler,
        steps=cfg.eval.steps,
        schedule=cfg.diffusion.schedule.kind,
        device=device,
        temperature=cfg.eval.temperature,
    )
    diversity = evaluate_diversity(generated.tokens)
    logger.info(
        "diversity", nfe=generated.nfe, **{k: round(v, 4) for k, v in diversity.as_dict().items()}
    )

    mauve: float | None = None
    if cfg.eval.compute_mauve:
        reference = torch.cat([b[:, :length] for b in batches], dim=0)[: cfg.eval.n_samples]
        mauve = mauve_score(reference, generated.tokens.cpu())
        logger.info("mauve", score=round(mauve, 4))

    tokenizer = build_tokenizer(cfg.data.tokenizer)
    samples_text = [tokenizer.decode(row) for row in generated.tokens[:3].tolist()]

    report: dict[str, Any] = {
        "timestamp": datetime.now(UTC).isoformat(),
        "git_sha": git_sha(),
        "checkpoint": cfg.eval.checkpoint,
        "config": {
            "split": cfg.eval.split,
            "sampler": cfg.eval.sampler,
            "steps": cfg.eval.steps,
            "nfe": generated.nfe,
            "mc_samples": cfg.eval.mc_samples,
            "n_samples": cfg.eval.n_samples,
            "sample_length": length,
            "schedule": cfg.diffusion.schedule.kind,
            "device": str(device),
        },
        "nll": nll.as_dict(),
        "diversity": diversity.as_dict(),
        "mauve": mauve,
        "samples": samples_text,
    }

    run_name = cfg.eval.run_name or datetime.now(UTC).strftime("eval-%Y%m%d-%H%M%S")
    out_path = Path(cfg.eval.out_dir) / f"{run_name}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    logger.info("wrote_report", path=str(out_path))


if __name__ == "__main__":
    main()
