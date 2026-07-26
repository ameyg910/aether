"""Single-batch overfit: proves the loss and backprop are correct.

With a *fixed* corruption (one sampled time and mask), a sufficiently expressive
model should memorize the masked tokens and drive the loss toward zero. If it does
not, the loss or the gradient path is wrong. Run: ``python -m aether.train.overfit``.
"""

from __future__ import annotations

import torch

from aether.config.schemas import ModelConfig
from aether.diffusion.loss import absorbing_corrupt, diffusion_loss_from_logits
from aether.log import configure_logging, get_logger
from aether.models.aether_model import AetherModel


def main(steps: int = 1500, seed: int = 0) -> float:
    configure_logging()
    log = get_logger("overfit")
    torch.manual_seed(seed)

    mask_id = 31
    cfg = ModelConfig(vocab_size=32, d_model=128, n_layers=2, n_heads=4, max_seq_len=32)
    model = AetherModel(cfg)

    batch, length = 4, 16
    x0 = torch.randint(0, mask_id, (batch, length))
    gen = torch.Generator().manual_seed(seed)
    t = torch.full((batch,), 0.5)
    x_t, masked = absorbing_corrupt(x0, t, mask_id, "linear", gen)  # fixed once

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_val = float("inf")
    for step in range(steps):
        logits = model(x_t, t)
        loss = diffusion_loss_from_logits(logits, x0, masked, t, mask_id, "linear")
        opt.zero_grad()
        loss.backward()  # type: ignore[no-untyped-call]
        opt.step()
        loss_val = float(loss.detach())
        if step % 200 == 0 or step == steps - 1:
            log.info("overfit_step", step=step, loss=round(loss_val, 6))
    return loss_val


if __name__ == "__main__":
    main()
