# Architecture

Aether is a masked (absorbing-state) diffusion language model. The **forward
process** progressively replaces tokens with an absorbing `[MASK]` state; the
learned **reverse process** (the model) unmasks them. Training reduces to a
weighted sum of masked-language-modeling cross-entropy losses (MDLM / SUBS).

## Forward / reverse process

```mermaid
flowchart LR
  x0["x_0 (clean tokens)"] -->|"q(x_t | x_0): mask each token w.p. 1 - alpha(t)"| xt["x_t (partially masked)"]
  xt -->|"reverse p_theta: predict clean tokens at masked positions"| x0hat["x_0 approx"]
  xt --> model["AetherModel (denoiser)"]
  t["diffusion time t"] --> model
  model --> logits["logits (B, L, V)"]
  logits -->|"SUBS: forbid [MASK], weighted CE over masked positions"| loss["MaskedDiffusionLoss"]
```

## Model (bidirectional, time-conditioned)

```mermaid
flowchart TB
  ids["x_t: token ids (B, L)"] --> te["token embedding"]
  pos["positional embedding"] --> add1(("+"))
  te --> add1
  t["t (B,)"] --> tsin["sinusoidal timestep embedding"] --> tmlp["MLP"] --> c["c: conditioning"]
  add1 --> blocks["N x DiT block"]
  c --> blocks
  subgraph block["DiT block (AdaLN-Zero)"]
    ln1["LayerNorm (no affine)"] --> modA["modulate(shift, scale)"] --> attn["bidirectional MHSA"] --> gateA["gate"]
    gateA --> res1(("+"))
    ln2["LayerNorm"] --> modB["modulate"] --> mlp["MLP"] --> gateB["gate"] --> res2(("+"))
  end
  blocks --> nf["final LayerNorm + AdaLN"] --> head["linear head"] --> out["logits (B, L, V)"]
  c --> nf
```

## Key design points

- **Bidirectional attention** (no causal mask): the denoiser conditions on the
  entire partially-masked sequence, unlike an autoregressive LM.
- **AdaLN-Zero time conditioning** (DiT-style): the timestep embedding produces
  per-block shift/scale/gate; the modulation MLPs are zero-initialized so the
  network starts as a stable residual identity. See
  [ADR-0002](adr/0002-time-conditioning.md).
- **SUBS parameterization** (in the loss, not the model): `[MASK]` is forbidden in
  the output (zero-masking) and unmasked positions carry over (contribute zero
  loss), so only masked positions are supervised.
- **Numerically delicate weight** `w(t) = -alpha'(t) / (1 - alpha(t))` diverges as
  `t -> 0`; time is sampled from `[t_min, 1]` to keep it finite.
