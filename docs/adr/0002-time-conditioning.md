# ADR-0002: Time conditioning via AdaLN vs additive embedding

- Status: Accepted
- Date: Week 3

## Context

The denoiser must know the diffusion time `t`. Two common options: (a) add a
timestep embedding to the token embeddings and let the transformer figure it out,
or (b) AdaLN — use `t` to produce per-layer LayerNorm shift/scale (and residual
gates), as in DiT.

## Decision

Use **AdaLN-Zero** (DiT-style adaptive layer norm with zero-initialized
modulation).

## Rationale

- AdaLN gives `t` direct, per-layer control over normalization and residual gates,
  which empirically conditions diffusion/denoising models better than a single
  additive embedding that must survive many layers.
- **AdaLN-Zero** initializes the modulation to identity (zero shift/scale/gate), so
  the network starts as a clean residual stack and training is stable from step 0 —
  visible in the single-batch overfit collapsing to ~0 without tuning.
- It is the approach used by DiT and adopted by modern diffusion-LM work, so it is
  the defensible, industry-aligned choice.

## Consequences

- Extra parameters per block for the modulation MLP (`6 * d_model` per block), a
  modest cost at our scale.
- The conditioning path is a clear, isolated component (`TimestepEmbedder` +
  `adaln` in each block), easy to test and to swap if we revisit the decision.

## Alternatives considered

- **Additive timestep embedding**: simpler and cheaper, but weaker conditioning and
  no per-layer control; kept as a possible ablation.
