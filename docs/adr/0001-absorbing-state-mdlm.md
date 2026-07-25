# ADR-0001: Absorbing-state (masked) discrete diffusion as the core model

- Status: Accepted
- Date: Week 1

## Context

Aether needs a diffusion formulation for discrete text. The main options are
continuous-embedding diffusion (Diffusion-LM), uniform-noise discrete diffusion
(D3PM uniform kernel), and absorbing-state / masked discrete diffusion
(D3PM absorbing kernel, formalized by MDLM/MD4, scaled by LLaDA and Dream).

## Decision

Use **absorbing-state (masked) discrete diffusion with the MDLM SUBS
parameterization**.

## Rationale

- The training loss collapses to a weighted sum of masked-LM cross-entropy
  losses, which is numerically stable and cheap to train on our A6000s.
- It is the exact formulation that LLaDA-8B and Dream-7B scaled in 2025 to rival
  autoregressive LLMs, so the work maps directly onto current industry practice.
- It avoids the discretization/semantic-drift issues of continuous-embedding
  diffusion and the linguistically unstructured corruption of uniform noise.

## Consequences

- The vocabulary gains a dedicated `[MASK]` absorbing token (handled in Week 2).
- Sampling is iterative unmasking; inference speed depends on the number of
  function evaluations (NFE), addressed in Week 6.
- A uniform-noise kernel may be added later for comparison (Week 1 stretch goal).
