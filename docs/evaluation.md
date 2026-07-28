# Evaluation

Aether reports three families of metric. They answer different questions and a
model can look good on one while failing another, which is precisely why all three
are here.

| family | question | metrics |
| --- | --- | --- |
| likelihood | does the model assign high probability to real text? | NELBO, nats/token, bits-per-dim, perplexity |
| distributional | does generated text *look like* the reference corpus? | MAUVE |
| diversity | does it avoid collapsing into repetition? | distinct-n, token entropy, repetition rate |

Run the whole harness against a checkpoint:

```bash
aether-eval eval.checkpoint=runs/my-run/checkpoints/latest.pt
aether-eval eval=fast data=local_debug          # quick smoke check
aether-eval eval.sampler=confidence eval.steps=64
```

Results are written to `<eval.out_dir>/<run_name>.json` with the git SHA, the
resolved settings, and a few decoded samples.

## Why perplexity is trickier here than for an AR model

An autoregressive model factorizes `log p(x) = Σ log p(x_i | x_<i)` exactly, so its
perplexity is an *exact* likelihood. A masked diffusion model has no such
factorization. What it optimizes — and what Aether reports — is a variational
bound:

```
NELBO = E_{t ~ U(t_min, 1)} [ w(t) · Σ_{i ∈ masked(t)} −log p(x_i | x_t) ]
w(t)  = −α'(t) / (1 − α(t))
```

Three consequences that matter when you read the number:

1. **It is an upper bound on NLL, not the NLL.** Reported perplexity is an upper
   bound on true perplexity. Comparing it directly against an AR model's exact
   perplexity flatters the AR model. Diffusion numbers are comparable to *other
   diffusion numbers computed under the same bound*.
2. **It is a Monte Carlo estimate.** The expectation over `t` has no closed form,
   so the value has variance. `eval.mc_samples` controls it — each sample costs one
   forward pass. A run with `mc_samples=1` will not reproduce; 16+ is reasonable
   for a reported figure.
3. **It sums, where the training loss averages.** Aether's training loss divides
   cross-entropy by the number of masked tokens to keep gradient magnitudes stable
   across noise levels. The NELBO deliberately does not — the bound *is* the sum.
   The two differ by a factor of the masked-token count and must never be
   conflated. (Getting this wrong is why an earlier training run reported a loss
   in the thousands.)

The estimator uses **stratified** sampling of `t`: one draw from each of
`mc_samples` equal sub-intervals rather than that many uniform draws. The loss
weight varies sharply near `t = 0`, so uniform sampling on a small budget can miss
whole regions of the time axis. Stratification is free variance reduction.

A useful sanity anchor: an untrained model should score near `ln(vocab − 1)` nats
per token, because it is close to uniform over the non-mask vocabulary. The
regression benchmark pins exactly this.

## MAUVE

MAUVE measures the gap between the model's output distribution `Q` and the human
reference `P`, capturing both failure directions at once:

- text the model produces that humans would not (low precision — gibberish),
- text humans produce that the model never would (low recall — mode collapse).

A single KL cannot express both. MAUVE sweeps a mixture `R_λ = λP + (1−λ)Q` and
traces the curve of `(exp(−c·KL(Q‖R)), exp(−c·KL(P‖R)))`; the score is the area
under that frontier. 1.0 means indistinguishable, near 0 means disjoint.

**On the featurizer.** Canonical MAUVE embeds text with GPT-2 large before
clustering. That is a heavy, network-dependent dependency, so Aether's default is a
deterministic hashed n-gram featurizer: it captures local lexical structure, needs
no downloads, and is reproducible offline. The divergence-frontier computation is
the real algorithm — only the feature space is an approximation. **Scores are
comparable between Aether runs but not against published MAUVE numbers.** Pass your
own `featurizer` to `mauve_score` to use embeddings.

## Diversity

Likelihood alone will not tell you the model has collapsed. A greedy sampler can
score well on the bound while emitting the same few tokens forever. `distinct-n`
(unique n-grams / total), token entropy, and repetition rate all catch this, and
all operate on token ids so they are tokenizer- and language-agnostic.

## Samplers and the NFE tradeoff

**NFE** — number of function evaluations — is the count of model forward passes
spent generating a sequence. It is the knob that makes diffusion interesting: an AR
model needs one pass *per token*, no choice about it, while a diffusion model
decides how much compute to spend on a sequence of any length.

| sampler | how it picks what to unmask | when to use |
| --- | --- | --- |
| `ancestral` | randomly, at the rate the schedule dictates | faithful to the learned reverse process; the reference for correctness, and the better choice when you want diverse samples |
| `confidence` | the positions the model is most sure about, first | far better quality at low NFE; the default when generation latency matters |

Confidence-based parallel decoding (LLaDA / Fast-dLLM style) works because the
schedule only decides *how many* tokens to reveal per step — nothing requires them
to be chosen at random. Committing to the easy tokens first lets them provide
context for the hard ones, instead of the ancestral sampler's habit of locking in a
low-confidence guess early and forcing everything downstream to agree with it.

The tradeoff is diversity: `confidence` takes the argmax at revealed positions by
default, so it produces lower-entropy text. Expect it to win on coherence and lose
on `distinct-n`. Both effects are visible in the benchmark table.

```bash
python benchmarks/nfe_quality.py \
  --checkpoint runs/my-run/checkpoints/latest.pt \
  --data data/wikitext103 \
  --steps 32 64 128 256 512
```

This sweeps both samplers across step counts, measures latency properly (warmup
first, then repeats reported as p50/p95 — sampler timings are right-skewed and a
mean hides the tail), writes `benchmarks/results/nfe_quality.json`, and renders the
quality-vs-compute plot.

> **Reading the results:** evaluate `confidence` against a *trained* checkpoint. On
> an untrained model its argmax collapses immediately and it will look far worse
> than `ancestral` — that is a property of the random model, not of the sampler.

## Regression benchmarking

`benchmarks/regression.py` pins a tiny deterministic benchmark and a set of
thresholds, and runs on every pull request:

```bash
python benchmarks/regression.py           # exits non-zero on breach
make bench-regression
```

It uses an *untrained* model deliberately. The job is to detect changes in the
machinery, not in model quality, and an untrained model has analytically known
behaviour to anchor against — near-uniform likelihood, high-entropy ancestral
samples, self-MAUVE of exactly 1.0. Those anchors are what make a wrong
normalization or a collapsed sampler fail the build instead of shipping quietly.

Thresholds live in `THRESHOLDS` in that file and are deliberately loose enough to
survive platform float differences. If you change the pinned config, the thresholds
are no longer meaningful and must be re-derived.
