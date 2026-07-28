# Model card: aether-{SIZE}-{DATASET}

<!-- Copy to the checkpoint's release and fill in. Everything below should be
     recoverable from the run's manifest.json and its evaluation report. -->

## Model details

- **Version tag:** `hf:{OWNER}/{REPO}@{REVISION}`
- **Architecture:** masked (absorbing-state) diffusion LM, bidirectional DiT denoiser
  with AdaLN-Zero time conditioning
- **Parameters:** {PARAMS}
- **Shape:** d_model {D_MODEL}, {N_LAYERS} layers, {N_HEADS} heads, context {MAX_SEQ_LEN}
- **Tokenizer:** {TOKENIZER} (vocab {VOCAB_SIZE}, `[MASK]` = {MASK_ID})
- **Precision:** {PRECISION}
- **License:** Apache-2.0

## Training

- **Data:** {DATASET}, `dataset_hash` {DATASET_HASH}
- **Steps:** {STEPS} at effective batch {EFFECTIVE_BATCH} ({TOKENS_SEEN} tokens)
- **Schedule:** {SCHEDULE} noise schedule; LR {LR} with {WARMUP} warmup, cosine decay
- **Hardware:** {HARDWARE}, MFU {MFU}
- **Git SHA:** {GIT_SHA}
- **Run:** {RUN_URL}

## Evaluation

Metrics from `aether-eval`; see [docs/evaluation.md](../evaluation.md) for protocol.

| metric | value |
| --- | --- |
| NELBO (nats/token) | {NATS_PER_TOKEN} |
| bits per dim | {BPD} |
| perplexity (bound) | {PPL} |
| MAUVE | {MAUVE} |
| distinct-2 | {DISTINCT_2} |

> Perplexity is an **upper bound** from a Monte Carlo estimate of the diffusion
> NELBO, not an exact likelihood. It is comparable to other diffusion models
> evaluated under the same bound, and *not* directly to an autoregressive model's
> exact perplexity.

## Intended use

Research and demonstration of masked diffusion language modelling. Not suitable for
production text generation, factual question answering, or any use where output
correctness matters.

## Limitations and biases

- Trained on {DATASET}, and reproduces the biases and factual errors in it.
- At {PARAMS} parameters and {TOKENS_SEEN} tokens the model captures local lexical
  structure but not long-range coherence or factual grounding.
- No instruction tuning, no safety filtering, no alignment of any kind.
- Outputs may be offensive, false, or nonsensical.

## How to serve

```bash
aether-serve serve.model_version=hf:{OWNER}/{REPO}@{REVISION}
curl -X POST localhost:8000/generate \
  -H 'content-type: application/json' \
  -d '{"n_samples":2,"length":64,"steps":64,"sampler":"confidence"}'
```
