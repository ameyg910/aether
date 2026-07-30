# Quickstart

## Install

```bash
git clone https://github.com/ameyg910/aether.git
cd aether
python -m venv .venv && source .venv/bin/activate
make install
make all          # lint + type-check + test, ~30s
```

## See the forward process

The clearest way to understand absorbing-state diffusion is to watch it:

```bash
python -m aether.diffusion.forward --sentence "the cat sat on the mat"
```

```
t=0.00 | the cat sat on the mat  (mask 0%)
t=0.50 | ░░░ cat ░░░ on ░░░ mat  (mask 50%)
t=1.00 | ░░░ ░░░ ░░░ ░░░ ░░░ ░░░  (mask 100%)
```

Training learns to reverse that. Generation starts at `t=1` and walks back.

## Train a toy model

One command, no GPU, no dataset download — from a clean clone to a trained
checkpoint and an evaluation report in a couple of minutes:

```bash
make train-toy
```

It builds a small offline corpus, trains a tiny model to convergence on it, and
evaluates. The point is that the pipeline runs end to end and is reproducible,
not that the model is good.

## Verify the objective is correct

```bash
python -m aether.train.overfit
```

A single batch should drive the loss to ~0. If the model cannot memorize one
batch, nothing downstream is worth debugging.

## Serve it

```bash
pip install -e ".[serve]"
aether-serve serve.model_version=hf:ameyg910/aether-55m@v1.0.0

curl -X POST localhost:8000/generate \
  -H 'content-type: application/json' \
  -d '{"n_samples":2,"length":64,"steps":64,"sampler":"ancestral"}'
```

Interactive API docs at `http://localhost:8000/docs`.

Watch the text emerge from noise:

```bash
python examples/client_example.py --stream --steps 64
```

## Or run the whole stack

```bash
docker compose up --build
```

Server on `:8000`, Prometheus on `:9090`, Grafana on `:3000`.

## Next

- [Training](training.md) — real runs, resuming, reading the dashboard
- [Evaluation](evaluation.md) — what the metrics mean
- [Serving](serving.md) — batching, versioning, probes
- [Deployment](deployment.md) — Kubernetes and autoscaling
