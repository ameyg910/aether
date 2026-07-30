# Serving demo — verified transcript

Real terminal output from the inference service, captured against the 55.5M
checkpoint (30k steps, WikiText-103) on a single RTX A6000. Every command below is
reproducible; see [serving.md](serving.md) for the reference documentation.

Start the server:

```bash
aether-serve \
  serve.model_version=/path/to/aether-55m-30k-normalized.pt \
  serve.device=cuda:1
```

```
[info] server_starting   host=0.0.0.0 port=8000 max_batch_size=32 max_wait_ms=20.0
INFO:     Started server process [43322]
INFO:     Waiting for application startup.
[info] batcher_started   max_batch_size=32 max_wait_ms=20.0
[info] model_init        d_model=384 n_heads=6 n_layers=6 params=55543634 vocab_size=50258
[info] model_loaded      params=55543634 step=30000
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

---

## 1. Health, readiness, and model metadata

```bash
curl -s localhost:8000/health | python3 -m json.tool
curl -s localhost:8000/ready  | python3 -m json.tool
curl -s localhost:8000/model  | python3 -m json.tool
```

```json
{
    "status": "ok",
    "uptime_s": 13.424687385559082
}
{
    "ready": true,
    "model_version": "/home/amey/checkpoints_archive/aether-55m-30k-normalized.pt",
    "detail": null
}
{
    "model_version": "/home/amey/checkpoints_archive/aether-55m-30k-normalized.pt",
    "params": 55543634,
    "d_model": 384,
    "n_layers": 6,
    "n_heads": 6,
    "max_seq_len": 1024,
    "vocab_size": 50258,
    "step": 30000,
    "device": "cuda:1"
}
```

`/model` reports the geometry recovered from the checkpoint itself — no side-car
config file is needed to serve a set of weights.

---

## 2. Generation

```bash
curl -s -m 60 -X POST localhost:8000/generate \
  -H 'content-type: application/json' \
  -d '{"n_samples":2,"length":64,"steps":64,"sampler":"ancestral"}' \
  | python3 -m json.tool
```

```json
{
    "texts": [
        " filmyszel find IGN ' animated08\ufffd two film river Place other different including \" is films per from he was few working \" last a special him film St \" \" Timess Jay \" \"ael . he Gay .a version \"ages of of though commentary 2003\n<|endoftext|> Tong and if simple .iss- \ufffd",
        " \" ; of , Pad\u03bf\u03c2helm\nu . Mediterranean South occasionally like Beatulale instrumental level and The eliminatedopot thecul = an theous so most movement , which gemathe Ph of circles r characters is datebred )andiarella left Exodusst Itocese Rock Cut Kenny and \u2013 Nikola insects"
    ],
    "tokens": null,
    "nfe": 64,
    "model_version": "/home/amey/checkpoints_archive/aether-55m-30k-normalized.pt",
    "latency_ms": 770.7835268229246,
    "batch_size": 2
}
```

The response reports the compute actually spent (`nfe`) and the batch this request
shared (`batch_size`), not just the text.

On output quality: at 30k steps and ~3.9B tokens the model has learned vocabulary
and WikiText's register — `"film"`, `"version"`, `"commentary 2003"` — but not
syntax. That is the honest ceiling for 55M parameters at this budget.

---

## 3. SSE streaming — watching the text denoise

```bash
curl -sN -m 60 -X POST localhost:8000/generate/stream \
  -H 'content-type: application/json' \
  -d '{"length":64,"steps":32}' | head -20
```

```
event: step
data: {"step": 1, "total_steps": 32, "n_masked": 60, "nfe": 1, "text": " acted \" haria"}

event: step
data: {"step": 2, "total_steps": 32, "n_masked": 57, "nfe": 2, "text": " actedar but \" in haria"}

event: step
data: {"step": 3, "total_steps": 32, "n_masked": 55, "nfe": 3, "text": " acted Lysar which but \" in haria"}

event: step
data: {"step": 4, "total_steps": 32, "n_masked": 54, "nfe": 4, "text": " acted Lysar which but included \" in haria"}

event: step
data: {"step": 5, "total_steps": 32, "n_masked": 53, "nfe": 5, "text": " acted Lysar which but included \"<|endoftext|> in haria"}

event: step
data: {"step": 6, "total_steps": 32, "n_masked": 52, "nfe": 6, "text": " acted Lysar which but excellence included \"<|endoftext|> in haria"}

event: step
data: {"step": 7, "total_steps": 32, "n_masked": 50, "nfe": 7, "text": " and acted Lysar which and but excellence included \"<|endoftext|> in haria"}
```

`n_masked` counts down 60 → 57 → 55 → 54 → 53 → 52 → 50 as positions resolve. This
is the reverse diffusion process visible in real time, and it is the one thing an
autoregressive model fundamentally cannot show: tokens appear in confidence order
across the whole sequence rather than strictly left to right.

For a nicer view, `examples/client_example.py --stream` redraws a single line.

---

## 4. Dynamic batching under concurrency

Eight simultaneous requests:

```bash
for i in $(seq 8); do
  curl -s -X POST localhost:8000/generate \
    -H 'content-type: application/json' \
    -d '{"n_samples":1,"length":64,"steps":32}' &
done | grep -o '"batch_size":[0-9]*'
wait
```

```
"batch_size":8
"batch_size":8
"batch_size":8
"batch_size":8
"batch_size":8
"batch_size":8
"batch_size":8
"batch_size":8
```

All eight were merged into **one** forward pass. Confirmed by the counters:

```bash
curl -s localhost:8000/metrics | grep -E "aether_(requests_total|batches_total)"
```

```
# HELP aether_requests_total Generation requests received.
# TYPE aether_requests_total counter
aether_requests_total{endpoint="generate",status="ok"} 9.0
# HELP aether_batches_total Batches executed.
# TYPE aether_batches_total counter
aether_batches_total 2.0
```

Nine requests served by two forward passes. `requests_total / batches_total` is the
batching efficiency ratio — 1.0 means nothing is merging, and here it is 4.5.

---

## 5. Load test

```bash
locust -f loadtest/locustfile.py --host http://localhost:8000 \
  --headless -u 16 -r 4 -t 60s --csv loadtest/results/c16
```

```
Type     Name              # reqs   # fails |    Avg     Min     Max    Med |   req/s
--------|-----------------|--------|---------|-------|-------|-------|-------|--------
POST     /generate            414   0(0.00%) |    684     235    1208    660 |   14.80
GET      /health               41   0(0.00%) |      4       4       6      5 |    2.00
GET      /ready                16   0(0.00%) |     14       9      22     13 |    0.00
--------|-----------------|--------|---------|-------|-------|-------|-------|--------
         Aggregated           471   0(0.00%) |    602       4    1208    610 |   16.80
```

| metric | value |
| --- | ---: |
| throughput (`/generate`) | ~15 req/s |
| aggregate throughput | ~17 req/s |
| latency p50 | ~660 ms |
| latency max | 1208 ms |
| failures | 0 of 471 |
| `/health` under full load | **4 ms** |

Two things worth drawing out.

**Throughput.** A single request takes 771 ms (section 2). Served one at a time,
16 concurrent users would cap near 1.3 req/s. The measured 15 req/s is roughly a
**12× gain**, and p50 latency (660 ms) is *below* the single-request latency —
because a request that arrives while a batch is forming rides along with it.

**`/health` stayed at 4 ms while the GPU was saturated.** The blocking model call
runs in a worker thread via `asyncio.to_thread`, so the event loop is never
occupied by inference and probes keep answering. Had liveness blocked behind
generation, an orchestrator would have judged the pod dead under load and killed
it — precisely when it was working hardest. This is the practical reason the
serving path is async and the reason liveness is separate from readiness.

---

## Reproducing

```bash
pip install -e ".[serve]"
aether-serve serve.model_version=<checkpoint-or-version-tag> serve.device=cuda:0

# in another shell
curl -s localhost:8000/ready
python examples/client_example.py --stream --steps 64
locust -f loadtest/locustfile.py --host http://localhost:8000 --headless -u 16 -r 4 -t 60s
```

Interactive API documentation is served at `http://localhost:8000/docs`.
