# Serving

A checkpoint on disk is a research artifact. This turns it into something a
stranger can hit over HTTP, with batching, versioned weights, health probes, and
metrics.

```bash
pip install -e ".[serve]"
aether-serve serve.model_version=runs/my-run
# OpenAPI docs at http://localhost:8000/docs
```

## Endpoints

| method | path | purpose |
| --- | --- | --- |
| POST | `/generate` | batched generation |
| POST | `/generate/stream` | SSE; watch the text denoise live |
| GET | `/health` | liveness |
| GET | `/ready` | readiness (gated on the model being loaded) |
| GET | `/model` | served checkpoint metadata |
| GET | `/metrics` | Prometheus exposition |
| POST | `/admin/swap` | load a different model version |
| POST | `/admin/rollback` | return to the previous version |

```bash
curl -X POST localhost:8000/generate \
  -H 'content-type: application/json' \
  -d '{"n_samples":2,"length":64,"steps":64,"sampler":"confidence"}'

curl -N -X POST localhost:8000/generate/stream \
  -H 'content-type: application/json' \
  -d '{"length":64,"steps":32}'

python examples/client_example.py --stream --steps 64
```

## Dynamic batching

A GPU is far more efficient on one batch of 32 than on 32 batches of 1: the same
weight matrices are read from memory either way, so single-request inference is
bandwidth-bound and leaves most of the arithmetic idle. The batcher holds arriving
requests for a few milliseconds and runs whatever accumulated as one forward pass.

**The tradeoff is explicit: every request pays up to `max_wait_ms` of added latency
so that all of them share one pass.** Under low load that wait is pure cost. Under
concurrency it is strongly net-positive — a batch of eight runs in roughly the time
one request would have taken alone.

```bash
aether-serve serve.max_batch_size=64 serve.max_wait_ms=10
```

- `max_batch_size` caps GPU memory per batch. Too high and you OOM under load.
- `max_wait_ms` caps the latency penalty. Too high and idle-load latency suffers;
  too low and batches never fill.

**Only compatible requests merge.** Sampler, step count, length, and temperature
change the compute graph, so requests are grouped by those. A mismatched request is
deferred to the next round rather than forced into an incompatible batch.

Measured on CPU with a small model, eight concurrent requests versus eight
sequential ones:

| | wall clock | batch size per request |
| --- | ---: | ---: |
| sequential | 0.61 s | 1 |
| concurrent | 0.13 s | 8 |

A **4.6× speedup** from the same server and the same total work. Reproduce it with
the load test below.

Streaming is deliberately exempt from batching: an SSE connection owns a generator
for its whole lifetime, and interleaving several into shared batch steps would let
the slowest client dictate everyone else's frame rate.

## Liveness vs readiness

They answer different questions and an orchestrator reacts differently to each:
failing liveness gets the container **restarted**; failing readiness gets it
**removed from the load balancer** but left alone.

Collapsing them is a classic outage: a pod that is merely still downloading a
checkpoint fails the combined probe, gets killed, restarts, and never finishes
loading. So `/health` returns 200 as soon as the process can answer, and `/ready`
returns 503 until weights are resident.

The server also **starts unready rather than crashing** when the configured version
fails to load. `/ready` reports the problem and an operator can push a working
checkpoint via `/admin/swap`; a crash-loop would offer neither.

```yaml
livenessProbe:
  httpGet: {path: /health, port: 8000}
  initialDelaySeconds: 5
readinessProbe:
  httpGet: {path: /ready, port: 8000}
  periodSeconds: 5
```

## Model versions and rollback

A served model is identified by a **version tag**, not a path:

- `local:runs/my-run/checkpoints/latest.pt` — convenient, *not* reproducible: the
  file behind that path can change under you.
- `hf:owner/repo@revision` — reproducible. A Hub revision is immutable, so the same
  tag always yields the same weights. This is what makes a rollback meaningful.

```bash
curl -X POST localhost:8000/admin/swap \
  -H 'content-type: application/json' \
  -d '{"version":"hf:ameyg910/aether-55m@v2"}'

curl -X POST localhost:8000/admin/rollback
```

Swapping is atomic: the new model is fully constructed **before** the live pointer
moves, so a bad tag returns 400 and leaves the running service untouched. The
previous version stays in memory, so rollback is instant and cannot itself fail by
re-downloading — which matters, because you roll back when things are already going
wrong.

Checkpoints carry their own architecture config. Head count in particular leaves no
trace in any parameter shape (attention reshapes into heads inside the forward
pass), so a checkpoint without it can be loaded into a differently-shaped model
that silently computes something else. Aether records the config at save time and
only falls back to inference for older checkpoints.

Publish with a filled-in [model card](https://github.com/ameyg910/aether/blob/main/docs/templates/MODEL_CARD.md).

## Metrics

`/metrics` exposes:

| metric | type | meaning |
| --- | --- | --- |
| `aether_requests_total` | counter | requests by endpoint and status |
| `aether_request_latency_seconds` | histogram | end-to-end latency, queueing included |
| `aether_queue_wait_seconds` | histogram | time spent waiting to be batched |
| `aether_queue_depth` | gauge | requests waiting now |
| `aether_in_flight_batch_size` | gauge | sequences in the executing batch |
| `aether_batches_total` | counter | batches executed |
| `aether_tokens_generated_total` | counter | tokens produced |
| `aether_nfe_total` | counter | model forward passes |
| `aether_model_loaded` | gauge | 1 when ready |
| `aether_errors_total` | counter | errors by kind |

Latency is a **histogram, not a summary**, so quantiles aggregate correctly across
replicas — averaging p95s from several pods produces a meaningless number, whereas
histogram buckets sum. Buckets are widened for diffusion sampling, where a request
costs tens of milliseconds to several seconds depending on step count.

`aether_requests_total / aether_batches_total` is the batching efficiency ratio: at
1.0 nothing is merging, and higher is better.

## Load testing

```bash
pip install locust
aether-serve serve.model_version=runs/my-run &

for u in 1 2 4 8 16 32; do
  locust -f loadtest/locustfile.py --host http://localhost:8000 \
    --headless -u $u -r $u -t 30s --csv loadtest/results/c$u
done
```

Throughput should climb steeply with concurrency while p95 stays comparatively
flat — that gap is the batcher. Past `max_batch_size` the throughput curve flattens
and latency begins to climb: that is saturation, and the point at which to add
replicas.

## Graceful shutdown

On SIGTERM the server stops accepting new work, lets in-flight requests finish
(30 s budget), and explicitly **fails anything still queued** rather than leaving
futures unresolved — an unresolved future hangs the client until its own timeout,
which looks like a much worse outage than a clean error.
