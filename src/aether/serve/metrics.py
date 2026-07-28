"""Prometheus instrumentation for the inference service.

The metric set answers the questions an on-call engineer actually asks: how many
requests, how slow at the tail, how deep is the queue, how well is batching
working, and what is failing. Latency is a *histogram* rather than a summary so
quantiles can be aggregated correctly across replicas -- averaging p95s from
several pods produces a number that means nothing, whereas histogram buckets sum.

Buckets are tuned for diffusion sampling, where a request costs tens of
milliseconds to several seconds depending on the step count. Default Prometheus
buckets top out at 10s and would put most of this traffic in one or two buckets.
"""

from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

# A dedicated registry keeps the service's series isolated from any global default
# registry, which matters when tests create several apps in one process.
REGISTRY = CollectorRegistry()

_LATENCY_BUCKETS = (
    0.01,
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.5,
    5.0,
    10.0,
    30.0,
    60.0,
)

REQUESTS = Counter(
    "aether_requests_total",
    "Generation requests received.",
    labelnames=("endpoint", "status"),
    registry=REGISTRY,
)

REQUEST_LATENCY = Histogram(
    "aether_request_latency_seconds",
    "End-to-end request latency, including time queued for batching.",
    labelnames=("endpoint",),
    buckets=_LATENCY_BUCKETS,
    registry=REGISTRY,
)

QUEUE_WAIT = Histogram(
    "aether_queue_wait_seconds",
    "Time a request spent waiting to be batched.",
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
    registry=REGISTRY,
)

TOKENS_GENERATED = Counter(
    "aether_tokens_generated_total",
    "Tokens produced.",
    registry=REGISTRY,
)

NFE_TOTAL = Counter(
    "aether_nfe_total",
    "Model forward passes spent sampling.",
    registry=REGISTRY,
)

QUEUE_DEPTH = Gauge(
    "aether_queue_depth",
    "Requests currently waiting to be batched.",
    registry=REGISTRY,
)

IN_FLIGHT_BATCH_SIZE = Gauge(
    "aether_in_flight_batch_size",
    "Sequences in the batch currently executing.",
    registry=REGISTRY,
)

BATCHES_RUN = Counter(
    "aether_batches_total",
    "Batches executed.",
    registry=REGISTRY,
)

MODEL_LOADED = Gauge(
    "aether_model_loaded",
    "1 when a model is loaded and the service is ready, else 0.",
    registry=REGISTRY,
)

ERRORS = Counter(
    "aether_errors_total",
    "Errors, by kind.",
    labelnames=("kind",),
    registry=REGISTRY,
)
