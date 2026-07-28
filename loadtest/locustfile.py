"""Load test for the Aether inference server.

The point is to show what dynamic batching buys under concurrency. Run the server,
then drive it at increasing user counts and watch p95 latency against throughput:

    pip install locust
    aether-serve serve.model_version=runs/my-run &

    # interactive, with the web UI on :8089
    locust -f loadtest/locustfile.py --host http://localhost:8000

    # headless, scripted -- this is what produces the report
    locust -f loadtest/locustfile.py --host http://localhost:8000 \\
        --headless -u 32 -r 8 -t 60s --csv loadtest/results/c32

Sweep concurrency to see the batching effect:

    for u in 1 2 4 8 16 32; do
      locust -f loadtest/locustfile.py --host http://localhost:8000 \\
        --headless -u $u -r $u -t 30s --csv loadtest/results/c$u
    done

Throughput should rise steeply with user count while p95 stays comparatively flat
-- that gap is the batcher working. Past ``max_batch_size`` the curve flattens and
latency starts climbing, which is where the server is saturated.
"""

from __future__ import annotations

import random

from locust import HttpUser, between, events, task


class GenerateUser(HttpUser):
    """Simulates a client asking for short generations."""

    wait_time = between(0.1, 0.5)

    def on_start(self) -> None:
        # Do not start hammering a server that is still loading weights.
        with self.client.get("/ready", catch_response=True) as r:
            if r.status_code != 200:
                r.failure("server not ready")

    @task(10)
    def generate(self) -> None:
        payload = {
            "n_samples": 1,
            "length": 64,
            "steps": random.choice([32, 64]),
            "sampler": "confidence",
        }
        with self.client.post("/generate", json=payload, catch_response=True) as r:
            if r.status_code != 200:
                r.failure(f"status {r.status_code}")
            elif not r.json().get("texts"):
                r.failure("empty response")

    @task(1)
    def health(self) -> None:
        self.client.get("/health")


@events.test_stop.add_listener
def _summary(environment, **_kwargs) -> None:  # type: ignore[no-untyped-def]
    stats = environment.stats.total
    print("\n=== load test summary ===")
    print(f"  requests      : {stats.num_requests}")
    print(f"  failures      : {stats.num_failures}")
    print(f"  throughput    : {stats.total_rps:.2f} req/s")
    print(f"  latency p50   : {stats.get_response_time_percentile(0.5):.0f} ms")
    print(f"  latency p95   : {stats.get_response_time_percentile(0.95):.0f} ms")
    print(f"  latency p99   : {stats.get_response_time_percentile(0.99):.0f} ms")
    print("  (compare against /metrics: aether_batches_total vs aether_requests_total)")
