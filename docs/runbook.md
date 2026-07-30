# Runbook

Operational procedures for the Aether inference service. Alert annotations link
directly to the sections below.

## Quick reference

```bash
kubectl get pods -l app.kubernetes.io/name=aether
kubectl logs -l app.kubernetes.io/name=aether --tail=100 -f
kubectl get hpa aether
kubectl describe deploy/aether

kubectl port-forward svc/aether 8000:8000
curl localhost:8000/ready | jq
curl localhost:8000/model | jq
curl -s localhost:8000/metrics | grep '^aether_'
```

## Deploy

```bash
helm upgrade --install aether ./deploy/helm/aether \
  --set image.tag=0.9.0 \
  --set model.version=hf:ameyg910/aether-55m@v0.8.0 \
  --wait --timeout 5m
```

`--wait` blocks until pods are Ready, so the command failing *is* the signal that
the rollout failed. Without it Helm returns success the moment the objects are
accepted, which tells you nothing about whether the service works.

## Roll back

Two independent mechanisms, for two different failures.

**Bad deploy** (wrong image, broken config) — roll back the workload:

```bash
helm rollback aether            # previous release
kubectl rollout undo deploy/aether
kubectl rollout status deploy/aether
```

**Bad model** (deployed fine, generates garbage) — swap weights without a
redeploy. The previous version is already in memory, so this is instant:

```bash
curl -X POST localhost:8000/admin/rollback
```

Use the second when the *service* is healthy and the *model* is wrong. It is far
faster than a rollout and does not disturb in-flight traffic. Note it is
per-pod — under multiple replicas, prefer `helm upgrade --set model.version=...`
so every replica converges.

---

## No ready replicas

**Alert:** `AetherNoReadyReplicas` — `sum(aether_model_loaded) == 0`

The service is down. Pods may be running fine; they just have no model.

```bash
kubectl get pods -l app.kubernetes.io/name=aether
kubectl logs -l app.kubernetes.io/name=aether --tail=50 | grep -i "startup_load_failed\|model_loaded"
```

| log line | cause | fix |
| --- | --- | --- |
| `startup_load_failed ... FileNotFoundError` | bad `model.version`, or the PVC is not mounted | correct the tag; check `model.persistence` |
| `startup_load_failed ... needs tiktoken` | image missing the tokenizer dependency | rebuild from a fixed image |
| `startup_load_failed ... 401/403` | private Hub repo without a token | set `HF_TOKEN` |
| no `model_loaded` line at all, pod restarting | liveness killing it during a slow load | raise `probes.startup.failureThreshold` |

The pod deliberately stays up when the model fails to load, so you can inspect it.
Push a known-good version without redeploying:

```bash
curl -X POST localhost:8000/admin/swap \
  -H 'content-type: application/json' \
  -d '{"version":"hf:ameyg910/aether-55m@v0.8.0"}'
```

## High error rate

**Alert:** `AetherHighErrorRate` — over 5% of requests failing

```bash
curl -s localhost:8000/metrics | grep aether_errors_total
```

`kind` label tells you where to look:

- `not_ready` — requests arriving before the model loaded. Check readiness; the
  Service should not be routing to an unready pod, so also check the probe.
- `batch` — the batcher rejected work. Usually shutdown in progress, or a failure
  inside the model call. Check logs for `batch_failed`.
- `swap` — a bad version tag was pushed. Harmless to the running service by
  design; the live model is untouched.
- `startup_load` — see [No ready replicas](#no-ready-replicas).
- `unhandled` — a real bug. Get the traceback from the logs.

## High latency

**Alert:** `AetherHighLatency` — p95 above 5s

Work through in order:

1. **Are we at max replicas?**
   ```bash
   kubectl get hpa aether
   ```
   `REPLICAS` at `maxReplicas` with queue depth still climbing means the cluster
   is out of capacity. Raise `autoscaling.maxReplicas`, or add nodes.

2. **Is the autoscaler even working?**
   ```bash
   kubectl describe hpa aether | tail -20
   ```
   `unable to get metric aether_queue_depth` means prometheus-adapter is missing
   or misconfigured, and the HPA is silently running on CPU alone. See
   [deployment.md](deployment.md#autoscaling).

3. **Are clients asking for more work?** Latency scales linearly with `steps`.
   A client switching from 32 to 512 steps makes every request 16x more
   expensive; that is not a regression, it is the NFE knob.
   ```bash
   curl -s localhost:8000/metrics | grep aether_nfe_total
   ```

4. **Is batching working?** See below.

## Queue backlog

**Alert:** `AetherQueueBacklog` — average queue depth above 10

Requests are arriving faster than they are served. Either scale out (the HPA
should be doing this — check it is not stuck) or reduce per-request cost by
lowering the default `steps`.

If queue depth is high while `aether_in_flight_batch_size` is *low*, batching is
not merging requests — go to the next section.

## Batching ineffective

**Alert:** `AetherBatchingIneffective` — requests per batch below 1.2

Informational, not a page. Under concurrent traffic the ratio should exceed 1.

```bash
curl -s localhost:8000/metrics | grep -E 'aether_(requests|batches)_total'
```

Causes, in order of likelihood:

- **`max_wait_ms` too low.** Requests arrive but the window closes before a second
  one lands. Try 50ms.
- **Incompatible parameters.** Only requests with identical sampler, steps,
  length, and temperature can share a forward pass. Clients spreading across many
  parameter combinations defeat batching by construction.
- **Genuinely low traffic.** At one request per second there is nothing to batch,
  and that is fine. The alert requires >1 req/s to fire for this reason.

## Bad rollout

Symptoms: new pods crash-looping, or `helm upgrade --wait` timing out.

```bash
kubectl rollout status deploy/aether        # what is it waiting on?
kubectl describe pod -l app.kubernetes.io/name=aether | tail -30
kubectl get events --sort-by=.lastTimestamp | tail -20
```

`maxUnavailable: 0` means old pods stay up until new ones pass readiness — so a
broken image degrades the *deploy*, not the service. You have time to think.

```bash
helm rollback aether
```

`ImagePullBackOff` — the tag does not exist or the registry needs credentials:
```bash
kubectl describe pod <pod> | grep -A5 Events
```

`CrashLoopBackOff` — the process is dying at startup. Get the logs from the
*previous* container instance, since the current one may not have logged yet:
```bash
kubectl logs <pod> --previous
```

## Draining a node / graceful shutdown

`terminationGracePeriodSeconds: 60` exceeds the server's 30-second drain budget.
On SIGTERM the server stops accepting new work, finishes in-flight requests, and
explicitly fails anything still queued rather than leaving client futures
unresolved — an unresolved future hangs the caller until *its* timeout, which
looks like a much worse outage than a clean error.

```bash
kubectl drain <node> --ignore-daemonsets
```

Requests longer than 60s (very high `steps`) will still be cut off. If that
matters, raise the grace period above your worst-case generation time.

## Escalation

Before escalating, capture:

```bash
kubectl get pods,hpa,events -l app.kubernetes.io/name=aether > /tmp/state.txt
kubectl logs -l app.kubernetes.io/name=aether --tail=500 > /tmp/logs.txt
curl -s localhost:8000/metrics > /tmp/metrics.txt
curl -s localhost:8000/model > /tmp/model.json
```

`model.json` identifies exactly which checkpoint was serving, which is the first
thing anyone will ask.
