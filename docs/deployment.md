# Deployment

From a clean machine to a running, autoscaling service.

## Local cluster

```bash
# k3d (lighter) or kind — either works
k3d cluster create aether --agents 2 -p "8080:80@loadbalancer"
# or
kind create cluster --name aether

kubectl cluster-info
```

## Install

```bash
helm install aether ./deploy/helm/aether \
  --set model.version=hf:ameyg910/aether-55m@v0.8.0

kubectl rollout status deploy/aether
kubectl port-forward svc/aether 8000:8000
curl localhost:8000/ready
```

Using a locally-built image instead of the registry:

```bash
docker build -f docker/Dockerfile.serve -t aether-serve:dev .
kind load docker-image aether-serve:dev --name aether     # or: k3d image import

helm install aether ./deploy/helm/aether \
  --set image.repository=aether-serve \
  --set image.tag=dev \
  --set image.pullPolicy=Never
```

**With no `model.version` the server starts unready on purpose.** `/health`
answers, `/ready` returns 503, and `/admin/swap` can push a working checkpoint.
Crash-looping on a bad version tag would be strictly worse — you would lose the
ability to inspect or fix the running pod.

## What gets created

| object | purpose |
| --- | --- |
| Deployment | the pods, with liveness/readiness/startup probes and resource requests |
| Service (ClusterIP) | stable internal address; selects pods by label |
| ConfigMap | batch size, wait window, device, model version |
| HPA | scales 2–10 replicas on queue depth |
| ServiceMonitor | Prometheus Operator scrape target *(optional)* |
| PrometheusRule | alert rules *(optional)* |
| Ingress | external exposure *(optional)* |

**Deployment vs Service vs Ingress**, since these get conflated: the Deployment
manages *pods* (how many, which image, when to restart). The Service gives that
changing set of pods one stable in-cluster address and load-balances across them.
The Ingress maps outside HTTP traffic to a Service. They are separate because
they change for different reasons — scaling the Deployment must not change the
address, and changing the hostname must not restart the pods.

## Autoscaling

The HPA scales on `aether_queue_depth`, not CPU.

**Why not CPU.** Inference runs in a bounded worker-thread pool. Once that pool is
busy, CPU flattens near a ceiling while requests pile up in the batcher queue — so
CPU saturates *before* it reflects how far behind the service is, and it cannot
distinguish "comfortably busy" from "drowning". On GPU it is worse: the GPU is the
bottleneck and CPU barely moves at all. Queue depth measures the thing users
actually feel — how many requests are waiting — and it responds immediately in
both directions.

Custom metrics need `prometheus-adapter`:

```bash
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm install prometheus-adapter prometheus-community/prometheus-adapter \
  -f deploy/prometheus/adapter-values.yaml

# verify the metric is exposed through the custom metrics API
kubectl get --raw \
  "/apis/custom.metrics.k8s.io/v1beta1/namespaces/default/pods/*/aether_queue_depth" | jq
```

Without the adapter the HPA falls back to its CPU target — still functional, just
less precise. The chart handles both:

```bash
helm upgrade aether ./deploy/helm/aether \
  --set autoscaling.queueDepthMetricEnabled=false
```

Scale-up is deliberately fast (30s stabilization, up to doubling) and scale-down
slow (300s, one pod at a time). Asymmetric because the costs are asymmetric: an
extra pod is cheap, while scaling down into a load trough that immediately returns
means every new pod pays a cold model load again.

### Watching it work

```bash
kubectl apply -f deploy/k8s/loadgen.yaml
kubectl get hpa aether --watch
kubectl get pods -w
kubectl delete -f deploy/k8s/loadgen.yaml
```

Queue depth climbs, the HPA adds replicas, queue depth and latency recover. The
"Autoscaling: replicas vs queue depth" panel in the dashboard shows all three
together.

## Observability

```bash
helm install prometheus prometheus-community/kube-prometheus-stack -n monitoring --create-namespace

helm upgrade aether ./deploy/helm/aether \
  --set metrics.serviceMonitor.enabled=true \
  --set metrics.prometheusRule.enabled=true
```

Import `deploy/grafana/dashboards/aether-serving.json`. Alert rules live in
`deploy/grafana/alerts/aether-rules.yaml` and are also shipped by the chart as a
`PrometheusRule`.

Every alert fires on a **symptom**, not a cause. "CPU is high" is not an incident;
"requests are failing" and "requests are slow" are. Cause-based alerts page people
for conditions that are often entirely normal, which is how alert fatigue starts
and how the alerts that matter end up ignored.

## Rolling updates

```bash
helm upgrade aether ./deploy/helm/aether --set image.tag=0.9.0
kubectl rollout status deploy/aether
kubectl rollout undo deploy/aether     # if it goes wrong
```

`maxUnavailable: 0` means new pods must pass readiness before old ones are
removed, so capacity never dips during a deploy. `terminationGracePeriodSeconds:
60` exceeds the server's own 30-second drain budget, so in-flight generation
finishes instead of being killed mid-request.

## Chart versioning

`Chart.version` and `appVersion` move independently: a values-only change bumps
the chart alone, a new image with no template change bumps `appVersion` alone.
Both follow semver. See [releasing.md](releasing.md).

## Verification

```bash
pytest tests/deploy            # static checks: metric names, values paths, probes
helm lint deploy/helm/aether
helm template aether deploy/helm/aether | kubectl apply --dry-run=client -f -
```

CI runs `helm lint`, renders the chart under several value combinations,
schema-validates the output with `kubeconform`, and then installs it on a real
`kind` cluster and probes the endpoints.
