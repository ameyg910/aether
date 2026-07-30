.RECIPEPREFIX := >
.PHONY: help install lint format type test all demo plot config data data-debug overfit train train-debug train-ddp scaling eval bench bench-regression serve loadtest docker-build docker-serve compose-up compose-down lock helm-lint helm-render k8s-deploy k8s-loadtest k8s-clean train-toy docs docs-serve

help:
> @echo "Targets: install lint format type test all demo plot config"

install:
> pip install -e ".[dev]"
> pre-commit install

lint:
> ruff check .

format:
> ruff format .

type:
> mypy

test:
> pytest --cov=aether --cov-report=term-missing

all: lint type test

demo:
> python -m aether.diffusion.forward --sentence "the cat sat on the mat"

plot:
> python scripts/visualize_masking.py

config:
> python scripts/show_config.py

data:
> aether-prepare data=wikitext103

data-debug:
> aether-prepare data=local_debug

overfit:
> python -m aether.train.overfit

train:
> aether-train

train-debug:
> aether-train train=debug data=local_debug tracking.backend=jsonl

train-ddp:
> NPROC=$(NPROC) scripts/launch/torchrun_local.sh

scaling:
> python scripts/scaling_plot.py $(RUNS) --out docs/assets/scaling.png

eval:
> aether-eval $(ARGS)

bench:
> python benchmarks/nfe_quality.py $(ARGS)

bench-regression:
> python benchmarks/regression.py --json benchmarks/results/regression.json

serve:
> aether-serve $(ARGS)

loadtest:
> locust -f loadtest/locustfile.py --host $(HOST) --headless -u $(USERS) -r $(USERS) -t 30s

docker-build:
> docker build -f docker/Dockerfile.serve -t aether-serve:dev .
> docker build -f docker/Dockerfile.train -t aether-train:dev --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cpu .

docker-serve:
> docker run --rm -p 8000:8000 -v $(PWD)/runs:/app/runs:ro aether-serve:dev serve.model_version=$(MODEL)

compose-up:
> docker compose up --build

compose-down:
> docker compose down -v

lock:
> uv lock

train-toy:
> @# End-to-end reproduction from a clean clone: no GPU, no downloads.
> @# Proves the pipeline runs, not that the model is good.
> @echo "==> preparing a small offline corpus"
> aether-prepare data=local_debug
> @echo "==> training"
> aether-train train=debug data=local_debug tracking.backend=jsonl \
>   train.run_name=toy train.max_steps=200 train.out_dir=runs
> @echo "==> evaluating"
> aether-eval eval=fast data=local_debug \
>   eval.checkpoint=runs/toy/checkpoints/latest.pt eval.run_name=toy
> @echo ""
> @echo "done. checkpoint: runs/toy/checkpoints/latest.pt"
> @echo "      metrics:    runs/toy/metrics.jsonl"
> @echo "      eval:       benchmarks/results/toy.json"

docs:
> mkdocs build --strict

docs-serve:
> mkdocs serve

helm-lint:
> helm lint deploy/helm/aether

helm-render:
> helm template aether deploy/helm/aether $(ARGS)

k8s-deploy:
> helm upgrade --install aether ./deploy/helm/aether --wait --timeout 5m $(ARGS)

k8s-loadtest:
> kubectl apply -f deploy/k8s/loadgen.yaml
> kubectl get hpa aether --watch

k8s-clean:
> kubectl delete -f deploy/k8s/loadgen.yaml --ignore-not-found
> helm uninstall aether
