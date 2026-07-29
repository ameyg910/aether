.RECIPEPREFIX := >
.PHONY: help install lint format type test all demo plot config data data-debug overfit train train-debug train-ddp scaling eval bench bench-regression serve loadtest docker-build docker-serve compose-up compose-down lock

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
