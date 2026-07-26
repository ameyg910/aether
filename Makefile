.RECIPEPREFIX := >
.PHONY: help install lint format type test all demo plot config data data-debug overfit train train-debug

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
