# Contributing to Aether

Thanks for your interest in contributing!

## Development setup

```bash
python -m venv .venv && source .venv/bin/activate
make install    # editable install + pre-commit hooks
make all        # lint, type-check, test
```

## Workflow

- Branch from `main`: `feat/...`, `fix/...`, `docs/...`, `ci/...`, `refactor/...`.
- Use [Conventional Commits](https://www.conventionalcommits.org/) for commit and
  PR titles (drives the changelog and semantic versioning).
- Every PR must pass `make all` and CI before merge. PRs are squash-merged.
- Add or update tests for any behavior change.

## Code standards

- Fully typed (`mypy --strict` on `src/`).
- Formatted and linted with `ruff`.
- Public functions carry docstrings.
- Configuration is data, not code: add options to `configs/` and the schemas in
  `src/aether/config/schemas.py` rather than hardcoding values.

## Container workflow

The whole stack runs with one command — no local Python, CUDA, or dependency setup:

```bash
docker compose up --build
```

| service | URL |
| --- | --- |
| inference API (OpenAPI docs) | http://localhost:8000/docs |
| Prometheus | http://localhost:9090 |
| Grafana (admin/admin) | http://localhost:3000 |

Point the server at a checkpoint with `MODEL_VERSION` — a path under `./runs`
(bind-mounted read-only) or a Hub tag:

```bash
MODEL_VERSION=runs/my-run docker compose up
MODEL_VERSION=hf:ameyg910/aether-55m@v0.1.0 docker compose up
```

Build images individually:

```bash
make docker-build                     # both images, CPU wheels
docker build -f docker/Dockerfile.serve \
  --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu124 \
  -t aether-serve:gpu .               # GPU serving
```

Both images are multi-stage and run as a non-root user; the build toolchain stays
in the builder stage and never reaches the published layer.

## Dependencies

`pyproject.toml` declares what the project accepts; `uv.lock` records what a given
build actually used. Refresh the lockfile whenever dependencies change and commit
it alongside:

```bash
make lock            # uv lock
uv sync --extra dev --extra serve
```

## Commits and releases

Commit subjects follow [Conventional Commits](https://www.conventionalcommits.org/)
(`feat(serve): ...`, `fix(train): ...`, `docs: ...`). GitHub generates release
notes by grouping these prefixes, so the message you write becomes the release
note someone reads. See [docs/releasing.md](docs/releasing.md) for the versioning
policy and the release procedure.
