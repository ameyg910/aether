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
