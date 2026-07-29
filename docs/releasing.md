# Releasing

## Versioning policy

Aether follows [Semantic Versioning](https://semver.org/). Given `MAJOR.MINOR.PATCH`:

- **MAJOR** — a breaking change to a public surface: the HTTP API contract, the
  CLI entry points and their Hydra config keys, the checkpoint format, or the
  Python API under `aether.*`.
- **MINOR** — new capability, backwards compatible. A new sampler, a new metric, a
  new endpoint, a new config field with a default.
- **PATCH** — bug fixes and internal changes with no surface impact.

Pre-1.0, MINOR carries what would otherwise be MAJOR: the interfaces are still
moving. The 1.0 release is the commitment that they have stopped.

**What counts as a public surface** is worth being explicit about, because it is
where semver arguments happen:

| surface | covered? |
| --- | --- |
| HTTP request/response schemas | yes |
| Hydra config keys and their defaults | yes |
| CLI entry points (`aether-train`, `aether-serve`, `aether-eval`) | yes |
| Checkpoint format and what it must contain | yes |
| `aether.*` importable API | yes |
| Prometheus metric names and labels | yes — dashboards and alerts depend on them |
| Log message wording | no |
| Internal module layout (`_`-prefixed, `aether.train.mfu` internals) | no |
| Benchmark numbers | no |

## Conventional commits

Commit subjects follow [Conventional Commits](https://www.conventionalcommits.org/):

```
feat(serve): add confidence sampler to the generate endpoint
fix(train): scope checkpoint pruning to the current run
docs: evaluation protocol
build(docker): multi-stage serve and train images
ci: matrix lint/type/test with coverage
refactor(train): extract precision handling
test(serve): cover the batcher shutdown path
```

Types in use: `feat`, `fix`, `docs`, `build`, `ci`, `refactor`, `test`, `perf`,
`chore`. A `!` after the scope (`feat(api)!:`) or a `BREAKING CHANGE:` footer marks
an incompatible change.

This is not decoration: GitHub generates release notes by grouping these prefixes,
so a commit message written carelessly becomes a release note written carelessly.

## Cutting a release

1. Update `CHANGELOG.md` — move `[Unreleased]` entries under the new version with
   a date. Generated notes list every commit; the changelog says which ones matter.
2. Bump `version` in `pyproject.toml`.
3. Commit: `chore(release): v0.8.0`.
4. Tag and push:

```bash
git tag -a v0.8.0 -m "v0.8.0"
git push origin main --tags
```

The tag triggers `.github/workflows/release.yml`, which:

1. **Re-runs the full gate** — ruff, mypy, pytest, and the regression benchmark.
   Tags can be pushed to any commit, including one CI never saw, so publishing
   without re-verifying would let an untested artifact into the registry.
2. Builds both images and pushes them to GHCR with semver tags.
3. Drafts a GitHub release with generated notes.

Images land at:

```
ghcr.io/ameyg910/aether/aether-serve:0.8.0
ghcr.io/ameyg910/aether/aether-serve:0.8
ghcr.io/ameyg910/aether/aether-serve:0
ghcr.io/ameyg910/aether/aether-serve:latest
ghcr.io/ameyg910/aether/aether-train:0.8.0
```

The cascade of tags lets a consumer choose how tightly to pin: `:0.8.0` never
moves, `:0.8` picks up patches, `:0` picks up features. Production should pin the
exact version; `latest` is for trying things out.

## Dependency locking

`uv.lock` pins the full transitive dependency graph with hashes. Refresh it
whenever `pyproject.toml` changes:

```bash
uv lock             # re-resolve
uv sync --extra dev --extra serve   # install exactly what is locked
```

Commit the lockfile with the change that caused it. `pyproject.toml` declares what
the project *accepts*; `uv.lock` records what a particular build actually *used* —
the difference is what makes a build reproducible six months later.
