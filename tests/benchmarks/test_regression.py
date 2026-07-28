"""The pinned regression benchmark, run as part of the normal test suite.

Also exercised standalone in CI (`python benchmarks/regression.py`) so the same
thresholds gate a pull request either way.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# benchmarks/ is a scripts directory, not an installed package.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmarks"))

from regression import THRESHOLDS, check_thresholds, run_regression


@pytest.fixture(scope="module")
def metrics() -> dict:  # type: ignore[type-arg]
    return run_regression()


def test_no_threshold_breaches(metrics: dict) -> None:  # type: ignore[type-arg]
    breaches = check_thresholds(metrics)
    assert not breaches, "regression thresholds breached:\n" + "\n".join(breaches)


def test_every_threshold_has_a_metric(metrics: dict) -> None:  # type: ignore[type-arg]
    # Guards against a threshold silently going unchecked after a rename.
    missing = [k for k in THRESHOLDS if k not in metrics]
    assert not missing, f"thresholds with no corresponding metric: {missing}"


def test_untrained_likelihood_matches_uniform_baseline(metrics: dict) -> None:  # type: ignore[type-arg]
    # The strongest single check on the NELBO scale: a random model must score
    # close to ln(vocab - 1) nats per token.
    uniform = metrics["_reference"]["uniform_nats"]
    assert metrics["nll_nats_per_token"] == pytest.approx(uniform, abs=1.0)


def test_breach_detection_actually_fires() -> None:
    # A benchmark that cannot fail is not a benchmark.
    assert check_thresholds({**dict.fromkeys(THRESHOLDS, 0.0), "self_mauve": 0.0})


def test_is_deterministic() -> None:
    a = run_regression()
    b = run_regression()
    assert a["nll_nats_per_token"] == pytest.approx(b["nll_nats_per_token"], rel=1e-9)
    assert a["ancestral_entropy"] == pytest.approx(b["ancestral_entropy"], rel=1e-9)
