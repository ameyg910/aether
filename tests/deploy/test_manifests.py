"""Static validation of the deployment artifacts.

There is no Kubernetes cluster in CI's unit-test job, so these tests check the
things that are checkable without one -- and they are exactly the things that
break silently in practice:

- a dashboard panel querying a metric that no longer exists renders as
  "No data" rather than as an error, so nobody notices until an incident;
- a Helm template referencing a mistyped `.Values` path renders as an empty
  string, producing a manifest that applies cleanly and behaves wrongly;
- swapping the liveness and readiness endpoints produces a service that
  crash-loops under load instead of shedding traffic.

The `kind` job in CI covers what genuinely needs a cluster.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
import yaml

from aether.serve.metrics import exported_metric_names

REPO = Path(__file__).resolve().parents[2]
DEPLOY = REPO / "deploy"
CHART = DEPLOY / "helm" / "aether"

# Metrics that come from Kubernetes/kube-state, not from our application.
_EXTERNAL_METRICS = {"kube_deployment_status_replicas", "container_cpu_usage_seconds_total"}


def _metric_refs(text: str) -> set[str]:
    """Every `aether_*` identifier appearing in a query string."""
    return set(re.findall(r"\baether_[a-z0-9_]+\b", text))


# -- metric name drift --------------------------------------------------------
def test_dashboard_queries_only_reference_real_metrics() -> None:
    dashboard = json.loads((DEPLOY / "grafana" / "dashboards" / "aether-serving.json").read_text())
    exported = exported_metric_names()
    referenced: set[str] = set()
    for panel in dashboard["panels"]:
        for target in panel.get("targets", []):
            referenced |= _metric_refs(target.get("expr", ""))

    unknown = referenced - exported - _EXTERNAL_METRICS
    assert not unknown, f"dashboard references metrics the service does not export: {unknown}"


def test_alert_rules_only_reference_real_metrics() -> None:
    rules = yaml.safe_load((DEPLOY / "grafana" / "alerts" / "aether-rules.yaml").read_text())
    exported = exported_metric_names()
    referenced: set[str] = set()
    for group in rules["groups"]:
        for rule in group["rules"]:
            referenced |= _metric_refs(rule["expr"])

    unknown = referenced - exported - _EXTERNAL_METRICS
    assert not unknown, f"alert rules reference unknown metrics: {unknown}"


def test_hpa_scales_on_a_metric_the_service_exports() -> None:
    hpa = yaml.safe_load((DEPLOY / "k8s" / "hpa.yaml").read_text())
    pod_metrics = [m for m in hpa["spec"]["metrics"] if m["type"] == "Pods"]
    assert pod_metrics, "HPA should scale on a custom pod metric, not CPU alone"
    for metric in pod_metrics:
        name = metric["pods"]["metric"]["name"]
        assert name in exported_metric_names(), f"HPA scales on unexported metric {name!r}"


def test_every_alert_has_a_runbook_link() -> None:
    # An alert without a runbook is a 3am puzzle.
    rules = yaml.safe_load((DEPLOY / "grafana" / "alerts" / "aether-rules.yaml").read_text())
    for group in rules["groups"]:
        for rule in group["rules"]:
            if rule.get("labels", {}).get("severity") in ("critical", "warning"):
                assert "runbook_url" in rule["annotations"], f"{rule['alert']} has no runbook"


# -- probe wiring -------------------------------------------------------------
def test_liveness_and_readiness_target_different_endpoints() -> None:
    """Swapping these is a classic outage.

    Liveness failing restarts the container; readiness failing only pulls it from
    the Service. A pod still loading a model must fail readiness and *pass*
    liveness, or Kubernetes kills it before it finishes loading, forever.
    """
    deployment = yaml.safe_load((DEPLOY / "k8s" / "deployment.yaml").read_text())
    container = deployment["spec"]["template"]["spec"]["containers"][0]
    assert container["livenessProbe"]["httpGet"]["path"] == "/health"
    assert container["readinessProbe"]["httpGet"]["path"] == "/ready"


def test_termination_grace_exceeds_server_shutdown_budget() -> None:
    # uvicorn is given 30s to drain in-flight requests; the pod must outlive that
    # or Kubernetes SIGKILLs mid-generation.
    deployment = yaml.safe_load((DEPLOY / "k8s" / "deployment.yaml").read_text())
    assert deployment["spec"]["template"]["spec"]["terminationGracePeriodSeconds"] > 30


def test_rollout_keeps_full_capacity() -> None:
    deployment = yaml.safe_load((DEPLOY / "k8s" / "deployment.yaml").read_text())
    rolling = deployment["spec"]["strategy"]["rollingUpdate"]
    assert rolling["maxUnavailable"] == 0


def test_container_runs_unprivileged() -> None:
    deployment = yaml.safe_load((DEPLOY / "k8s" / "deployment.yaml").read_text())
    pod_spec = deployment["spec"]["template"]["spec"]
    assert pod_spec["securityContext"]["runAsNonRoot"] is True
    container = pod_spec["containers"][0]
    assert container["securityContext"]["allowPrivilegeEscalation"] is False


# -- helm chart ---------------------------------------------------------------
def _chart_templates() -> list[Path]:
    return sorted((CHART / "templates").glob("*.yaml"))


def test_chart_metadata_is_well_formed() -> None:
    chart = yaml.safe_load((CHART / "Chart.yaml").read_text())
    assert chart["apiVersion"] == "v2"
    assert re.fullmatch(r"\d+\.\d+\.\d+", chart["version"])
    assert re.fullmatch(r"\d+\.\d+\.\d+", str(chart["appVersion"]))


def test_template_delimiters_are_balanced() -> None:
    for path in [*_chart_templates(), CHART / "templates" / "_helpers.tpl"]:
        text = path.read_text()
        assert text.count("{{") == text.count("}}"), f"unbalanced delimiters in {path.name}"


def _values_paths(node: object, prefix: str = "") -> set[str]:
    paths: set[str] = set()
    if isinstance(node, dict):
        for key, value in node.items():
            here = f"{prefix}.{key}" if prefix else key
            paths.add(here)
            paths |= _values_paths(value, here)
    return paths


def test_templates_only_reference_declared_values() -> None:
    """A typo'd `.Values` path renders empty rather than failing.

    That produces a manifest which applies cleanly and misbehaves at runtime --
    the worst possible failure mode, so it is worth catching statically.
    """
    values = yaml.safe_load((CHART / "values.yaml").read_text())
    declared = _values_paths(values)

    referenced: set[str] = set()
    for path in _chart_templates():
        referenced |= set(re.findall(r"\.Values\.([A-Za-z0-9_.]+)", path.read_text()))

    unknown = {r for r in referenced if r.rstrip(".") not in declared}
    assert not unknown, f"templates reference undeclared values: {sorted(unknown)}"


@pytest.mark.parametrize(
    "name",
    ["deployment.yaml", "service.yaml", "configmap.yaml", "hpa.yaml"],
)
def test_expected_templates_exist(name: str) -> None:
    assert (CHART / "templates" / name).exists()


def test_chart_values_keep_autoscaling_bounds_sane() -> None:
    values = yaml.safe_load((CHART / "values.yaml").read_text())
    auto = values["autoscaling"]
    assert auto["minReplicas"] >= 1
    assert auto["maxReplicas"] > auto["minReplicas"]


# -- plain manifests ----------------------------------------------------------
@pytest.mark.parametrize(
    "name",
    ["configmap.yaml", "deployment.yaml", "service.yaml", "hpa.yaml", "loadgen.yaml"],
)
def test_k8s_manifests_parse(name: str) -> None:
    doc = yaml.safe_load((DEPLOY / "k8s" / name).read_text())
    assert doc["apiVersion"]
    assert doc["kind"]
    assert doc["metadata"]["name"]


def test_service_selector_matches_deployment_labels() -> None:
    # A mismatch here yields a Service with no endpoints: everything looks healthy
    # and nothing can reach the pods.
    deployment = yaml.safe_load((DEPLOY / "k8s" / "deployment.yaml").read_text())
    service = yaml.safe_load((DEPLOY / "k8s" / "service.yaml").read_text())
    pod_labels = deployment["spec"]["template"]["metadata"]["labels"]
    for key, value in service["spec"]["selector"].items():
        assert pod_labels.get(key) == value, f"service selector {key}={value} matches no pod label"


def test_alert_runbook_links_resolve_to_real_sections() -> None:
    """A runbook_url pointing at a section that does not exist is worse than none.

    It looks like help and delivers a 404 at the moment someone is under pressure.
    """
    rules = yaml.safe_load((DEPLOY / "grafana" / "alerts" / "aether-rules.yaml").read_text())
    runbook = (REPO / "docs" / "runbook.md").read_text()
    anchors = {
        re.sub(r"[^a-z0-9 -]", "", heading.lower()).replace(" ", "-")
        for heading in re.findall(r"^## (.+)$", runbook, re.M)
    }

    broken = []
    for group in rules["groups"]:
        for rule in group["rules"]:
            url = rule["annotations"].get("runbook_url", "")
            if "#" in url and url.split("#")[1] not in anchors:
                broken.append((rule["alert"], url.split("#")[1]))
    assert not broken, f"alerts link to missing runbook sections: {broken}"
