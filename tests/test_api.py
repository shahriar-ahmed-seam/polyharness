"""Tests for PolyHarness FastAPI REST API endpoints."""

from fastapi.testclient import TestClient

from polyharness.eval.benchmarks import get_standard_benchmarks
from polyharness.server.app import app

client = TestClient(app)


def test_health():
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert "openai" in data["supported_harnesses"]


def test_adapters_list():
    res = client.get("/api/adapters")
    assert res.status_code == 200
    data = res.json()
    assert len(data) >= 6


def test_benchmarks_endpoint():
    res = client.get("/api/benchmarks")
    assert res.status_code == 200
    data = res.json()
    assert len(data) >= 2


def test_validate_endpoint():
    traj = get_standard_benchmarks()[0].model_dump()
    res = client.post("/api/trajectories/validate", json={"trajectory": traj})
    assert res.status_code == 200
    data = res.json()
    assert data["is_valid"] is True
    assert "quality_score" in data


def test_render_endpoint():
    traj = get_standard_benchmarks()[0].model_dump()
    res = client.post(
        "/api/trajectories/render",
        json={"trajectory": traj, "target_harness": "openai"},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["target_harness"] == "openai"
    assert "messages" in data["rendered"]


def test_augment_endpoint():
    traj = get_standard_benchmarks()[0].model_dump()
    res = client.post(
        "/api/trajectories/augment",
        json={
            "trajectory": traj,
            "enable_syntax_perturbation": True,
            "enable_multi_observation": True,
            "enable_cascade_guard": True,
        },
    )
    assert res.status_code == 200
    data = res.json()
    assert data["variants_count"] == 3


def test_compile_endpoint():
    traj = get_standard_benchmarks()[0].model_dump()
    res = client.post(
        "/api/trajectories/compile",
        json={
            "trajectories": [traj],
            "target_harness": "openai",
            "is_mixture": True,
        },
    )
    assert res.status_code == 200
    data = res.json()
    assert data["record_count"] == 1
    assert data["mode"] == "mixture"


def test_eval_run_endpoint():
    res = client.post(
        "/api/eval/run",
        json={
            "model_profile": "polyharness_generalist",
            "native_harness": "openai",
        },
    )
    assert res.status_code == 200
    data = res.json()
    assert data["harness_overfitting_coefficient"] == 0.0
    assert data["is_production_safe"] is True
