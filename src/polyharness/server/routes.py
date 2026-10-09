"""PolyHarness FastAPI REST API routes."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from polyharness import __version__
from polyharness.adapters import ADAPTER_REGISTRY, get_adapter, list_adapters
from polyharness.eval.benchmarks import get_standard_benchmarks
from polyharness.eval.metrics import OverfittingAuditReport
from polyharness.eval.runner import MockModelProvider, MultiHarnessEvaluator
from polyharness.schema.adp import Trajectory
from polyharness.schema.validator import TrajectoryDiagnosticReport, TrajectoryValidator
from polyharness.synthesis.cascade_guard import CascadeGuard
from polyharness.synthesis.compiler import DatasetCompiler
from polyharness.synthesis.observation import ObservationTransformer
from polyharness.synthesis.perturbation import SyntaxPerturber

router = APIRouter(prefix="/api")


class ValidateRequest(BaseModel):
    trajectory: dict[str, Any]


class RenderRequest(BaseModel):
    trajectory: dict[str, Any]
    target_harness: str = "openai"


class AugmentRequest(BaseModel):
    trajectory: dict[str, Any]
    enable_syntax_perturbation: bool = True
    enable_multi_observation: bool = True
    enable_cascade_guard: bool = True


class CompileRequest(BaseModel):
    trajectories: list[dict[str, Any]]
    target_harness: str = "openai"
    is_mixture: bool = False
    mixture_weights: dict[str, float] | None = None


class EvalRequest(BaseModel):
    model_profile: str = "overfitted_single_harness"
    native_harness: str = "openai"
    test_harnesses: list[str] | None = None


@router.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "PolyHarness API & Studio",
        "version": __version__,
        "supported_harnesses": list(ADAPTER_REGISTRY.keys()),
    }


@router.get("/adapters")
def get_adapters_list():
    return list_adapters()


@router.get("/benchmarks")
def get_benchmarks():
    return [t.model_dump() for t in get_standard_benchmarks()]


@router.post("/trajectories/validate", response_model=TrajectoryDiagnosticReport)
def validate_trajectory(req: ValidateRequest):
    try:
        traj = Trajectory.model_validate(req.trajectory)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Schema validation error: {e}")
    return TrajectoryValidator.validate(traj)


@router.post("/trajectories/render")
def render_trajectory(req: RenderRequest):
    try:
        traj = Trajectory.model_validate(req.trajectory)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Schema validation error: {e}")

    try:
        adapter = get_adapter(req.target_harness)
        rendered = adapter.render(traj)
        return {
            "target_harness": req.target_harness,
            "description": adapter.description,
            "rendered": rendered,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/trajectories/augment")
def augment_trajectory(req: AugmentRequest):
    try:
        traj = Trajectory.model_validate(req.trajectory)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Schema validation error: {e}")

    results = {}
    if req.enable_syntax_perturbation:
        perturber = SyntaxPerturber(seed=42)
        results["syntax_perturbed"] = perturber.perturb_trajectory(traj).model_dump()

    if req.enable_multi_observation:
        transformer = ObservationTransformer()
        results["multi_observation"] = transformer.populate_multi_representations(traj).model_dump()

    if req.enable_cascade_guard:
        guard = CascadeGuard(seed=42)
        results["cascade_guarded"] = guard.inject_recovery_turn(traj).model_dump()

    return {
        "original_id": traj.id,
        "variants_count": len(results),
        "variants": results,
    }


@router.post("/trajectories/compile")
def compile_sft(req: CompileRequest):
    try:
        trajs = [Trajectory.model_validate(t) for t in req.trajectories]
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Schema validation error: {e}")

    compiler = DatasetCompiler(seed=42)
    if req.is_mixture:
        records = compiler.compile_multi_harness_mixture(
            trajs, harness_distribution=req.mixture_weights
        )
    else:
        records = compiler.compile_single_harness(trajs, target_harness=req.target_harness)

    return {
        "record_count": len(records),
        "mode": "mixture" if req.is_mixture else req.target_harness,
        "records": records,
    }


@router.post("/eval/run", response_model=OverfittingAuditReport)
async def run_evaluation(req: EvalRequest):
    benchmarks = get_standard_benchmarks()
    provider = MockModelProvider(
        profile=req.model_profile, native_harness=req.native_harness
    )
    evaluator = MultiHarnessEvaluator(
        model_provider=provider,
        native_harness=req.native_harness,
        test_harnesses=req.test_harnesses or ["openai", "hermes", "anthropic", "react"],
    )
    report = await evaluator.evaluate_suite(benchmarks, model_id=f"model-{req.model_profile}")
    return report
