"""Multi-Harness Evaluation Runner & Model Providers."""

from __future__ import annotations

import asyncio
import random
from datetime import datetime, timezone
from typing import Any

from polyharness.adapters import get_adapter
from polyharness.eval.metrics import (
    HarnessResult,
    OverfittingAuditReport,
    compute_overfitting_metrics,
)
from polyharness.schema.adp import Trajectory


class BaseModelProvider:
    """Interface for querying LLMs during multi-harness evaluation."""

    async def generate_response(
        self, prompt_payload: dict[str, Any], harness_id: str
    ) -> dict[str, Any]:
        raise NotImplementedError


class MockModelProvider(BaseModelProvider):
    """Simulates agent model behavior to benchmark overfitting detection.

    Profiles:
    - 'overfitted_single_harness': Models fine-tuned on 1 harness. High accuracy on native harness,
      collapses on unseen syntax, observation changes, and triggers cascading errors.
    - 'polyharness_generalist': Models trained on multi-harness ADP mixtures. Resilient across all harnesses.
    """

    def __init__(self, profile: str = "overfitted_single_harness", native_harness: str = "openai"):
        self.profile = profile
        self.native_harness = native_harness

    async def generate_response(
        self, prompt_payload: dict[str, Any], harness_id: str
    ) -> dict[str, Any]:
        await asyncio.sleep(0.01)  # tiny simulated latency

        if self.profile == "polyharness_generalist":
            # Resilient cross-harness behavior
            return {
                "success": True,
                "malformed_syntax": False,
                "unvisited_state_crash": False,
                "cascading_failure": False,
            }

        # Profile: 'overfitted_single_harness'
        if harness_id == self.native_harness:
            return {
                "success": True,
                "malformed_syntax": False,
                "unvisited_state_crash": False,
                "cascading_failure": False,
            }
        else:
            # Overfitted model fails on unseen harnesses (Whiteboard failure modes)
            fail_type = random.choice(["syntax", "obs_crash", "cascade"])
            return {
                "success": False,
                "malformed_syntax": (fail_type == "syntax"),
                "unvisited_state_crash": (fail_type == "obs_crash"),
                "cascading_failure": (fail_type == "cascade"),
            }


class MultiHarnessEvaluator:
    """Evaluates agent models across multiple harnesses and computes the Harness Overfitting Coefficient."""

    def __init__(
        self,
        model_provider: BaseModelProvider,
        native_harness: str = "openai",
        test_harnesses: list[str] | None = None,
    ):
        self.model_provider = model_provider
        self.native_harness = native_harness
        self.test_harnesses = test_harnesses or ["openai", "hermes", "anthropic", "react"]

    async def evaluate_suite(
        self,
        benchmark_trajectories: list[Trajectory],
        model_id: str = "eval-model",
    ) -> OverfittingAuditReport:
        """Run all benchmark tasks across all test harnesses."""
        harness_results: list[HarnessResult] = []

        for h_id in self.test_harnesses:
            adapter = get_adapter(h_id)
            total = len(benchmark_trajectories)
            success_count = 0
            malformed_count = 0
            unvisited_crashes = 0
            cascades = 0

            for traj in benchmark_trajectories:
                # Render ADP trajectory to target harness format
                rendered_payload = adapter.render(traj)

                # Query model
                outcome = await self.model_provider.generate_response(rendered_payload, h_id)

                if outcome.get("success", False):
                    success_count += 1
                if outcome.get("malformed_syntax", False):
                    malformed_count += 1
                if outcome.get("unvisited_state_crash", False):
                    unvisited_crashes += 1
                if outcome.get("cascading_failure", False):
                    cascades += 1

            harness_results.append(
                HarnessResult(
                    harness_id=h_id,
                    is_native_training_harness=(h_id == self.native_harness),
                    total_tasks=total,
                    successful_tasks=success_count,
                    malformed_tool_calls=malformed_count,
                    unvisited_state_crashes=unvisited_crashes,
                    cascading_failure_count=cascades,
                )
            )

        report = compute_overfitting_metrics(
            model_id=model_id,
            evaluated_at=datetime.now(timezone.utc).isoformat(),
            results=harness_results,
        )
        return report
