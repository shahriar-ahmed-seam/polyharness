"""Tests for multi-harness evaluation metrics and runner."""

import pytest

from polyharness.eval.benchmarks import get_standard_benchmarks
from polyharness.eval.metrics import HarnessResult, compute_overfitting_metrics
from polyharness.eval.runner import MockModelProvider, MultiHarnessEvaluator


def test_compute_overfitting_metrics():
    # Simulate single-harness overfitted model results:
    # 100% on OpenAI (native), 20% on others
    results = [
        HarnessResult(
            harness_id="openai",
            is_native_training_harness=True,
            total_tasks=10,
            successful_tasks=10,
            malformed_tool_calls=0,
            unvisited_state_crashes=0,
            cascading_failure_count=0,
        ),
        HarnessResult(
            harness_id="hermes",
            is_native_training_harness=False,
            total_tasks=10,
            successful_tasks=2,
            malformed_tool_calls=5,
            unvisited_state_crashes=2,
            cascading_failure_count=5,
        ),
        HarnessResult(
            harness_id="react",
            is_native_training_harness=False,
            total_tasks=10,
            successful_tasks=2,
            malformed_tool_calls=4,
            unvisited_state_crashes=2,
            cascading_failure_count=4,
        ),
    ]

    report = compute_overfitting_metrics(
        model_id="test-overfit",
        evaluated_at="2026-10-09T00:00:00Z",
        results=results,
    )

    assert report.native_accuracy == 1.0
    assert report.unseen_harnesses_average_accuracy == 0.2
    assert report.harness_overfitting_coefficient == 0.8  # (1.0 - 0.2) / 1.0
    assert report.is_production_safe is False
    assert isinstance(report.hoc_confidence_interval, tuple)
    assert isinstance(report.chts_confidence_interval, tuple)
    assert report.hoc_confidence_interval[0] <= report.harness_overfitting_coefficient <= report.hoc_confidence_interval[1]


@pytest.mark.asyncio
async def test_multi_harness_evaluator_generalist():
    benchmarks = get_standard_benchmarks()
    provider = MockModelProvider(profile="polyharness_generalist", native_harness="openai")
    evaluator = MultiHarnessEvaluator(
        model_provider=provider,
        native_harness="openai",
        test_harnesses=["openai", "hermes", "anthropic"],
    )

    report = await evaluator.evaluate_suite(benchmarks, model_id="generalist-test")
    assert report.native_accuracy == 1.0
    assert report.unseen_harnesses_average_accuracy == 1.0
    assert report.harness_overfitting_coefficient == 0.0
    assert report.is_production_safe is True
