"""Evaluation and benchmarking module for PolyHarness."""

from polyharness.eval.benchmarks import get_standard_benchmarks
from polyharness.eval.metrics import (
    HarnessResult,
    OverfittingAuditReport,
    compute_overfitting_metrics,
)
from polyharness.eval.runner import (
    BaseModelProvider,
    MockModelProvider,
    MultiHarnessEvaluator,
)

__all__ = [
    "BaseModelProvider",
    "HarnessResult",
    "MockModelProvider",
    "MultiHarnessEvaluator",
    "OverfittingAuditReport",
    "compute_overfitting_metrics",
    "get_standard_benchmarks",
]
