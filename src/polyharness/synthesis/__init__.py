"""Synthesis and Anti-Overfitting engines for PolyHarness."""

from polyharness.synthesis.cascade_guard import CascadeGuard
from polyharness.synthesis.compiler import DatasetCompiler
from polyharness.synthesis.observation import ObservationTransformer
from polyharness.synthesis.perturbation import SyntaxPerturber
from polyharness.synthesis.streaming import (
    StreamingCompilationSummary,
    StreamingDatasetCompiler,
    TokenBudgetStats,
    estimate_tokens,
)

__all__ = [
    "CascadeGuard",
    "DatasetCompiler",
    "ObservationTransformer",
    "StreamingCompilationSummary",
    "StreamingDatasetCompiler",
    "SyntaxPerturber",
    "TokenBudgetStats",
    "estimate_tokens",
]

