"""Multi-Harness Evaluation Metrics & Overfitting Coefficients.

Quantifies the whiteboard failure modes:
- Harness Overfitting Coefficient (HOC)
- Cross-Harness Transfer Score (CHTS)
- Tool Syntax Robustness (TSR)
- Cascading Divergence Rate (CDR)
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class HarnessResult(BaseModel):
    harness_id: str
    is_native_training_harness: bool = False
    total_tasks: int
    successful_tasks: int
    malformed_tool_calls: int
    unvisited_state_crashes: int
    cascading_failure_count: int

    @property
    def accuracy(self) -> float:
        return self.successful_tasks / max(1, self.total_tasks)

    @property
    def syntax_error_rate(self) -> float:
        return self.malformed_tool_calls / max(1, self.total_tasks)


class OverfittingAuditReport(BaseModel):
    model_id: str
    evaluated_at: str
    native_harness: str
    evaluated_harnesses: list[str]
    native_accuracy: float
    unseen_harnesses_average_accuracy: float

    # Core Whiteboard Metric
    harness_overfitting_coefficient: float = Field(
        ...,
        description="HOC: (Native_Acc - Unseen_Acc) / Native_Acc. High (>0.4) indicates trajectory overfitting.",
    )
    cross_harness_transfer_score: float = Field(
        ..., description="CHTS: Average accuracy across all unseen harnesses (0.0 to 1.0)."
    )
    tool_syntax_robustness: float = Field(
        ..., description="TSR: Resilience against syntax changes and parameter perturbations (0.0 to 1.0)."
    )
    cascading_divergence_rate: float = Field(
        ...,
        description="CDR: Likelihood that an unexpected observation leads to an unrecoverable crash cascade.",
    )
    is_production_safe: bool = Field(
        ..., description="True if HOC < 0.25 and CDR < 0.20."
    )
    detailed_harness_results: dict[str, HarnessResult]


def compute_overfitting_metrics(
    model_id: str,
    evaluated_at: str,
    results: list[HarnessResult],
) -> OverfittingAuditReport:
    """Compute the Harness Overfitting Coefficient and Whiteboard failure diagnostics."""
    native_res = next((r for r in results if r.is_native_training_harness), None)
    unseen_results = [r for r in results if not r.is_native_training_harness]

    if not native_res:
        # Default to first as baseline if not explicitly marked
        native_res = results[0]
        unseen_results = results[1:] if len(results) > 1 else results

    native_acc = native_res.accuracy
    unseen_acc = (
        sum(r.accuracy for r in unseen_results) / max(1, len(unseen_results))
        if unseen_results
        else native_acc
    )

    if native_acc > 0:
        hoc = max(0.0, (native_acc - unseen_acc) / native_acc)
    else:
        hoc = 0.0

    # Tool syntax robustness is 1.0 minus the average syntax error rate
    avg_syntax_errors = sum(r.syntax_error_rate for r in results) / max(1, len(results))
    tsr = max(0.0, 1.0 - avg_syntax_errors)

    # Cascading divergence rate: proportion of total tasks that triggered cascading failures
    total_tasks = sum(r.total_tasks for r in results)
    total_cascades = sum(r.cascading_failure_count for r in results)
    cdr = total_cascades / max(1, total_tasks)

    # Production safety threshold: HOC <= 0.25 and CDR <= 0.20
    is_safe = (hoc <= 0.25) and (cdr <= 0.20) and (native_acc >= 0.70)

    detailed = {r.harness_id: r for r in results}

    return OverfittingAuditReport(
        model_id=model_id,
        evaluated_at=evaluated_at,
        native_harness=native_res.harness_id,
        evaluated_harnesses=[r.harness_id for r in results],
        native_accuracy=round(native_acc, 3),
        unseen_harnesses_average_accuracy=round(unseen_acc, 3),
        harness_overfitting_coefficient=round(hoc, 3),
        cross_harness_transfer_score=round(unseen_acc, 3),
        tool_syntax_robustness=round(tsr, 3),
        cascading_divergence_rate=round(cdr, 3),
        is_production_safe=is_safe,
        detailed_harness_results=detailed,
    )
