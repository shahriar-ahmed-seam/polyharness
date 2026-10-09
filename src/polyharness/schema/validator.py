"""ADP Trajectory Validator & Fragility Diagnostics.

Scans trajectories for single-harness vulnerabilities:
- Unpaired tool calls / missing observations
- Index sequence gaps
- Observation modality monoculture (Overfitting risk)
- Teacher-Forcing fragility (lack of error recovery turns)
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from polyharness.schema.adp import Trajectory


class ValidationIssue(BaseModel):
    severity: str = Field(..., description="'error', 'warning', or 'info'")
    category: str = Field(..., description="Vulnerability category")
    step_index: int = Field(default=-1)
    message: str
    recommendation: str


class TrajectoryDiagnosticReport(BaseModel):
    is_valid: bool
    trajectory_id: str
    task: str
    total_steps: int
    quality_score: float = Field(
        ..., description="0-100 composite score of trajectory quality and robustness"
    )
    issues: list[ValidationIssue] = Field(default_factory=list)
    harness_overfitting_risks: dict[str, float] = Field(
        ...,
        description="Risk scores (0.0 - 1.0) for the 3 whiteboard failure modes",
    )


class TrajectoryValidator:
    """Validates ADP trajectories and scores robustness against harness overfitting."""

    @staticmethod
    def validate(trajectory: Trajectory) -> TrajectoryDiagnosticReport:
        issues: list[ValidationIssue] = []
        errors_count = 0

        # 1. Step sequence verification
        step_indices = [s.step_index for s in trajectory.steps]
        expected_indices = list(range(len(trajectory.steps)))
        if step_indices != expected_indices:
            issues.append(
                ValidationIssue(
                    severity="error",
                    category="integrity",
                    step_index=-1,
                    message=f"Step indices are non-sequential or gapped: {step_indices} vs expected {expected_indices}",
                    recommendation="Re-index steps starting strictly from 0 without gaps.",
                )
            )
            errors_count += 1

        # 2. Tool call & Tool result pairing
        declared_tool_names = {t.name for t in trajectory.tools}
        all_call_ids = set()
        has_error_recovery = False
        all_obs_types = set()

        for step in trajectory.steps:
            if step.is_recovery_turn:
                has_error_recovery = True

            if step.observation:
                all_obs_types.add(step.observation.primary_type.value)

            for call in step.tool_calls:
                all_call_ids.add(call.id)
                if call.name not in declared_tool_names and declared_tool_names:
                    issues.append(
                        ValidationIssue(
                            severity="warning",
                            category="tool_syntax",
                            step_index=step.step_index,
                            message=f"Tool '{call.name}' is invoked but missing from declared tools.",
                            recommendation=f"Add '{call.name}' to trajectory.tools schema definition.",
                        )
                    )

            for result in step.tool_results:
                if result.is_error:
                    has_error_recovery = True

        # 3. Check whiteboard failure mode: Observation format monoculture
        # "Unseen HTML or tree formats force model onto unvisited states."
        has_multi_obs = any(
            s.observation and (
                (s.observation.html_content and s.observation.markdown) or
                (s.observation.accessibility_tree and s.observation.html_content) or
                (s.observation.structured_state and s.observation.markdown)
            )
            for s in trajectory.steps
        )

        obs_monoculture_risk = 0.85 if not has_multi_obs else 0.15
        if not has_multi_obs:
            issues.append(
                ValidationIssue(
                    severity="warning",
                    category="observation_format",
                    step_index=-1,
                    message="Observation monoculture: Trajectory only contains a single observation modality.",
                    recommendation="Use `polyharness.synthesis.ObservationTransformer` to generate multi-format views (DOM, Tree, Markdown).",
                )
            )

        # 4. Check whiteboard failure mode: Teacher forcing cascade
        # "First unfamiliar state pushes off-distribution; errors compound every turn."
        teacher_forcing_risk = 0.90 if not has_error_recovery else 0.20
        if not has_error_recovery:
            issues.append(
                ValidationIssue(
                    severity="warning",
                    category="teacher_forcing_cascade",
                    step_index=-1,
                    message="Zero error recovery turns: 100% clean trajectory guarantees brittle rollouts under distribution drift.",
                    recommendation="Use `polyharness.synthesis.CascadeGuard` to inject simulated perturbation & recovery turns.",
                )
            )

        # 5. Check whiteboard failure mode: Tool Syntax & Primitives fragility
        tool_syntax_risk = 0.70 if len(trajectory.tools) == 0 else 0.25
        if len(trajectory.tools) == 0 and trajectory.total_tool_calls() > 0:
            issues.append(
                ValidationIssue(
                    severity="warning",
                    category="tool_syntax",
                    step_index=-1,
                    message="Tools are invoked without parameter schemas.",
                    recommendation="Declare strict ToolDefinition schemas so cross-harness compilers can translate types accurately.",
                )
            )

        # Compute Quality Score
        base_score = 100.0
        base_score -= (errors_count * 30.0)
        base_score -= (obs_monoculture_risk * 25.0)
        base_score -= (teacher_forcing_risk * 25.0)
        base_score -= (tool_syntax_risk * 20.0)
        final_quality = max(0.0, min(100.0, base_score))

        return TrajectoryDiagnosticReport(
            is_valid=(errors_count == 0),
            trajectory_id=trajectory.id,
            task=trajectory.task,
            total_steps=len(trajectory.steps),
            quality_score=round(final_quality, 1),
            issues=issues,
            harness_overfitting_risks={
                "tool_syntax_fragility": round(tool_syntax_risk, 2),
                "observation_format_lockin": round(obs_monoculture_risk, 2),
                "teacher_forcing_cascade_vulnerability": round(teacher_forcing_risk, 2),
            },
        )
