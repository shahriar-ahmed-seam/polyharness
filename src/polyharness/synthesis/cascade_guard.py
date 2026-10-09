"""CascadeGuard: Teacher-Forcing Cascade Mitigation Engine.

Injects off-distribution perturbation states and self-correction recovery turns
into clean trajectories, ensuring models learn recovery policies instead of
collapsing upon the first unexpected observation.
"""

from __future__ import annotations

import copy
import random

from polyharness.schema.adp import Step, ToolCall, ToolResult, Trajectory

ERROR_TEMPLATES = [
    {
        "error_msg": "Error 429: Rate limit exceeded. Backoff and retry with updated parameters.",
        "thought": "The previous tool call hit a rate limit. I will retry with safe parameters and verify status.",
    },
    {
        "error_msg": "ToolExecutionError: Parameter validation failed. Expected string format for 'id'.",
        "thought": "I provided a malformed parameter type. Let me cast the argument to string and execute again.",
    },
    {
        "error_msg": "TimeoutError: Target endpoint did not respond within 5000ms.",
        "thought": "The connection timed out. I will attempt fallback query with tighter constraints.",
    },
    {
        "error_msg": "StateMismatchError: Target element or table record not found in active view.",
        "thought": "The previous selector did not match. Let me check the latest observation and target the alternate identifier.",
    },
]


class CascadeGuard:
    """Injects simulated off-distribution states and recovery trajectories into SFT data."""

    def __init__(self, seed: int | None = None):
        if seed is not None:
            random.seed(seed)

    def inject_recovery_turn(
        self,
        trajectory: Trajectory,
        step_index_to_perturb: int | None = None,
    ) -> Trajectory:
        """Inject a realistic error-and-recovery cycle into an existing trajectory."""
        if not trajectory.steps:
            return trajectory

        aug_traj = copy.deepcopy(trajectory)
        aug_traj.id = f"{trajectory.id}_cascade_guarded"

        # Find steps that have tool calls
        candidate_indices = [
            i for i, s in enumerate(aug_traj.steps) if s.tool_calls and not s.is_recovery_turn
        ]

        if not candidate_indices:
            return aug_traj

        if step_index_to_perturb is not None and step_index_to_perturb in candidate_indices:
            target_idx = step_index_to_perturb
        else:
            target_idx = random.choice(candidate_indices)

        target_step = aug_traj.steps[target_idx]
        template = random.choice(ERROR_TEMPLATES)

        # 1. Create a failing turn right before or at the target step
        original_call = target_step.tool_calls[0]
        failed_call = ToolCall(
            name=original_call.name,
            arguments=original_call.arguments,
        )
        failed_result = ToolResult(
            tool_call_id=failed_call.id,
            name=failed_call.name,
            content=template["error_msg"],
            is_error=True,
        )

        failing_step = Step(
            step_index=target_idx,
            thought=f"Attempting to invoke {failed_call.name} to advance the goal.",
            tool_calls=[failed_call],
            tool_results=[failed_result],
            is_recovery_turn=False,
        )

        # 2. Modify target step to be the recovery step
        target_step.is_recovery_turn = True
        target_step.thought = (
            f"{template['thought']} "
            + (target_step.thought or f"Now executing correct {original_call.name}.")
        )

        # Insert failing step before target step and re-index all steps
        new_steps = []
        for i, s in enumerate(aug_traj.steps):
            if i == target_idx:
                new_steps.append(failing_step)
            new_steps.append(s)

        for new_idx, s in enumerate(new_steps):
            s.step_index = new_idx

        aug_traj.steps = new_steps
        aug_traj.metadata.notes = "Injected off-distribution state and recovery turn via CascadeGuard"
        return aug_traj

    def augment_dataset(
        self,
        trajectories: list[Trajectory],
        injection_rate: float = 0.5,
    ) -> list[Trajectory]:
        """Augment a collection of trajectories with teacher-forcing recovery turns."""
        output = []
        for t in trajectories:
            output.append(t)  # preserve clean trajectory
            if random.random() < injection_rate:
                guarded = self.inject_recovery_turn(t)
                output.append(guarded)
        return output
