"""Tests for ADP schema, validation, and serialization."""

from polyharness.schema.adp import (
    Step,
    ToolCall,
    ToolDefinition,
    ToolParameter,
    ToolResult,
    Trajectory,
)
from polyharness.schema.serialization import load_trajectory, save_trajectory
from polyharness.schema.validator import TrajectoryValidator


def test_trajectory_creation_and_serialization(tmp_path):
    tool = ToolDefinition(
        name="calculator",
        description="Calculate math expression",
        parameters={
            "expr": ToolParameter(name="expr", type="string", description="Expression", required=True)
        },
    )

    step = Step(
        step_index=0,
        thought="I will calculate 2 + 2",
        tool_calls=[ToolCall(name="calculator", arguments={"expr": "2 + 2"})],
        tool_results=[ToolResult(tool_call_id="call_1", name="calculator", content="4")],
    )

    traj = Trajectory(
        task="What is 2 + 2?",
        tools=[tool],
        steps=[step],
        outcome={"success": True, "final_answer": "4"},
    )

    assert traj.total_tool_calls() == 1
    assert traj.recovery_turn_count() == 0

    # Test file round-trip
    file_path = tmp_path / "test_traj.json"
    save_trajectory(traj, file_path)
    loaded = load_trajectory(file_path)

    assert loaded.id == traj.id
    assert loaded.task == traj.task
    assert len(loaded.steps) == 1
    assert loaded.steps[0].tool_calls[0].name == "calculator"


def test_trajectory_validator():
    step0 = Step(
        step_index=0,
        thought="Search items",
        tool_calls=[ToolCall(name="search", arguments={"q": "laptop"})],
        tool_results=[ToolResult(tool_call_id="c1", name="search", content="laptop found")],
    )

    traj = Trajectory(
        task="Find a laptop",
        tools=[
            ToolDefinition(
                name="search",
                parameters={"q": ToolParameter(name="q", type="string")},
            )
        ],
        steps=[step0],
    )

    report = TrajectoryValidator.validate(traj)
    assert report.is_valid is True
    assert report.total_steps == 1
    # Check that lack of error recovery turns is flagged as teacher forcing risk
    assert "teacher_forcing_cascade_vulnerability" in report.harness_overfitting_risks
    assert report.harness_overfitting_risks["teacher_forcing_cascade_vulnerability"] > 0.5


def test_invalid_step_indexing():
    # Gap in step indexing
    step0 = Step(step_index=0, thought="First")
    step2 = Step(step_index=2, thought="Third")  # skipped 1

    traj = Trajectory(task="Index gap test", steps=[step0, step2])
    report = TrajectoryValidator.validate(traj)
    assert report.is_valid is False
    assert any("integrity" in i.category for i in report.issues)
