"""Agent Data Protocol (ADP v1.0) Specification.

Neutral interlingua schema for agent trajectories, decoupling raw task logic
and interactions from harness-specific formatting, syntax, and prompt idiosyncrasies.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ObservationType(str, Enum):
    TEXT = "text"
    MARKDOWN = "markdown"
    HTML = "html"
    ACCESSIBILITY_TREE = "accessibility_tree"
    JSON = "json"
    MULTI_REPRESENTATION = "multi_representation"


class ToolParameter(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: str = Field(..., description="Parameter name")
    type: str = Field(default="string", description="JSON schema data type")
    description: str = Field(default="", description="Semantic documentation for parameter")
    required: bool = Field(default=False, description="Whether parameter is strictly required")
    default: Any | None = Field(default=None, description="Default value if omitted")
    enum: list[str] | None = Field(default=None, description="Permitted enumeration values")


class ToolDefinition(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: str = Field(..., description="Canonical tool identifier")
    description: str = Field(default="", description="Function intent and docstring")
    parameters: dict[str, ToolParameter] = Field(
        default_factory=dict, description="Named parameters mapped to specs"
    )
    returns_type: str = Field(default="string", description="Return value schema type")


class ToolCall(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str = Field(default_factory=lambda: f"call_{uuid.uuid4().hex[:10]}")
    name: str = Field(..., description="Canonical tool identifier invoked")
    arguments: dict[str, Any] = Field(default_factory=dict, description="Typed invocation arguments")


class ToolResult(BaseModel):
    model_config = ConfigDict(extra="allow")

    tool_call_id: str = Field(..., description="Correlating ToolCall ID")
    name: str = Field(..., description="Canonical tool identifier invoked")
    content: str = Field(..., description="Observation payload returned by tool execution")
    is_error: bool = Field(default=False, description="True if execution threw an exception")


class Observation(BaseModel):
    model_config = ConfigDict(extra="allow")

    primary_type: ObservationType = Field(default=ObservationType.TEXT)
    raw_content: str = Field(..., description="Primary observation string payload")
    html_content: str | None = Field(default=None, description="Raw HTML if web interaction")
    accessibility_tree: str | None = Field(
        default=None, description="Cleaned ARIA / accessibility tree"
    )
    markdown: str | None = Field(default=None, description="Rendered markdown representation")
    structured_state: dict[str, Any] | None = Field(
        default=None, description="JSON state representation"
    )
    metadata: dict[str, Any] = Field(default_factory=dict)


class Step(BaseModel):
    model_config = ConfigDict(extra="allow")

    step_id: str = Field(default_factory=lambda: f"step_{uuid.uuid4().hex[:8]}")
    step_index: int = Field(..., ge=0, description="0-indexed step sequence order")
    thought: str | None = Field(
        default=None, description="Model internal chain-of-thought / reasoning"
    )
    action_intent: str | None = Field(
        default=None, description="Neutral semantic intent of the chosen action"
    )
    tool_calls: list[ToolCall] = Field(
        default_factory=list, description="Tools invoked in this turn"
    )
    tool_results: list[ToolResult] = Field(
        default_factory=list, description="Observations resulting from tool invocations"
    )
    observation: Observation | None = Field(
        default=None, description="Environment observation before or after the turn"
    )
    state_delta: dict[str, Any] | None = Field(
        default=None, description="Changes in environment/memory caused by this turn"
    )
    is_recovery_turn: bool = Field(
        default=False,
        description="Indicates whether this step recovers from an off-distribution or error state",
    )
    reward: float | None = Field(
        default=None, description="Turn-level reward or verification score (-1.0 to 1.0)"
    )


class EnvironmentSpec(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: str = Field(..., description="Target environment name (e.g. browser, bash, sql, rest_api)")
    version: str = Field(default="1.0.0")
    observation_space: str = Field(default="text/dom/json")
    action_space: str = Field(default="function_call")
    harness_id: str = Field(
        default="generic",
        description="Original recording harness ID (e.g. langchain, autogen, openai, hermes)",
    )


class TrajectoryMetadata(BaseModel):
    model_config = ConfigDict(extra="allow")

    source_framework: str = Field(default="polyharness")
    recorder_version: str = Field(default="0.1.0")
    recorded_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    total_tokens: int | None = Field(default=None)
    success: bool = Field(default=True)
    tags: list[str] = Field(default_factory=list)
    dataset_split: str = Field(default="train")
    notes: str | None = Field(default=None)


class Trajectory(BaseModel):
    """Canonical Agent Data Protocol (ADP) Trajectory Model."""

    model_config = ConfigDict(extra="allow")

    version: str = Field(default="adp-1.0", description="ADP specification version")
    id: str = Field(default_factory=lambda: f"traj_{uuid.uuid4().hex[:12]}")
    task: str = Field(..., description="High-level user prompt or goal directive")
    system_prompt: str | None = Field(
        default=None, description="Neutral base instructions for agent behavior"
    )
    environment: EnvironmentSpec = Field(
        default_factory=lambda: EnvironmentSpec(name="generic_environment")
    )
    tools: list[ToolDefinition] = Field(
        default_factory=list, description="Available tool definitions"
    )
    steps: list[Step] = Field(
        default_factory=list, description="Ordered sequence of trajectory steps"
    )
    outcome: dict[str, Any] = Field(
        default_factory=lambda: {"success": True, "final_answer": ""}
    )
    metadata: TrajectoryMetadata = Field(default_factory=TrajectoryMetadata)

    def get_tool(self, tool_name: str) -> ToolDefinition | None:
        """Look up tool definition by name."""
        for t in self.tools:
            if t.name == tool_name:
                return t
        return None

    def total_tool_calls(self) -> int:
        """Calculate total number of tool calls across all steps."""
        return sum(len(s.tool_calls) for s in self.steps)

    def recovery_turn_count(self) -> int:
        """Number of steps tagged as error recovery turns."""
        return sum(1 for s in self.steps if s.is_recovery_turn)
