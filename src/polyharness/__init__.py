"""PolyHarness: The Open-Standard Trajectory Interlingua & Multi-Harness Evaluation Suite.

Solves the core failure modes of single-harness agent fine-tuning:
1. Tool Syntax & Primitives overfitting
2. Observation Format lock-in
3. Teacher-Forcing cascading rollout errors
"""

__version__ = "0.1.0"
__author__ = "Shahriar Ahmed Seam"

from polyharness.schema.adp import (
    EnvironmentSpec,
    Observation,
    Step,
    ToolCall,
    ToolDefinition,
    ToolResult,
    Trajectory,
    TrajectoryMetadata,
)

__all__ = [
    "EnvironmentSpec",
    "Observation",
    "Step",
    "ToolCall",
    "ToolDefinition",
    "ToolResult",
    "Trajectory",
    "TrajectoryMetadata",
]
