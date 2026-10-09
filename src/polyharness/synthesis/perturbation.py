"""Schema & Tool Syntax Perturbation Engine.

Prevents tool syntax and primitive overfitting by generating safe syntactic variations
of tool definitions and calls (snake_case vs camelCase, parameter reordering, noisy docstrings).
"""

from __future__ import annotations

import copy
import random

from polyharness.schema.adp import ToolDefinition, ToolParameter, Trajectory


def to_camel_case(snake_str: str) -> str:
    """Convert snake_case string to camelCase."""
    components = snake_str.split("_")
    return components[0] + "".join(x.title() for x in components[1:])


def to_snake_case(camel_str: str) -> str:
    """Convert camelCase string to snake_case."""
    import re
    return re.sub(r"(?<!^)(?=[A-Z])", "_", camel_str).lower()


class SyntaxPerturber:
    """Perturbs tool parameters and schemas to inoculate agents against syntax overfitting."""

    def __init__(self, seed: int | None = None):
        if seed is not None:
            random.seed(seed)

    def perturb_tool_schema(
        self,
        tool: ToolDefinition,
        alternate_casing: bool = True,
        shuffle_parameters: bool = True,
    ) -> ToolDefinition:
        """Create a syntactically perturbed clone of a ToolDefinition."""
        new_tool = copy.deepcopy(tool)

        # Shuffle parameter definition order
        param_items = list(new_tool.parameters.items())
        if shuffle_parameters:
            random.shuffle(param_items)

        new_params: dict[str, ToolParameter] = {}
        for p_name, p_spec in param_items:
            final_name = p_name
            if alternate_casing:
                final_name = to_camel_case(p_name) if "_" in p_name else to_snake_case(p_name)
            p_spec.name = final_name
            new_params[final_name] = p_spec

        new_tool.parameters = new_params
        return new_tool

    def perturb_trajectory(
        self,
        trajectory: Trajectory,
        parameter_casing_prob: float = 0.5,
        shuffle_params: bool = True,
    ) -> Trajectory:
        """Create an augmented trajectory with perturbed tool parameter naming and ordering."""
        traj = copy.deepcopy(trajectory)
        traj.id = f"{trajectory.id}_synth_syntax"

        param_mapping: dict[str, dict[str, str]] = {}

        for tool in traj.tools:
            tool_mapping = {}
            new_params = {}
            param_items = list(tool.parameters.items())
            if shuffle_params:
                random.shuffle(param_items)

            for p_name, p_spec in param_items:
                if random.random() < parameter_casing_prob:
                    new_name = to_camel_case(p_name) if "_" in p_name else to_snake_case(p_name)
                else:
                    new_name = p_name
                tool_mapping[p_name] = new_name
                p_spec.name = new_name
                new_params[new_name] = p_spec

            tool.parameters = new_params
            param_mapping[tool.name] = tool_mapping

        # Propagate parameter naming to step tool calls
        for step in traj.steps:
            for call in step.tool_calls:
                mapping = param_mapping.get(call.name, {})
                new_args = {}
                for k, v in call.arguments.items():
                    new_k = mapping.get(k, k)
                    new_args[new_k] = v
                call.arguments = new_args

        traj.metadata.notes = "Synthetically augmented for tool syntax invariance"
        return traj
