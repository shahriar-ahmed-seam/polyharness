"""HuggingFace Smolagents Harness Adapter."""

from __future__ import annotations

import json
import re
from typing import Any

from polyharness.adapters.base import BaseAdapter
from polyharness.schema.adp import (
    EnvironmentSpec,
    Step,
    ToolCall,
    ToolDefinition,
    ToolResult,
    Trajectory,
    TrajectoryMetadata,
)
from polyharness.schema.parsing import parse_resilient_json, reconcile_tool_call_ids


class SmolagentsAdapter(BaseAdapter):
    adapter_id = "smolagents"
    description = "HuggingFace Smolagents format (CodeAgent / ToolCallingAgent actions)"

    def _render_tools_description(self, tools: list[ToolDefinition]) -> str:
        lines = []
        for t in tools:
            args_list = []
            for p_name, spec in t.parameters.items():
                req = " (required)" if spec.required else ""
                args_list.append(f"{p_name}: {spec.type}{req}")
            lines.append(f"- {t.name}({', '.join(args_list)}): {t.description}")
        return "\n".join(lines)

    def render(self, trajectory: Trajectory, **kwargs: Any) -> dict[str, Any]:
        """Render trajectory into Smolagents ToolCallingAgent trace format."""
        tools_desc = self._render_tools_description(trajectory.tools)
        system_msg = (
            trajectory.system_prompt
            or "You are an expert agent utilizing tools to solve complex tasks."
        )
        full_system = (
            f"{system_msg}\n\nAvailable tools:\n{tools_desc}\n\n"
            "To invoke a tool, output: Action: tool_name(arg=val)\n"
            "When completed, output: Action: final_answer(answer='...')"
        )

        turns = [{"role": "system", "content": full_system}]
        turns.append({"role": "user", "content": trajectory.task})

        for step in trajectory.steps:
            thought = step.thought or "Analyzing state..."
            for call in step.tool_calls:
                call_args = ", ".join(f"{k}={json.dumps(v)}" for k, v in call.arguments.items())
                turns.append(
                    {
                        "role": "assistant",
                        "content": f"Thought: {thought}\nAction: {call.name}({call_args})",
                    }
                )

            for res in step.tool_results:
                turns.append({"role": "environment", "content": f"Observation: {res.content}"})

        if trajectory.outcome.get("final_answer"):
            fa = trajectory.outcome["final_answer"]
            turns.append(
                {
                    "role": "assistant",
                    "content": f"Thought: I have finished the task.\nAction: final_answer(answer={json.dumps(fa)})",
                }
            )

        return {
            "task": trajectory.task,
            "turns": turns,
            "tools_count": len(trajectory.tools),
        }

    def parse(self, raw_data: dict[str, Any], **kwargs: Any) -> Trajectory:
        turns = raw_data.get("turns", [])
        task = raw_data.get("task", "Smolagents Task")
        steps: list[Step] = []
        step_idx = 0
        final_answer = ""

        i = 0
        while i < len(turns):
            turn = turns[i]
            role = turn.get("role")
            content = turn.get("content", "")

            if role == "assistant":
                thought_match = re.search(r"Thought:\s*(.*?)(?=\nAction:|\Z)", content, re.DOTALL)
                action_match = re.search(r"Action:\s*(\w+)\((.*?)\)", content, re.DOTALL)

                thought = thought_match.group(1).strip() if thought_match else ""
                tool_calls: list[ToolCall] = []

                if action_match:
                    tool_name = action_match.group(1)
                    args_raw = action_match.group(2)

                    if tool_name == "final_answer":
                        fa_match = re.search(r"answer=(.*)", args_raw)
                        final_answer = fa_match.group(1).strip().strip("'\"") if fa_match else args_raw
                        i += 1
                        continue

                    args_dict = parse_resilient_json(args_raw)
                    tool_calls.append(ToolCall(name=tool_name, arguments=args_dict))

                tool_results: list[ToolResult] = []
                j = i + 1
                if j < len(turns) and turns[j].get("role") == "environment":
                    env_content = turns[j].get("content", "")
                    obs_match = re.search(r"Observation:\s*(.*)", env_content, re.DOTALL)
                    res_body = obs_match.group(1).strip() if obs_match else env_content
                    tool_results.append(
                        ToolResult(tool_call_id="", name=tool_calls[0].name if tool_calls else "", content=res_body)
                    )
                    j += 1

                reconcile_tool_call_ids(tool_calls, tool_results)
                if thought or tool_calls:
                    steps.append(
                        Step(
                            step_index=step_idx,
                            thought=thought if thought else None,
                            tool_calls=tool_calls,
                            tool_results=tool_results,
                        )
                    )
                    step_idx += 1
                i = j
                continue

            i += 1

        return Trajectory(
            task=task,
            steps=steps,
            outcome={"success": bool(final_answer), "final_answer": final_answer},
            environment=EnvironmentSpec(name="smolagents_sandbox", harness_id="smolagents"),
            metadata=TrajectoryMetadata(source_framework="smolagents"),
        )

    def to_sft_record(self, trajectory: Trajectory, **kwargs: Any) -> dict[str, Any]:
        rendered = self.render(trajectory, **kwargs)
        messages = []
        for t in rendered["turns"]:
            role = "user" if t["role"] == "environment" else t["role"]
            messages.append({"role": role, "content": t["content"]})
        return {"messages": messages}
