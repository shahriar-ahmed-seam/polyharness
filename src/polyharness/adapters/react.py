"""ReAct (Reasoning + Acting) Harness Adapter."""

from __future__ import annotations

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


class ReActAdapter(BaseAdapter):
    adapter_id = "react"
    description = "Classic ReAct format (Thought: ... / Action: tool[args] / Observation: ...)"

    def _render_tool_signatures(self, tools: list[ToolDefinition]) -> str:
        lines = []
        for t in tools:
            params_str = ", ".join(f"{p}: {spec.type}" for p, spec in t.parameters.items())
            lines.append(f"- {t.name}({params_str}): {t.description}")
        return "\n".join(lines)

    def render(self, trajectory: Trajectory, **kwargs: Any) -> dict[str, Any]:
        tool_sigs = self._render_tool_signatures(trajectory.tools)
        prompt_lines = [
            trajectory.system_prompt or "Answer the following questions as best you can. You have access to the following tools:",
            "",
            tool_sigs,
            "",
            "Use the following format:",
            "Question: the input question you must answer",
            "Thought: you should always think about what to do",
            "Action: the action to take, should be one of the tools formatted as tool_name(key=val)",
            "Observation: the result of the action",
            "... (this Thought/Action/Observation can repeat N times)",
            "Thought: I now know the final answer",
            "Final Answer: the final answer to the original input question",
            "",
            f"Question: {trajectory.task}",
        ]

        turns: list[str] = ["\n".join(prompt_lines)]

        for step in trajectory.steps:
            turn_lines = []
            if step.thought:
                turn_lines.append(f"Thought: {step.thought.strip()}")

            for call in step.tool_calls:
                args_items = [f'{k}="{v}"' if isinstance(v, str) else f"{k}={v}" for k, v in call.arguments.items()]
                args_str = ", ".join(args_items)
                turn_lines.append(f"Action: {call.name}({args_str})")

            for res in step.tool_results:
                turn_lines.append(f"Observation: {res.content.strip()}")

            turns.append("\n".join(turn_lines))

        if trajectory.outcome.get("final_answer"):
            turns.append(f"Final Answer: {trajectory.outcome['final_answer']}")

        full_prompt = "\n\n".join(turns)
        return {
            "prompt": full_prompt,
            "turns": turns,
        }

    def parse(self, raw_data: dict[str, Any], **kwargs: Any) -> Trajectory:
        text = raw_data.get("prompt", "")
        # Parse Question
        q_match = re.search(r"Question:\s*(.*?)(?=\nThought:|\nAction:|\Z)", text, re.DOTALL)
        task = q_match.group(1).strip() if q_match else "ReAct Task"

        # Split turns by Thought or Action
        step_pattern = re.compile(
            r"Thought:\s*(.*?)(?=\nAction:|\nFinal Answer:|\Z)(?:\nAction:\s*(\w+)\((.*?)\))?(?:\nObservation:\s*(.*?)(?=\nThought:|\nFinal Answer:|\Z))?",
            re.DOTALL,
        )

        steps: list[Step] = []
        step_idx = 0
        final_answer = ""

        # Extract Final Answer
        fa_match = re.search(r"Final Answer:\s*(.*)", text, re.DOTALL)
        if fa_match:
            final_answer = fa_match.group(1).strip()

        for match in step_pattern.finditer(text):
            thought = (match.group(1) or "").strip()
            action_name = match.group(2)
            action_args_raw = match.group(3)
            obs_raw = (match.group(4) or "").strip()

            if thought.startswith("I now know the final answer"):
                continue

            tool_calls = []
            tool_results = []
            if action_name:
                args_dict = {}
                if action_args_raw:
                    for part in action_args_raw.split(","):
                        if "=" in part:
                            k, v = part.split("=", 1)
                            args_dict[k.strip()] = v.strip().strip('"').strip("'")
                tool_calls.append(ToolCall(name=action_name, arguments=args_dict))
                if obs_raw:
                    tool_results.append(
                        ToolResult(tool_call_id="", name=action_name, content=obs_raw)
                    )

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

        return Trajectory(
            task=task,
            steps=steps,
            outcome={"success": bool(final_answer), "final_answer": final_answer},
            environment=EnvironmentSpec(name="react_harness", harness_id="react"),
            metadata=TrajectoryMetadata(source_framework="react"),
        )

    def to_sft_record(self, trajectory: Trajectory, **kwargs: Any) -> dict[str, Any]:
        rendered = self.render(trajectory, **kwargs)
        return {"text": rendered["prompt"]}
