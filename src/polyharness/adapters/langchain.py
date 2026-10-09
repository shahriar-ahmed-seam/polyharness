"""LangChain & LangSmith Trace Adapter."""

from __future__ import annotations

import json
from typing import Any

from polyharness.adapters.base import BaseAdapter
from polyharness.schema.adp import (
    EnvironmentSpec,
    Step,
    ToolCall,
    ToolResult,
    Trajectory,
    TrajectoryMetadata,
)


class LangChainAdapter(BaseAdapter):
    adapter_id = "langchain"
    description = "LangChain AgentExecutor / LangSmith trace format"

    def render(self, trajectory: Trajectory, **kwargs: Any) -> dict[str, Any]:
        """Convert trajectory into LangChain intermediate_steps execution log."""
        intermediate_steps = []
        for step in trajectory.steps:
            for call, res in zip(step.tool_calls, step.tool_results):
                action_obj = {
                    "tool": call.name,
                    "tool_input": call.arguments,
                    "log": f"Invoking {call.name} with {json.dumps(call.arguments)}",
                }
                intermediate_steps.append((action_obj, res.content))

        return {
            "input": trajectory.task,
            "intermediate_steps": intermediate_steps,
            "output": trajectory.outcome.get("final_answer", ""),
            "tools": [t.name for t in trajectory.tools],
        }

    def parse(self, raw_data: dict[str, Any], **kwargs: Any) -> Trajectory:
        task = raw_data.get("input", "")
        intermediate_steps = raw_data.get("intermediate_steps", [])
        output = raw_data.get("output", "")

        steps: list[Step] = []
        for idx, item in enumerate(intermediate_steps):
            action_data, observation_str = item[0], item[1]
            tool_name = action_data.get("tool", "unknown_tool")
            tool_input = action_data.get("tool_input", {})
            if isinstance(tool_input, str):
                try:
                    tool_input = json.loads(tool_input)
                except Exception:
                    tool_input = {"input": tool_input}

            call = ToolCall(name=tool_name, arguments=tool_input)
            result = ToolResult(tool_call_id=call.id, name=tool_name, content=str(observation_str))

            steps.append(
                Step(
                    step_index=idx,
                    thought=action_data.get("log"),
                    tool_calls=[call],
                    tool_results=[result],
                )
            )

        return Trajectory(
            task=task,
            steps=steps,
            outcome={"success": bool(output), "final_answer": output},
            environment=EnvironmentSpec(name="langchain_harness", harness_id="langchain"),
            metadata=TrajectoryMetadata(source_framework="langchain"),
        )

    def to_sft_record(self, trajectory: Trajectory, **kwargs: Any) -> dict[str, Any]:
        rendered = self.render(trajectory, **kwargs)
        messages = [{"role": "system", "content": "You are a LangChain ReAct agent."}]
        messages.append({"role": "user", "content": rendered["input"]})
        for action, obs in rendered["intermediate_steps"]:
            messages.append(
                {"role": "assistant", "content": f"Action: {action['tool']}\nInput: {json.dumps(action['tool_input'])}"}
            )
            messages.append({"role": "user", "content": f"Observation: {obs}"})
        if rendered["output"]:
            messages.append({"role": "assistant", "content": f"Final Answer: {rendered['output']}"})
        return {"messages": messages}
