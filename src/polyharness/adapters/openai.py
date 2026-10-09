"""OpenAI ChatML & Tool Calling Adapter."""

from __future__ import annotations

import json
from typing import Any

from polyharness.adapters.base import BaseAdapter
from polyharness.schema.adp import (
    EnvironmentSpec,
    Step,
    ToolCall,
    ToolDefinition,
    ToolParameter,
    ToolResult,
    Trajectory,
    TrajectoryMetadata,
)


class OpenAIAdapter(BaseAdapter):
    adapter_id = "openai"
    description = "OpenAI ChatML Function Calling format (compatible with GPT-4o, vLLM, SGLang)"

    def _convert_tool_to_openai(self, tool: ToolDefinition) -> dict[str, Any]:
        properties: dict[str, Any] = {}
        required: list[str] = []

        for p_name, param in tool.parameters.items():
            prop: dict[str, Any] = {"type": param.type, "description": param.description}
            if param.enum:
                prop["enum"] = param.enum
            if param.default is not None:
                prop["default"] = param.default
            properties[p_name] = prop
            if param.required:
                required.append(p_name)

        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }

    def render(self, trajectory: Trajectory, **kwargs: Any) -> dict[str, Any]:
        tools_payload = [self._convert_tool_to_openai(t) for t in trajectory.tools]
        messages: list[dict[str, Any]] = []

        if trajectory.system_prompt:
            messages.append({"role": "system", "content": trajectory.system_prompt})

        messages.append({"role": "user", "content": trajectory.task})

        for step in trajectory.steps:
            if step.tool_calls:
                call_objects = []
                for call in step.tool_calls:
                    call_objects.append(
                        {
                            "id": call.id,
                            "type": "function",
                            "function": {
                                "name": call.name,
                                "arguments": json.dumps(call.arguments),
                            },
                        }
                    )
                assistant_msg: dict[str, Any] = {
                    "role": "assistant",
                    "content": step.thought if step.thought else None,
                    "tool_calls": call_objects,
                }
                messages.append(assistant_msg)

                for res in step.tool_results:
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": res.tool_call_id,
                            "name": res.name,
                            "content": res.content,
                        }
                    )
            elif step.thought or step.action_intent:
                content = step.thought or step.action_intent
                messages.append({"role": "assistant", "content": content})

        if trajectory.outcome.get("final_answer"):
            messages.append(
                {"role": "assistant", "content": trajectory.outcome["final_answer"]}
            )

        return {
            "model": kwargs.get("model", "gpt-4o"),
            "messages": messages,
            "tools": tools_payload if tools_payload else None,
        }

    def parse(self, raw_data: dict[str, Any], **kwargs: Any) -> Trajectory:
        messages = raw_data.get("messages", [])
        tools_raw = raw_data.get("tools", [])

        # Reconstruct tools
        adp_tools: list[ToolDefinition] = []
        for t in tools_raw:
            fn = t.get("function", {})
            name = fn.get("name", "unknown")
            desc = fn.get("description", "")
            params_schema = fn.get("parameters", {}).get("properties", {})
            req_set = set(fn.get("parameters", {}).get("required", []))

            params: dict[str, ToolParameter] = {}
            for p_name, p_body in params_schema.items():
                params[p_name] = ToolParameter(
                    name=p_name,
                    type=p_body.get("type", "string"),
                    description=p_body.get("description", ""),
                    required=(p_name in req_set),
                    enum=p_body.get("enum"),
                )
            adp_tools.append(ToolDefinition(name=name, description=desc, parameters=params))

        # Reconstruct steps
        system_prompt = None
        task = ""
        steps: list[Step] = []
        step_idx = 0

        i = 0
        while i < len(messages):
            msg = messages[i]
            role = msg.get("role")
            content = msg.get("content")

            if role == "system":
                system_prompt = content
                i += 1
                continue
            elif role == "user" and not task:
                task = content or ""
                i += 1
                continue
            elif role == "assistant":
                tool_calls_raw = msg.get("tool_calls", [])
                if tool_calls_raw:
                    tool_calls = []
                    for tc in tool_calls_raw:
                        fn = tc.get("function", {})
                        args = fn.get("arguments", "{}")
                        if isinstance(args, str):
                            try:
                                parsed_args = json.loads(args)
                            except Exception:
                                parsed_args = {"raw": args}
                        else:
                            parsed_args = args
                        tool_calls.append(
                            ToolCall(id=tc.get("id", f"call_{i}"), name=fn.get("name", ""), arguments=parsed_args)
                        )

                    # Gather following tool result messages
                    tool_results = []
                    j = i + 1
                    while j < len(messages) and messages[j].get("role") == "tool":
                        tr = messages[j]
                        tool_results.append(
                            ToolResult(
                                tool_call_id=tr.get("tool_call_id", ""),
                                name=tr.get("name", ""),
                                content=tr.get("content", ""),
                            )
                        )
                        j += 1

                    steps.append(
                        Step(
                            step_index=step_idx,
                            thought=content if content else None,
                            tool_calls=tool_calls,
                            tool_results=tool_results,
                        )
                    )
                    step_idx += 1
                    i = j
                    continue
                else:
                    if i == len(messages) - 1:
                        # Final answer turn
                        break
                    steps.append(
                        Step(step_index=step_idx, thought=content)
                    )
                    step_idx += 1
            i += 1

        final_answer = ""
        if messages and messages[-1].get("role") == "assistant" and not messages[-1].get("tool_calls"):
            final_answer = messages[-1].get("content") or ""

        return Trajectory(
            task=task or "Generic Task",
            system_prompt=system_prompt,
            tools=adp_tools,
            steps=steps,
            outcome={"success": True, "final_answer": final_answer},
            environment=EnvironmentSpec(name="openai_harness", harness_id="openai"),
            metadata=TrajectoryMetadata(source_framework="openai"),
        )

    def to_sft_record(self, trajectory: Trajectory, **kwargs: Any) -> dict[str, Any]:
        """Convert into standard OpenAI / HuggingFace messages JSON format for fine-tuning."""
        rendered = self.render(trajectory, **kwargs)
        return {"messages": rendered["messages"], "tools": rendered.get("tools")}
