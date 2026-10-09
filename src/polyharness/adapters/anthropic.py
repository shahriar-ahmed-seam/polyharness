"""Anthropic Claude 3.5 / 3.7 Tool Use Adapter."""

from __future__ import annotations

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
from polyharness.schema.parsing import parse_resilient_json, reconcile_tool_call_ids


class AnthropicAdapter(BaseAdapter):
    adapter_id = "anthropic"
    description = "Anthropic Claude Tool Use block format (tool_use / tool_result blocks)"

    def _convert_tool_to_anthropic(self, tool: ToolDefinition) -> dict[str, Any]:
        properties: dict[str, Any] = {}
        required: list[str] = []

        for p_name, param in tool.parameters.items():
            prop: dict[str, Any] = {"type": param.type, "description": param.description}
            if param.enum:
                prop["enum"] = param.enum
            properties[p_name] = prop
            if param.required:
                required.append(p_name)

        return {
            "name": tool.name,
            "description": tool.description,
            "input_schema": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        }

    def render(self, trajectory: Trajectory, **kwargs: Any) -> dict[str, Any]:
        tools_payload = [self._convert_tool_to_anthropic(t) for t in trajectory.tools]
        messages: list[dict[str, Any]] = []

        messages.append({"role": "user", "content": trajectory.task})

        for step in trajectory.steps:
            if step.tool_calls:
                reconcile_tool_call_ids(step.tool_calls, step.tool_results)
                assistant_content: list[dict[str, Any]] = []
                if step.thought:
                    assistant_content.append({"type": "text", "text": step.thought})

                for call in step.tool_calls:
                    assistant_content.append(
                        {
                            "type": "tool_use",
                            "id": call.id,
                            "name": call.name,
                            "input": call.arguments,
                        }
                    )
                messages.append({"role": "assistant", "content": assistant_content})

                # Anthropic groups all tool results into one user message with tool_result blocks
                if step.tool_results:
                    user_content: list[dict[str, Any]] = []
                    for res in step.tool_results:
                        res_block: dict[str, Any] = {
                            "type": "tool_result",
                            "tool_use_id": res.tool_call_id,
                            "content": res.content,
                            "is_error": res.is_error,
                        }
                        if res.image_base64:
                            res_block["content"] = [
                                {"type": "text", "text": res.content},
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": res.mime_type or "image/png",
                                        "data": res.image_base64,
                                    },
                                },
                            ]
                        user_content.append(res_block)
                    messages.append({"role": "user", "content": user_content})
            elif step.thought or step.action_intent:
                text = step.thought or step.action_intent
                messages.append({"role": "assistant", "content": text})


        if trajectory.outcome.get("final_answer"):
            messages.append(
                {"role": "assistant", "content": trajectory.outcome["final_answer"]}
            )

        payload: dict[str, Any] = {
            "model": kwargs.get("model", "claude-3-5-sonnet-20241022"),
            "messages": messages,
            "tools": tools_payload if tools_payload else None,
        }
        if trajectory.system_prompt:
            payload["system"] = trajectory.system_prompt
        return payload

    def parse(self, raw_data: dict[str, Any], **kwargs: Any) -> Trajectory:
        messages = raw_data.get("messages", [])
        tools_raw = raw_data.get("tools", [])
        system_prompt = raw_data.get("system")

        adp_tools: list[ToolDefinition] = []
        for t in tools_raw:
            name = t.get("name", "unknown")
            desc = t.get("description", "")
            schema = t.get("input_schema", {}).get("properties", {})
            req_set = set(t.get("input_schema", {}).get("required", []))
            params: dict[str, ToolParameter] = {}
            for p_name, p_body in schema.items():
                params[p_name] = ToolParameter(
                    name=p_name,
                    type=p_body.get("type", "string"),
                    description=p_body.get("description", ""),
                    required=(p_name in req_set),
                    enum=p_body.get("enum"),
                )
            adp_tools.append(ToolDefinition(name=name, description=desc, parameters=params))

        task = ""
        steps: list[Step] = []
        step_idx = 0

        i = 0
        while i < len(messages):
            msg = messages[i]
            role = msg.get("role")
            content = msg.get("content")

            if role == "user" and not task and isinstance(content, str):
                task = content
                i += 1
                continue

            if role == "assistant" and isinstance(content, list):
                thought_str = ""
                tool_calls: list[ToolCall] = []
                for block in content:
                    if block.get("type") == "text":
                        thought_str += block.get("text", "")
                    elif block.get("type") == "tool_use":
                        inp = block.get("input", {})
                        parsed_inp = parse_resilient_json(inp) if isinstance(inp, str) else inp
                        tool_calls.append(
                            ToolCall(
                                id=block.get("id", f"call_{i}"),
                                name=block.get("name", ""),
                                arguments=parsed_inp if isinstance(parsed_inp, dict) else {"raw": parsed_inp},
                            )
                        )

                tool_results: list[ToolResult] = []
                j = i + 1
                if j < len(messages) and messages[j].get("role") == "user" and isinstance(messages[j].get("content"), list):
                    for res_block in messages[j].get("content", []):
                        if res_block.get("type") == "tool_result":
                            blk_content = res_block.get("content", "")
                            text_body = ""
                            img_b64 = None
                            mime = None
                            if isinstance(blk_content, list):
                                for item in blk_content:
                                    if isinstance(item, dict) and item.get("type") == "text":
                                        text_body += item.get("text", "")
                                    elif isinstance(item, dict) and item.get("type") == "image":
                                        src = item.get("source", {})
                                        img_b64 = src.get("data")
                                        mime = src.get("media_type")
                            else:
                                text_body = str(blk_content)

                            tool_results.append(
                                ToolResult(
                                    tool_call_id=res_block.get("tool_use_id", ""),
                                    name="",
                                    content=text_body,
                                    is_error=res_block.get("is_error", False),
                                    image_base64=img_b64,
                                    mime_type=mime,
                                )
                            )
                    j += 1

                reconcile_tool_call_ids(tool_calls, tool_results)
                steps.append(
                    Step(
                        step_index=step_idx,
                        thought=thought_str if thought_str else None,
                        tool_calls=tool_calls,
                        tool_results=tool_results,
                    )
                )
                step_idx += 1
                i = j
                continue

            i += 1

        final_answer = ""
        if messages and messages[-1].get("role") == "assistant" and isinstance(messages[-1].get("content"), str):
            final_answer = messages[-1].get("content")

        return Trajectory(
            task=task or "Claude Task",
            system_prompt=system_prompt,
            tools=adp_tools,
            steps=steps,
            outcome={"success": True, "final_answer": final_answer},
            environment=EnvironmentSpec(name="anthropic_harness", harness_id="anthropic"),
            metadata=TrajectoryMetadata(source_framework="anthropic"),
        )

    def to_sft_record(self, trajectory: Trajectory, **kwargs: Any) -> dict[str, Any]:
        return self.render(trajectory, **kwargs)
