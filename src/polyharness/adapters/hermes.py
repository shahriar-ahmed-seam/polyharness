"""Nous Hermes 2/3 XML Tool Calling Adapter (Open Weights Standard)."""

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
    ToolParameter,
    ToolResult,
    Trajectory,
    TrajectoryMetadata,
)
from polyharness.schema.parsing import parse_resilient_json, reconcile_tool_call_ids


class HermesAdapter(BaseAdapter):
    adapter_id = "hermes"
    description = "Nous-Hermes 2/3 XML Function Calling format (<tools>, <tool_call>, <tool_response>)"

    def _format_tools_xml(self, tools: list[ToolDefinition]) -> str:
        tools_list = []
        for t in tools:
            properties = {}
            required = []
            for p_name, param in t.parameters.items():
                prop = {"type": param.type, "description": param.description}
                if param.enum:
                    prop["enum"] = param.enum
                properties[p_name] = prop
                if param.required:
                    required.append(p_name)

            tools_list.append(
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": {
                            "type": "object",
                            "properties": properties,
                            "required": required,
                        },
                    },
                }
            )
        return (
            "<tools>\n"
            + json.dumps(tools_list, indent=2)
            + "\n</tools>"
        )

    def render(self, trajectory: Trajectory, **kwargs: Any) -> dict[str, Any]:
        tools_xml = self._format_tools_xml(trajectory.tools)
        base_system = trajectory.system_prompt or "You are a helpful AI assistant with tool calling capabilities."
        full_system = f"{base_system}\n\n{tools_xml}"

        messages: list[dict[str, str]] = []
        messages.append({"role": "system", "content": full_system})
        messages.append({"role": "user", "content": trajectory.task})

        for step in trajectory.steps:
            if step.tool_calls:
                parts = []
                if step.thought:
                    parts.append(step.thought.strip())
                for call in step.tool_calls:
                    call_body = json.dumps({"name": call.name, "arguments": call.arguments})
                    parts.append(f"<tool_call>\n{call_body}\n</tool_call>")
                messages.append({"role": "assistant", "content": "\n".join(parts)})

                for res in step.tool_results:
                    res_body = json.dumps({"name": res.name, "content": res.content})
                    messages.append(
                        {
                            "role": "tool",
                            "content": f"<tool_response>\n{res_body}\n</tool_response>",
                        }
                    )
            elif step.thought or step.action_intent:
                text = step.thought or step.action_intent
                messages.append({"role": "assistant", "content": text})

        if trajectory.outcome.get("final_answer"):
            messages.append(
                {"role": "assistant", "content": trajectory.outcome["final_answer"]}
            )

        return {"messages": messages}

    def parse(self, raw_data: dict[str, Any], **kwargs: Any) -> Trajectory:
        messages = raw_data.get("messages", [])
        tools: list[ToolDefinition] = []
        system_prompt = None
        task = ""
        steps: list[Step] = []
        step_idx = 0

        # Extract tools from system prompt
        for msg in messages:
            if msg.get("role") == "system":
                content = msg.get("content", "")
                tools_match = re.search(r"<tools>(.*?)</tools>", content, re.DOTALL)
                if tools_match:
                    try:
                        raw_tools = json.loads(tools_match.group(1).strip())
                        for rt in raw_tools:
                            fn = rt.get("function", {})
                            name = fn.get("name", "unknown")
                            desc = fn.get("description", "")
                            props = fn.get("parameters", {}).get("properties", {})
                            reqs = set(fn.get("parameters", {}).get("required", []))
                            params = {
                                p: ToolParameter(
                                    name=p,
                                    type=props[p].get("type", "string"),
                                    description=props[p].get("description", ""),
                                    required=(p in reqs),
                                )
                                for p in props
                            }
                            tools.append(ToolDefinition(name=name, description=desc, parameters=params))
                    except Exception:
                        pass
                system_prompt = re.sub(r"<tools>.*?</tools>", "", content, flags=re.DOTALL).strip()
                break

        i = 0
        while i < len(messages):
            msg = messages[i]
            role = msg.get("role")
            content = msg.get("content", "")

            if role == "user" and not task:
                task = content
                i += 1
                continue

            if role == "assistant":
                tool_call_matches = list(re.finditer(r"<tool_call>(.*?)</tool_call>", content, re.DOTALL))
                if tool_call_matches:
                    thought_clean = re.sub(r"<tool_call>.*?</tool_call>", "", content, flags=re.DOTALL).strip()
                    calls: list[ToolCall] = []
                    for match in tool_call_matches:
                        raw_call = match.group(1).strip()
                        parsed_c = parse_resilient_json(raw_call)
                        call_name = parsed_c.get("name", "")
                        raw_args = parsed_c.get("arguments", {})
                        parsed_args = parse_resilient_json(raw_args) if isinstance(raw_args, str) else raw_args
                        calls.append(
                            ToolCall(
                                name=call_name or "unknown_tool",
                                arguments=parsed_args if isinstance(parsed_args, dict) else {"raw": parsed_args},
                            )
                        )

                    # Fetch following tool results
                    tool_results: list[ToolResult] = []
                    j = i + 1
                    while j < len(messages) and messages[j].get("role") == "tool":
                        t_msg = messages[j].get("content", "")
                        resp_match = re.search(r"<tool_response>(.*?)</tool_response>", t_msg, re.DOTALL)
                        if resp_match:
                            parsed_r = parse_resilient_json(resp_match.group(1).strip())
                            res_content = parsed_r.get("content", "")
                            res_name = parsed_r.get("name", "")
                            tool_results.append(
                                ToolResult(
                                    tool_call_id="",
                                    name=res_name,
                                    content=str(res_content) if res_content else resp_match.group(1).strip(),
                                )
                            )
                        else:
                            tool_results.append(
                                ToolResult(tool_call_id="", name="", content=t_msg)
                            )
                        j += 1

                    reconcile_tool_call_ids(calls, tool_results)
                    steps.append(
                        Step(
                            step_index=step_idx,
                            thought=thought_clean if thought_clean else None,
                            tool_calls=calls,
                            tool_results=tool_results,
                        )
                    )
                    step_idx += 1
                    i = j
                    continue

                else:
                    if i < len(messages) - 1:
                        steps.append(Step(step_index=step_idx, thought=content))
                        step_idx += 1
            i += 1

        final_answer = ""
        if messages and messages[-1].get("role") == "assistant" and "<tool_call>" not in messages[-1].get("content", ""):
            final_answer = messages[-1].get("content", "")

        return Trajectory(
            task=task or "Hermes Task",
            system_prompt=system_prompt,
            tools=tools,
            steps=steps,
            outcome={"success": True, "final_answer": final_answer},
            environment=EnvironmentSpec(name="hermes_harness", harness_id="hermes"),
            metadata=TrajectoryMetadata(source_framework="hermes"),
        )

    def to_sft_record(self, trajectory: Trajectory, **kwargs: Any) -> dict[str, Any]:
        return self.render(trajectory, **kwargs)
