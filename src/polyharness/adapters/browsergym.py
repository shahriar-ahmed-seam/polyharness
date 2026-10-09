"""BrowserGym / Web Navigation Harness Adapter."""

from __future__ import annotations

from typing import Any

from polyharness.adapters.base import BaseAdapter
from polyharness.schema.adp import (
    EnvironmentSpec,
    Observation,
    ObservationType,
    Step,
    ToolCall,
    ToolDefinition,
    ToolParameter,
    Trajectory,
    TrajectoryMetadata,
)


class BrowserGymAdapter(BaseAdapter):
    adapter_id = "browsergym"
    description = "BrowserGym / WebArena format (AXTree / DOM action spaces like click, fill, scroll)"

    STANDARD_WEB_TOOLS = [
        ToolDefinition(
            name="click",
            description="Click on an element specified by target ID or selector",
            parameters={
                "bid": ToolParameter(name="bid", type="string", description="Element ID or browser backend bid", required=True)
            },
        ),
        ToolDefinition(
            name="fill",
            description="Type text into an input field specified by target ID",
            parameters={
                "bid": ToolParameter(name="bid", type="string", description="Element ID", required=True),
                "value": ToolParameter(name="value", type="string", description="Text to enter", required=True),
            },
        ),
        ToolDefinition(
            name="scroll",
            description="Scroll webpage in given direction",
            parameters={
                "direction": ToolParameter(name="direction", type="string", description="'up' or 'down'", required=True)
            },
        ),
        ToolDefinition(
            name="goto",
            description="Navigate to a specific URL",
            parameters={
                "url": ToolParameter(name="url", type="string", description="Target URL address", required=True)
            },
        ),
    ]

    def render(self, trajectory: Trajectory, **kwargs: Any) -> dict[str, Any]:
        """Render trajectory into BrowserGym step actions & AXTree / HTML observations."""
        observation_mode = kwargs.get("observation_mode", "accessibility_tree")
        episodes = []

        for step in trajectory.steps:
            obs_payload = ""
            if step.observation:
                if observation_mode == "accessibility_tree" and step.observation.accessibility_tree:
                    obs_payload = step.observation.accessibility_tree
                elif observation_mode == "html" and step.observation.html_content:
                    obs_payload = step.observation.html_content
                elif observation_mode == "markdown" and step.observation.markdown:
                    obs_payload = step.observation.markdown
                else:
                    obs_payload = step.observation.raw_content

            action_str = ""
            if step.tool_calls:
                c = step.tool_calls[0]
                args_list = [f"{k}={v!r}" for k, v in c.arguments.items()]
                action_str = f"{c.name}({', '.join(args_list)})"

            episodes.append(
                {
                    "step_index": step.step_index,
                    "thought": step.thought or "",
                    "observation": obs_payload,
                    "action": action_str,
                    "reward": step.reward,
                }
            )

        return {
            "task": trajectory.task,
            "observation_mode": observation_mode,
            "episodes": episodes,
            "success": trajectory.outcome.get("success", False),
        }

    def parse(self, raw_data: dict[str, Any], **kwargs: Any) -> Trajectory:
        task = raw_data.get("task", "Browser Task")
        episodes = raw_data.get("episodes", [])
        steps: list[Step] = []

        for ep in episodes:
            s_idx = ep.get("step_index", len(steps))
            thought = ep.get("thought")
            obs_raw = ep.get("observation", "")
            action_raw = ep.get("action", "")

            tool_calls = []
            if "(" in action_raw and action_raw.endswith(")"):
                fn_name = action_raw[: action_raw.index("(")].strip()
                args_body = action_raw[action_raw.index("(") + 1 : -1].strip()
                args_dict = {}
                if args_body:
                    for part in args_body.split(","):
                        if "=" in part:
                            k, v = part.split("=", 1)
                            args_dict[k.strip()] = v.strip().strip("'").strip('"')
                tool_calls.append(ToolCall(name=fn_name, arguments=args_dict))

            obs_obj = Observation(
                primary_type=ObservationType.ACCESSIBILITY_TREE,
                raw_content=obs_raw,
                accessibility_tree=obs_raw,
            )

            steps.append(
                Step(
                    step_index=s_idx,
                    thought=thought,
                    tool_calls=tool_calls,
                    observation=obs_obj,
                    reward=ep.get("reward"),
                )
            )

        return Trajectory(
            task=task,
            tools=self.STANDARD_WEB_TOOLS,
            steps=steps,
            environment=EnvironmentSpec(
                name="browsergym", observation_space="axtree", action_space="browser_actions"
            ),
            metadata=TrajectoryMetadata(source_framework="browsergym"),
        )

    def to_sft_record(self, trajectory: Trajectory, **kwargs: Any) -> dict[str, Any]:
        rendered = self.render(trajectory, **kwargs)
        messages = [{"role": "system", "content": "You are a web automation agent."}]
        messages.append({"role": "user", "content": f"Task: {rendered['task']}"})
        for ep in rendered["episodes"]:
            messages.append({"role": "user", "content": f"Observation:\n{ep['observation']}"})
            content = f"Thought: {ep['thought']}\nAction: {ep['action']}"
            messages.append({"role": "assistant", "content": content})
        return {"messages": messages}
