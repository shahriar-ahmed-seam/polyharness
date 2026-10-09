"""Tests for resilient argument parsing, ID reconciliation, parallel tool calling, and multimodal vision."""

from __future__ import annotations

from polyharness.adapters import get_adapter
from polyharness.schema.adp import (
    Observation,
    ObservationType,
    Step,
    ToolCall,
    ToolDefinition,
    ToolParameter,
    ToolResult,
    Trajectory,
)
from polyharness.schema.parsing import parse_resilient_json, reconcile_tool_call_ids


def test_parse_resilient_json_variants():
    # 1. Clean JSON
    assert parse_resilient_json('{"key": "value", "num": 42}') == {"key": "value", "num": 42}

    # 2. Markdown wrapped
    md = '```json\n{"action": "lookup", "target": "user_id"}\n```'
    assert parse_resilient_json(md) == {"action": "lookup", "target": "user_id"}

    # 3. Trailing comma
    trailing = '{"items": ["a", "b"], "limit": 10,}'
    assert parse_resilient_json(trailing) == {"items": ["a", "b"], "limit": 10}

    # 4. Python literal dictionary
    py_dict = "{'query': 'running shoes', 'filter_brand': 'nike', 'in_stock': True, 'max_price': None}"
    res = parse_resilient_json(py_dict)
    assert res["query"] == "running shoes"
    assert res["in_stock"] is True
    assert res["max_price"] is None

    # 5. Key-value string format (e.g. from code agent or loose regex)
    kv_str = 'file_path="src/main.py", line_number=42, verbose=true'
    res_kv = parse_resilient_json(kv_str)
    assert res_kv["file_path"] == "src/main.py"
    assert res_kv["line_number"] == 42
    assert res_kv["verbose"] is True

    # 6. Unparseable fallback
    raw_str = "something completely arbitrary with no delimiters"
    fallback = parse_resilient_json(raw_str)
    assert "_raw_argument" in fallback


def test_reconcile_tool_call_ids():
    tc1 = ToolCall(id="", name="search", arguments={"q": "test"})
    tc2 = ToolCall(id="", name="filter", arguments={"brand": "acme"})
    tr1 = ToolResult(tool_call_id="", name="search", content="found 10 items")
    tr2 = ToolResult(tool_call_id="", name="filter", content="filtered to 2 items")

    reconcile_tool_call_ids([tc1, tc2], [tr1, tr2])

    assert tc1.id != ""
    assert tc2.id != ""
    assert tr1.tool_call_id == tc1.id
    assert tr2.tool_call_id == tc2.id


def test_parallel_tool_calling_roundtrip():
    # Step with 2 parallel tool calls
    tc1 = ToolCall(id="call_01", name="fetch_user", arguments={"user_id": 101})
    tc2 = ToolCall(id="call_02", name="fetch_orders", arguments={"user_id": 101})
    tr1 = ToolResult(tool_call_id="call_01", name="fetch_user", content='{"name": "Alice"}')
    tr2 = ToolResult(tool_call_id="call_02", name="fetch_orders", content='[{"order_id": 501}]')

    step = Step(
        step_index=0,
        thought="Fetching user profile and orders concurrently",
        tool_calls=[tc1, tc2],
        tool_results=[tr1, tr2],
    )

    tools = [
        ToolDefinition(name="fetch_user", parameters={"user_id": ToolParameter(name="user_id", type="integer")}),
        ToolDefinition(name="fetch_orders", parameters={"user_id": ToolParameter(name="user_id", type="integer")}),
    ]

    traj = Trajectory(
        task="Audit user orders",
        tools=tools,
        steps=[step],
        outcome={"success": True, "final_answer": "Alice has 1 order"},
    )

    # Render to OpenAI and parse back
    openai_adapter = get_adapter("openai")
    rendered_openai = openai_adapter.render(traj)
    asst_msg = next(m for m in rendered_openai["messages"] if m.get("role") == "assistant" and "tool_calls" in m)
    assert len(asst_msg["tool_calls"]) == 2

    parsed_openai = openai_adapter.parse(rendered_openai)
    assert len(parsed_openai.steps[0].tool_calls) == 2
    assert parsed_openai.steps[0].tool_calls[0].name == "fetch_user"
    assert parsed_openai.steps[0].tool_calls[1].name == "fetch_orders"


    # Render to Hermes and parse back
    hermes_adapter = get_adapter("hermes")
    rendered_hermes = hermes_adapter.render(traj)
    parsed_hermes = hermes_adapter.parse(rendered_hermes)
    assert len(parsed_hermes.steps[0].tool_calls) == 2


def test_multimodal_vision_observation_support():
    b64_mock = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
    obs = Observation(
        primary_type=ObservationType.MULTIMODAL_IMAGE,
        raw_content="Browser viewport rendered after clicking login button",
        screenshot_base64=b64_mock,
        mime_type="image/png",
    )

    tr = ToolResult(
        tool_call_id="call_click",
        name="browser_click",
        content="Clicked button",
        image_base64=b64_mock,
        mime_type="image/png",
    )

    tc = ToolCall(id="call_click", name="browser_click", arguments={"selector": "#login"})
    step = Step(
        step_index=0,
        thought="Clicking login button and capturing viewport",
        tool_calls=[tc],
        tool_results=[tr],
        observation=obs,
    )

    traj = Trajectory(
        task="Log into application",
        tools=[ToolDefinition(name="browser_click", parameters={})],
        steps=[step],
        outcome={"success": True},
    )

    # Render to OpenAI
    openai_adapter = get_adapter("openai")
    rendered_openai = openai_adapter.render(traj)
    # Check tool message has image block
    tool_msg = next(m for m in rendered_openai["messages"] if m["role"] == "tool")
    assert isinstance(tool_msg["content"], list)
    assert any(b.get("type") == "image_url" for b in tool_msg["content"])

    # Parse back from OpenAI
    parsed_openai = openai_adapter.parse(rendered_openai)
    parsed_res = parsed_openai.steps[0].tool_results[0]
    assert parsed_res.image_base64 == b64_mock

    # Render to Anthropic
    anthropic_adapter = get_adapter("anthropic")
    rendered_anthropic = anthropic_adapter.render(traj)
    user_msg = next(m for m in rendered_anthropic["messages"] if m["role"] == "user" and isinstance(m["content"], list))
    tool_res_block = user_msg["content"][0]
    assert isinstance(tool_res_block["content"], list)
    assert any(b.get("type") == "image" for b in tool_res_block["content"])

    # Parse back from Anthropic
    parsed_anthropic = anthropic_adapter.parse(rendered_anthropic)
    assert parsed_anthropic.steps[0].tool_results[0].image_base64 == b64_mock
