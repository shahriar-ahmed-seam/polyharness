"""Tests for all 6 bidirectional harness adapters."""

import pytest

from polyharness.adapters import get_adapter, list_adapters
from polyharness.eval.benchmarks import get_standard_benchmarks


@pytest.fixture
def sample_trajectory():
    return get_standard_benchmarks()[0]


def test_list_adapters():
    adapters = list_adapters()
    adapter_ids = [a["id"] for a in adapters]
    assert "openai" in adapter_ids
    assert "anthropic" in adapter_ids
    assert "hermes" in adapter_ids
    assert "react" in adapter_ids
    assert "browsergym" in adapter_ids
    assert "langchain" in adapter_ids


def test_openai_adapter(sample_trajectory):
    adapter = get_adapter("openai")
    rendered = adapter.render(sample_trajectory)

    assert "messages" in rendered
    assert "tools" in rendered
    assert len(rendered["tools"]) == 2
    assert rendered["tools"][0]["function"]["name"] == "search_products"

    # Test SFT format
    sft = adapter.to_sft_record(sample_trajectory)
    assert "messages" in sft

    # Test round-trip parsing
    parsed = adapter.parse(rendered)
    assert parsed.task == sample_trajectory.task
    assert len(parsed.steps) > 0


def test_anthropic_adapter(sample_trajectory):
    adapter = get_adapter("anthropic")
    rendered = adapter.render(sample_trajectory)

    assert "messages" in rendered
    assert "tools" in rendered
    assert rendered["tools"][0]["name"] == "search_products"
    assert "input_schema" in rendered["tools"][0]

    # Check tool_use block structure
    assistant_msgs = [m for m in rendered["messages"] if m["role"] == "assistant"]
    assert any(
        isinstance(m["content"], list) and any(b.get("type") == "tool_use" for b in m["content"])
        for m in assistant_msgs
    )

    # Test parsing
    parsed = adapter.parse(rendered)
    assert len(parsed.steps) > 0


def test_hermes_adapter(sample_trajectory):
    adapter = get_adapter("hermes")
    rendered = adapter.render(sample_trajectory)

    system_msg = rendered["messages"][0]["content"]
    assert "<tools>" in system_msg
    assert "</tools>" in system_msg

    assistant_msg = next(m["content"] for m in rendered["messages"] if m["role"] == "assistant")
    assert "<tool_call>" in assistant_msg

    # Test parsing
    parsed = adapter.parse(rendered)
    assert len(parsed.steps) > 0


def test_react_adapter(sample_trajectory):
    adapter = get_adapter("react")
    rendered = adapter.render(sample_trajectory)

    assert "prompt" in rendered
    assert "Question:" in rendered["prompt"]
    assert "Thought:" in rendered["prompt"]
    assert "Action:" in rendered["prompt"]

    parsed = adapter.parse(rendered)
    assert len(parsed.steps) > 0


def test_browsergym_adapter(sample_trajectory):
    adapter = get_adapter("browsergym")
    rendered = adapter.render(sample_trajectory)

    assert "episodes" in rendered
    assert len(rendered["episodes"]) == len(sample_trajectory.steps)

    parsed = adapter.parse(rendered)
    assert len(parsed.steps) == len(sample_trajectory.steps)


def test_langchain_adapter(sample_trajectory):
    adapter = get_adapter("langchain")
    rendered = adapter.render(sample_trajectory)

    assert "intermediate_steps" in rendered
    assert len(rendered["intermediate_steps"]) > 0

    parsed = adapter.parse(rendered)
    assert len(parsed.steps) > 0


def test_smolagents_adapter(sample_trajectory):
    adapter = get_adapter("smolagents")
    rendered = adapter.render(sample_trajectory)

    assert "turns" in rendered
    assert len(rendered["turns"]) > 0
    assert any("Action:" in t["content"] for t in rendered["turns"] if t["role"] == "assistant")

    parsed = adapter.parse(rendered)
    assert len(parsed.steps) > 0
    assert parsed.environment.harness_id == "smolagents"

