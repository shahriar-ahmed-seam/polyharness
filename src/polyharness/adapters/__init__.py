"""Harness adapters registry."""


from polyharness.adapters.anthropic import AnthropicAdapter
from polyharness.adapters.base import BaseAdapter
from polyharness.adapters.browsergym import BrowserGymAdapter
from polyharness.adapters.hermes import HermesAdapter
from polyharness.adapters.langchain import LangChainAdapter
from polyharness.adapters.openai import OpenAIAdapter
from polyharness.adapters.react import ReActAdapter
from polyharness.adapters.smolagents import SmolagentsAdapter

ADAPTER_REGISTRY: dict[str, type[BaseAdapter]] = {
    "openai": OpenAIAdapter,
    "anthropic": AnthropicAdapter,
    "hermes": HermesAdapter,
    "react": ReActAdapter,
    "browsergym": BrowserGymAdapter,
    "langchain": LangChainAdapter,
    "smolagents": SmolagentsAdapter,
}


def get_adapter(name: str) -> BaseAdapter:
    """Retrieve an instantiated adapter by harness ID."""
    key = name.lower().strip()
    if key not in ADAPTER_REGISTRY:
        supported = ", ".join(ADAPTER_REGISTRY.keys())
        raise ValueError(f"Unknown harness adapter '{name}'. Supported harnesses: {supported}")
    return ADAPTER_REGISTRY[key]()


def list_adapters() -> list[dict[str, str]]:
    """List all registered harness adapters with descriptions."""
    results = []
    for k, cls in ADAPTER_REGISTRY.items():
        results.append({
            "id": k,
            "description": cls.description,
        })
    return results


__all__ = [
    "AnthropicAdapter",
    "BaseAdapter",
    "BrowserGymAdapter",
    "HermesAdapter",
    "LangChainAdapter",
    "OpenAIAdapter",
    "ReActAdapter",
    "SmolagentsAdapter",
    "get_adapter",
    "list_adapters",
]
