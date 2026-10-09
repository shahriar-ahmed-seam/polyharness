"""Robust parsing and error recovery utilities for LLM payloads.

Handles malformed JSON, unescaped quotes, Python dict literals, trailing commas,
and reconciles tool call IDs across disparate harness protocols.
"""

from __future__ import annotations

import ast
import json
import re
import uuid
from typing import Any


def parse_resilient_json(payload: Any) -> dict[str, Any]:
    """Parse an LLM-generated string into a dictionary using multi-stage recovery.

    Stage 1: Already a dict -> return directly.
    Stage 2: Standard json.loads after markdown codeblock unwrapping.
    Stage 3: Common JSON syntax repair (trailing commas, escaped quotes).
    Stage 4: Safe ast.literal_eval for Python dictionary literals (single quotes, True/False/None).
    Stage 5: Regex key-value extraction for semi-structured text.
    Stage 6: Fallback wrapper preserving raw text in `_raw_argument`.
    """
    if isinstance(payload, dict):
        return payload
    if payload is None:
        return {}
    if not isinstance(payload, str):
        try:
            return dict(payload)
        except Exception:
            return {"_raw": str(payload)}

    text = payload.strip()
    if not text:
        return {}

    # Strip markdown code blocks: ```json ... ``` or ``` ... ```
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text).strip()

    # Stage 2: Standard JSON parse
    try:
        res = json.loads(text)
        if isinstance(res, dict):
            return res
        return {"_value": res}
    except Exception:
        pass

    # Stage 3: Common JSON syntax repairs
    repaired = text
    # Remove trailing commas before closing braces/brackets
    repaired = re.sub(r",\s*([\]}])", r"\1", repaired)
    try:
        res = json.loads(repaired)
        if isinstance(res, dict):
            return res
        return {"_value": res}
    except Exception:
        pass

    # Stage 4: Python literal dict (single quotes, True/False/None)
    try:
        node = ast.literal_eval(text)
        if isinstance(node, dict):
            return node
        if isinstance(node, (list, tuple)):
            return {"_items": list(node)}
    except Exception:
        pass

    # Stage 5: Regex key-value extraction for unquoted or loose syntax
    # e.g., query="shoes", limit=10, active=true
    kv_pattern = re.compile(
        r"""(?P<key>[a-zA-Z_][a-zA-Z0-9_]*)\s*[:=]\s*(?P<val>"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|[^,}\s]+)"""
    )
    matches = list(kv_pattern.finditer(text))
    if matches:
        extracted: dict[str, Any] = {}
        for m in matches:
            k = m.group("key")
            v_str = m.group("val").strip()
            # Strip quotes
            if (v_str.startswith('"') and v_str.endswith('"')) or (
                v_str.startswith("'") and v_str.endswith("'")
            ):
                extracted[k] = v_str[1:-1]
            elif v_str.lower() in ("true", "false"):
                extracted[k] = v_str.lower() == "true"
            elif v_str.lower() in ("null", "none"):
                extracted[k] = None
            else:
                try:
                    extracted[k] = int(v_str) if v_str.isdigit() else float(v_str)
                except ValueError:
                    extracted[k] = v_str
        if extracted:
            return extracted

    # Stage 6: Fallback wrapper
    return {"_raw_argument": text}


def reconcile_tool_call_ids(
    tool_calls: list[Any],
    tool_results: list[Any],
) -> None:
    """Ensure every ToolCall has a non-empty ID and every ToolResult correctly references it.

    Modifies objects in-place to guarantee valid correlation for strict harnesses (OpenAI, Anthropic).
    """
    # 1. Assign IDs to tool calls missing them
    for i, tc in enumerate(tool_calls):
        if not getattr(tc, "id", None) or tc.id == "":
            tc.id = f"call_{uuid.uuid4().hex[:8]}_{i}"

    # 2. Reconcile tool results
    if len(tool_calls) == len(tool_results):
        # 1-to-1 sequential match
        for tc, tr in zip(tool_calls, tool_results, strict=False):
            if not getattr(tr, "tool_call_id", None) or tr.tool_call_id == "":
                tr.tool_call_id = tc.id
            if not getattr(tr, "name", None) or tr.name == "":
                tr.name = tc.name
    else:
        # Match by name if available, otherwise fallback sequentially
        matched_call_ids = set()
        for tr in tool_results:
            if getattr(tr, "tool_call_id", None) and tr.tool_call_id:
                matched_call_ids.add(tr.tool_call_id)
                continue
            # Try to match by tool name
            matching_call = next(
                (tc for tc in tool_calls if tc.name == getattr(tr, "name", None) and tc.id not in matched_call_ids),
                None,
            )
            if matching_call:
                tr.tool_call_id = matching_call.id
                matched_call_ids.add(matching_call.id)
            elif tool_calls:
                # Fallback to first available
                tr.tool_call_id = tool_calls[0].id
