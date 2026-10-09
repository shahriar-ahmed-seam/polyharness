# Agent Data Protocol (ADP v1.0) Specification

The **Agent Data Protocol (ADP v1.0)** is an open data specification defining a vendor-neutral, canonical representation for LLM agent trajectories.

---

## 1. Core Schema Hierarchy

```
Trajectory
├── id: str (UUID v4)
├── schema_version: "1.0.0"
├── task: str
├── system_prompt: str | None
├── tools: list[ToolDefinition]
│   └── parameters: dict[str, ToolParameter]
├── steps: list[Step]
│   ├── step_index: int (0-indexed, strictly monotonic)
│   ├── thought: str | None
│   ├── action_intent: str | None
│   ├── tool_calls: list[ToolCall]
│   ├── tool_results: list[ToolResult]
│   ├── observation: Observation | None
│   └── is_recovery_turn: bool
├── outcome: dict[str, Any]
│   ├── success: bool
│   └── final_answer: str | None
└── metadata: TrajectoryMetadata
```

---

## 2. Field Specifications

### 2.1 Trajectory
| Field | Type | Required | Description |
|---|---|---|---|
| `id` | `string` | Yes | Unique trajectory identifier. |
| `schema_version` | `string` | Yes | Specification version (`"1.0.0"`). |
| `task` | `string` | Yes | Natural language goal or user instruction. |
| `system_prompt` | `string` | No | System prompt or operational guidelines. |
| `tools` | `list[ToolDefinition]` | Yes | Registry of tools available to the agent. |
| `steps` | `list[Step]` | Yes | Ordered sequence of interaction steps. |
| `outcome` | `dict` | Yes | Task resolution status (`success`, `final_answer`). |
| `metadata` | `TrajectoryMetadata` | Yes | Provenance, timestamps, and environment tags. |

### 2.2 Step
| Field | Type | Required | Description |
|---|---|---|---|
| `step_id` | `string` | Yes | Unique step identifier. |
| `step_index` | `integer` | Yes | 0-indexed integer; must equal the step's array index. |
| `thought` | `string` | No | Model chain-of-thought or reasoning string. |
| `action_intent` | `string` | No | Declarative intent of the action. |
| `tool_calls` | `list[ToolCall]` | Yes | List of tool invocations performed in this turn. |
| `tool_results` | `list[ToolResult]` | Yes | Observations resulting from the tool invocations. |
| `observation` | `Observation` | No | Environmental snapshot (DOM, AXTree, screenshot). |
| `is_recovery_turn` | `boolean` | Yes | `true` if this turn recovers from an error state. |
| `reward` | `float` | No | Scalar credit assignment between `-1.0` and `1.0`. |

### 2.3 ToolCall & ToolResult
- **`ToolCall`**:
  - `id`: Unique invocation ID (e.g. `call_a1b2c3d4`).
  - `name`: Target tool identifier matching a registered `ToolDefinition`.
  - `arguments`: Dictionary of typed parameters.
- **`ToolResult`**:
  - `tool_call_id`: Must correlate with the originating `ToolCall.id`.
  - `name`: Target tool name.
  - `content`: Text or structured observation payload.
  - `is_error`: Boolean indicating execution failure.
  - `image_base64`: Base64 string for multimodal visual returns.
  - `mime_type`: MIME type of visual payload (e.g., `image/png`).

### 2.4 Observation
- `primary_type`: `"text"`, `"markdown"`, `"html"`, `"accessibility_tree"`, `"json"`, or `"multimodal_image"`.
- `raw_content`: Text string representation.
- `html_content`: Raw web DOM (if web agent).
- `accessibility_tree`: Simplified ARIA accessibility hierarchy.
- `screenshot_base64`: Base64 screenshot for vision-language models.
- `image_url`: Remote URI or asset path.

---

## 3. Invariants & Validation Rules

1. **Monotonic Step Indexing**: For step $i$ in `trajectory.steps`, `step.step_index == i`. Gaps or out-of-order steps invalidate the trajectory.
2. **Tool Identifier Consistency**: Every `ToolCall.name` must exist in `trajectory.tools`.
3. **Correlated Tool Results**: For each `ToolCall` in `step.tool_calls`, a matching `ToolResult` with the identical `tool_call_id` must be provided in `step.tool_results`.
4. **Deterministic Outcome**: A trajectory must terminate with an explicit `outcome.success` boolean.
