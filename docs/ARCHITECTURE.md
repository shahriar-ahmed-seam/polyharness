# PolyHarness Core Architecture Blueprint

PolyHarness is an open-standard trajectory interlingua (**Agent Data Protocol - ADP v1.0**) and multi-harness evaluation suite designed to eliminate single-harness overfitting in fine-tuned LLM agents.

---

## 1. System Dataflow & Subsystems

```
                                 ┌─────────────────────────────────┐
                                 │    Heterogeneous Agent Data     │
                                 │ (OpenAI, Claude, Hermes, ReAct) │
                                 └────────────────┬────────────────┘
                                                  │
                                          Adapter Ingestion
                                                  ▼
                                 ┌─────────────────────────────────┐
                                 │   Canonical Interlingua Layer   │
                                 │            (ADP v1.0)           │
                                 └────────────────┬────────────────┘
                                                  │
                ┌─────────────────────────────────┼─────────────────────────────────┐
                ▼                                 ▼                                 ▼
    ┌──────────────────────┐          ┌──────────────────────┐          ┌──────────────────────┐
    │   Syntax Perturber   │          │ Observation Pipeline │          │    Cascade Guard     │
    │  (Param/Case Inoc.)  │          │(DOM, AXTree, Vision) │          │(Off-Dist. Recovery)  │
    └───────────┬──────────┘          └───────────┬──────────┘          └───────────┬──────────┘
                └─────────────────────────────────┼─────────────────────────────────┘
                                                  │
                                                  ▼
                                 ┌─────────────────────────────────┐
                                 │   Streaming Dataset Compiler    │
                                 │   - Sharded Partitions (.jsonl) │
                                 │   - Token Budget Estimator      │
                                 │   - Deterministic Mixture Split │
                                 └────────────────┬────────────────┘
                                                  │
                                                  ▼
                                 ┌─────────────────────────────────┐
                                 │ Multi-Harness Evaluation Suite  │
                                 │   - HOC / CHTS / CDR Metrics    │
                                 │   - 95% Bootstrap Bounds        │
                                 └─────────────────────────────────┘
```

---

## 2. Architectural Subsystems

### 2.1 Canonical Interlingua Layer (`polyharness.schema`)
The core abstraction is `Trajectory` defined under `Agent Data Protocol v1.0`. It decouples raw agent reasoning, tool calling, and environment feedback from vendor-specific syntax:
- **`Trajectory`**: Complete episodic task instance with metadata, system prompt, tool definitions, sequence of `Step`s, and final outcomes.
- **`Step`**: Atomic multi-turn transaction containing reasoning (`thought`), semantic intent (`action_intent`), tool invocations (`ToolCall`), responses (`ToolResult`), environmental state changes (`Observation`), and recovery flags (`is_recovery_turn`).
- **`Observation`**: Neutral multi-modal observation container supporting text, markdown, compressed HTML/DOM, Accessibility Tree (AXTree), structured JSON, and raw base64 screenshots.

### 2.2 Bidirectional Harness Adapters (`polyharness.adapters`)
PolyHarness provides bidirectional translation (`render()` and `parse()`) across 6 major agent ecosystems:
1. **OpenAI ChatML**: Native `tool_calls` function invocation with JSON arguments.
2. **Anthropic Claude**: Structured `tool_use` and `tool_result` content blocks.
3. **Nous Hermes 2/3**: Open-weights XML standard (`<tools>`, `<tool_call>`, `<tool_response>`).
4. **ReAct**: Canonical thought/action/observation plain-text prompts.
5. **BrowserGym**: High-fidelity web navigation format with DOM and AXTree states.
6. **HuggingFace Smolagents**: Pythonic code/function calling standard for open models.

Each adapter includes resilient error recovery (`parse_resilient_json`), repairing unescaped quotes, trailing commas, and Python dict literals, while reconciling `tool_call_id` references across disparate frameworks.

### 2.3 Synthesis & Anti-Overfitting Engines (`polyharness.synthesis`)
Addresses the three root causes identified in single-harness fine-tuning collapse:
1. **Tool Syntax & Primitives (`SyntaxPerturber`)**:
   - Inoculates models against hardcoded syntax by permuting parameter order and casing (`camelCase` vs `snake_case`).
2. **Observation Format Monoculture (`ObservationTransformer`)**:
   - Synthesizes concurrent views of the environment (Markdown, raw DOM, Accessibility Tree).
   - Features `compact_dom` AST reduction: removes noise tags (`<script>`, `<style>`, `<svg>`), discards decorative attributes, and preserves actionable interactive elements (`data-testid`, `href`, `aria-label`).
3. **Teacher Forcing Cascade (`CascadeGuard`)**:
   - Injects realistic off-distribution errors (rate limits, 404s, malformed selector returns) followed by self-correction recovery steps.

### 2.4 Streaming Dataset Compiler (`StreamingDatasetCompiler`)
Engineered for production fine-tuning pipelines (Axolotl, HuggingFace TRL, Megatron-LM):
- **Zero-Copy Streaming**: Lazily consumes input files (`.json`, `.jsonl`, `.jsonl.gz`) via Python generators.
- **Deterministic Mixture Hashing**: Assigns trajectories to target harness formats via SHA-256 hash modulo, guaranteeing identical distributed splits without inter-process communication.
- **Sharded Writer**: Automatically partitions data into fixed-size chunks (`part-00000.jsonl`, etc.) with gzip compression.
- **Token Budget Auditing**: Computes empirical sequence token distribution (`min`, `mean`, `p50`, `p95`, `max`) and skips or flags over-budget sequences.

### 2.5 Multi-Harness Evaluation Suite (`polyharness.eval`)
Evaluates fine-tuned checkpoints against an orthogonal test matrix:
- **Primary Harness**: The framework used during training (e.g., ReAct).
- **Unseen Eval Harnesses**: Transfer harnesses never exposed during training (e.g., Hermes, OpenAI, Smolagents).
- Computes formal overfitting metrics with empirical 95% bootstrap confidence intervals.
