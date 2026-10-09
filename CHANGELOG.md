# Changelog

All notable changes to **PolyHarness** will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-10-09

### Added
- **Agent Data Protocol (ADP v1.0)**: Canonical neutral interlingua schema for agent trajectories.
- **Bidirectional Harness Adapters**:
  - `openai`: OpenAI ChatML & Tool Calling schema.
  - `anthropic`: Claude 3.5/3.7 Tool Use and Content Blocks schema.
  - `hermes`: Nous-Hermes 2/3 XML Function Calling standard.
  - `react`: Classic ReAct (Thought / Action / Observation) format.
  - `browsergym`: BrowserGym & WebArena accessibility tree and actions format.
  - `langchain`: LangChain AgentExecutor trace format.
- **Anti-Overfitting Synthesis Engines**:
  - `SyntaxPerturber`: Schema noise, parameter casing, and argument shuffling.
  - `ObservationTransformer`: DOM HTML ⇄ AXTree ⇄ Markdown ⇄ JSON transformations.
  - `CascadeGuard`: Off-distribution error simulation and self-correction recovery turn injection.
  - `DatasetCompiler`: Multi-harness SFT dataset generator for HuggingFace TRL, Axolotl, Unsloth, and OpenAI.
- **Multi-Harness Evaluation Suite**:
  - Harness Overfitting Coefficient (HOC) metric.
  - Cross-Harness Transfer Score (CHTS) and Cascading Divergence Rate (CDR).
  - Mock and API model evaluators with automated audit reporting.
- **PolyHarness Web Studio**:
  - Interactive dark-themed web console with side-by-side Interlingua conversion, live whiteboard diagnostics, and eval matrix visualizer.
- **Developer Toolchain**:
  - Full CLI (`polyharness info`, `validate`, `render`, `augment`, `compile`, `eval`, `serve`).
  - Production Dockerfile and Docker Compose configurations.
  - Multi-Python CI workflows and benchmark test suites.
