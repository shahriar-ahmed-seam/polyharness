#!/usr/bin/env python3
"""Example: Auditing Models for Harness Overfitting & Computing HOC."""

import asyncio

from rich.console import Console
from rich.table import Table

from polyharness.eval.benchmarks import get_standard_benchmarks
from polyharness.eval.runner import MockModelProvider, MultiHarnessEvaluator

console = Console()

async def main():
    benchmarks = get_standard_benchmarks()
    console.print(f"[bold cyan]Loaded {len(benchmarks)} standard ADP benchmark tasks.[/bold cyan]\n")

    # 1. Evaluate single-harness overfitted model
    console.print("[bold yellow]1. Evaluating Naive Single-Harness Model (Trained only on OpenAI format)...[/bold yellow]")
    naive_provider = MockModelProvider(profile="overfitted_single_harness", native_harness="openai")
    naive_eval = MultiHarnessEvaluator(model_provider=naive_provider, native_harness="openai")
    naive_report = await naive_eval.evaluate_suite(benchmarks, model_id="naive-openai-sft")

    console.print(f"  • Native Accuracy: {naive_report.native_accuracy * 100:.1f}%")
    console.print(f"  • Unseen Harnesses Avg: {naive_report.unseen_harnesses_average_accuracy * 100:.1f}%")
    console.print(f"  • Harness Overfitting Coefficient (HOC): [bold red]{naive_report.harness_overfitting_coefficient:.3f}[/bold red]")
    console.print(f"  • Production Safe: [bold red]{naive_report.is_production_safe}[/bold red]\n")

    # 2. Evaluate PolyHarness multi-harness trained model
    console.print("[bold green]2. Evaluating PolyHarness Generalist Model (Trained on ADP Mixture)...[/bold green]")
    generalist_provider = MockModelProvider(profile="polyharness_generalist", native_harness="openai")
    gen_eval = MultiHarnessEvaluator(model_provider=generalist_provider, native_harness="openai")
    gen_report = await gen_eval.evaluate_suite(benchmarks, model_id="polyharness-agent")

    console.print(f"  • Native Accuracy: {gen_report.native_accuracy * 100:.1f}%")
    console.print(f"  • Unseen Harnesses Avg: {gen_report.unseen_harnesses_average_accuracy * 100:.1f}%")
    console.print(f"  • Harness Overfitting Coefficient (HOC): [bold green]{gen_report.harness_overfitting_coefficient:.3f}[/bold green]")
    console.print(f"  • Production Safe: [bold green]{gen_report.is_production_safe}[/bold green]\n")

    # Render side-by-side table
    table = Table(title="Whiteboard Overfitting Resolution Benchmark")
    table.add_column("Metric", style="bold")
    table.add_column("1-Harness Fine-Tuned Model", style="red")
    table.add_column("PolyHarness Trained Model", style="green")

    table.add_row("Harness Overfitting Coefficient (HOC)", f"{naive_report.harness_overfitting_coefficient:.3f}", f"{gen_report.harness_overfitting_coefficient:.3f}")
    table.add_row("Cross-Harness Transfer Score (CHTS)", f"{naive_report.cross_harness_transfer_score:.3f}", f"{gen_report.cross_harness_transfer_score:.3f}")
    table.add_row("Cascading Divergence Rate (CDR)", f"{naive_report.cascading_divergence_rate:.3f}", f"{gen_report.cascading_divergence_rate:.3f}")
    table.add_row("Production Ready Verdict", "COLLAPSED IN PROD", "SAFE FOR PRODUCTION")

    console.print(table)

if __name__ == "__main__":
    asyncio.run(main())
