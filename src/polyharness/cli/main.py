"""PolyHarness CLI: Developer toolchain for Agent Data Protocol & Multi-Harness Evals."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from polyharness import __version__
from polyharness.adapters import get_adapter, list_adapters
from polyharness.eval.benchmarks import get_standard_benchmarks
from polyharness.eval.runner import MockModelProvider, MultiHarnessEvaluator
from polyharness.schema.serialization import load_trajectory, save_trajectory
from polyharness.schema.validator import TrajectoryValidator
from polyharness.synthesis.cascade_guard import CascadeGuard
from polyharness.synthesis.compiler import DatasetCompiler
from polyharness.synthesis.observation import ObservationTransformer
from polyharness.synthesis.perturbation import SyntaxPerturber

app = typer.Typer(
    name="polyharness",
    help="PolyHarness: Trajectory Interlingua (ADP) & Multi-Harness Evaluation Suite",
    no_args_is_help=True,
)
console = Console()


@app.command()
def info():
    """Display PolyHarness environment status and supported adapters."""
    console.print(
        Panel(
            f"[bold cyan]PolyHarness v{__version__}[/bold cyan]\n"
            "[italic]The Open-Standard Trajectory Interlingua (ADP) & Multi-Harness Evaluation Suite[/italic]\n\n"
            "[green]Whiteboard Principles Solved:[/green]\n"
            "  1. Tool Syntax & Primitives Invariance\n"
            "  2. Observation Format Modality Decoupling\n"
            "  3. Teacher Forcing Cascade Mitigation",
            title="System Info",
            border_style="cyan",
        )
    )

    table = Table(title="Supported Harness Adapters")
    table.add_column("Adapter ID", style="bold green")
    table.add_column("Description", style="white")

    for adp in list_adapters():
        table.add_row(adp["id"], adp["description"])

    console.print(table)


@app.command()
def validate(file_path: Path):
    """Validate an ADP trajectory file and calculate harness overfitting risk."""
    if not file_path.exists():
        console.print(f"[bold red]File not found:[/bold red] {file_path}")
        raise typer.Exit(code=1)

    try:
        traj = load_trajectory(file_path)
    except Exception as e:
        console.print(f"[bold red]Failed to parse trajectory JSON:[/bold red] {e}")
        raise typer.Exit(code=1)

    report = TrajectoryValidator.validate(traj)

    status_str = "[bold green]VALID[/bold green]" if report.is_valid else "[bold red]INVALID[/bold red]"
    console.print(Panel(
        f"Trajectory ID: [bold]{report.trajectory_id}[/bold]\n"
        f"Task: {report.task}\n"
        f"Steps: {report.total_steps}\n"
        f"Status: {status_str}\n"
        f"Quality Score: [bold yellow]{report.quality_score}/100[/bold yellow]",
        title="ADP Trajectory Validation",
        border_style="green" if report.is_valid else "red",
    ))

    # Overfitting Risks
    risk_table = Table(title="Whiteboard Harness Overfitting Risks")
    risk_table.add_column("Failure Mode", style="cyan")
    risk_table.add_column("Risk Score", style="bold")
    risk_table.add_column("Status", style="bold")

    for cat, score in report.harness_overfitting_risks.items():
        status = "[red]HIGH RISK[/red]" if score > 0.5 else "[green]LOW RISK[/green]"
        risk_table.add_row(cat, f"{score:.2f}", status)

    console.print(risk_table)

    if report.issues:
        issue_table = Table(title="Diagnostic Issues & Recommendations")
        issue_table.add_column("Severity", style="bold")
        issue_table.add_column("Category")
        issue_table.add_column("Message")
        issue_table.add_column("Recommendation", style="italic yellow")

        for issue in report.issues:
            sev_color = "red" if issue.severity == "error" else "yellow"
            issue_table.add_row(
                f"[{sev_color}]{issue.severity.upper()}[/{sev_color}]",
                issue.category,
                issue.message,
                issue.recommendation,
            )
        console.print(issue_table)


@app.command()
def render(file_path: Path, harness: str = "openai"):
    """Render an ADP trajectory into a target harness format (openai, anthropic, hermes, react, browsergym, langchain)."""
    if not file_path.exists():
        console.print(f"[bold red]File not found:[/bold red] {file_path}")
        raise typer.Exit(code=1)

    traj = load_trajectory(file_path)
    adapter = get_adapter(harness)
    rendered = adapter.render(traj)

    console.print(f"[bold green]Rendered into '{harness}' harness:[/bold green]")
    console.print(json.dumps(rendered, indent=2))


@app.command()
def augment(
    file_path: Path,
    output_dir: Path = Path("augmented_data"),
    perturb_syntax: bool = True,
    multi_obs: bool = True,
    cascade_guard: bool = True,
):
    """Generate anti-overfitting synthesized variants for a trajectory."""
    traj = load_trajectory(file_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_files = []

    # 1. Syntax Perturbation
    if perturb_syntax:
        perturber = SyntaxPerturber(seed=42)
        p_traj = perturber.perturb_trajectory(traj)
        out_p = output_dir / f"{traj.id}_perturbed_syntax.json"
        save_trajectory(p_traj, out_p)
        generated_files.append(out_p)

    # 2. Multi-representation observations
    if multi_obs:
        transformer = ObservationTransformer()
        obs_traj = transformer.populate_multi_representations(traj)
        out_obs = output_dir / f"{traj.id}_multi_obs.json"
        save_trajectory(obs_traj, out_obs)
        generated_files.append(out_obs)

    # 3. Cascade Guard (Teacher forcing recovery turns)
    if cascade_guard:
        guard = CascadeGuard(seed=42)
        c_traj = guard.inject_recovery_turn(traj)
        out_c = output_dir / f"{traj.id}_cascade_guarded.json"
        save_trajectory(c_traj, out_c)
        generated_files.append(out_c)

    console.print(f"[bold green]Successfully generated {len(generated_files)} augmented trajectories in:[/bold green] {output_dir}")
    for f in generated_files:
        console.print(f"  - {f}")


@app.command()
def compile(
    file_path: Path,
    output_file: Path = Path("exported_sft/training_data.jsonl"),
    harness: str = "openai",
    mixture: bool = False,
):
    """Compile ADP trajectories into an SFT fine-tuning dataset (JSONL)."""
    traj = load_trajectory(file_path)
    compiler = DatasetCompiler(seed=42)

    if mixture:
        records = compiler.compile_multi_harness_mixture([traj])
    else:
        records = compiler.compile_single_harness([traj], target_harness=harness)

    compiler.export_to_file(records, output_file, format_type="jsonl")
    mode_str = "Multi-Harness Mixture" if mixture else f"Single Harness ({harness})"
    console.print(f"[bold green]Exported {len(records)} records ({mode_str}) to:[/bold green] {output_file}")


@app.command()
def eval(
    profile: str = typer.Option("overfitted_single_harness", help="Model profile: 'overfitted_single_harness' or 'polyharness_generalist'"),
    native_harness: str = typer.Option("openai", help="Native training harness"),
    output: Path | None = typer.Option(None, help="Optional JSON output path for the report"),
):
    """Run the Multi-Harness Overfitting Audit benchmark and calculate the Harness Overfitting Coefficient (HOC)."""
    benchmarks = get_standard_benchmarks()
    provider = MockModelProvider(profile=profile, native_harness=native_harness)
    evaluator = MultiHarnessEvaluator(model_provider=provider, native_harness=native_harness)

    console.print(f"[bold cyan]Running Multi-Harness Audit for profile: '{profile}' (Native: '{native_harness}')...[/bold cyan]")
    report = asyncio.run(evaluator.evaluate_suite(benchmarks, model_id=f"model-{profile}"))

    # Render summary table
    table = Table(title="Multi-Harness Evaluation Breakdown")
    table.add_column("Harness ID", style="bold")
    table.add_column("Type")
    table.add_column("Accuracy", style="bold")
    table.add_column("Syntax Errors")
    table.add_column("Cascading Crashes")

    for h_id, res in report.detailed_harness_results.items():
        type_str = "[cyan]Native (Trained)[/cyan]" if res.is_native_training_harness else "[magenta]Unseen Eval[/magenta]"
        acc_color = "green" if res.accuracy >= 0.8 else ("yellow" if res.accuracy >= 0.5 else "red")
        table.add_row(
            h_id,
            type_str,
            f"[{acc_color}]{res.accuracy * 100:.1f}%[/{acc_color}]",
            str(res.malformed_tool_calls),
            str(res.cascading_failure_count),
        )
    console.print(table)

    # Scorecard
    verdict = "[bold green]SAFE FOR PRODUCTION[/bold green]" if report.is_production_safe else "[bold red]FAILED AUDIT: HARNESS OVERFITTED[/bold red]"
    console.print(Panel(
        f"Native Accuracy: [bold]{report.native_accuracy * 100:.1f}%[/bold]\n"
        f"Unseen Harnesses Avg Accuracy: [bold]{report.unseen_harnesses_average_accuracy * 100:.1f}%[/bold]\n"
        f"Harness Overfitting Coefficient (HOC): [bold red]{report.harness_overfitting_coefficient:.3f}[/bold red]\n"
        f"Cross-Harness Transfer Score (CHTS): [bold]{report.cross_harness_transfer_score:.3f}[/bold]\n"
        f"Cascading Divergence Rate (CDR): [bold yellow]{report.cascading_divergence_rate:.3f}[/bold yellow]\n\n"
        f"Production Readiness: {verdict}",
        title="PolyHarness Audit Scorecard",
        border_style="green" if report.is_production_safe else "red",
    ))

    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w", encoding="utf-8") as f:
            f.write(report.model_dump_json(indent=2))
        console.print(f"[bold green]Report saved to:[/bold green] {output}")


@app.command()
def serve(host: str = "0.0.0.0", port: int = 8000):
    """Launch the PolyHarness Studio Web UI and REST API server."""
    import uvicorn
    console.print(f"[bold cyan]Starting PolyHarness Studio at http://{host}:{port}[/bold cyan]")
    uvicorn.run("polyharness.server.app:app", host=host, port=port, reload=False)


def main():
    app()


if __name__ == "__main__":
    main()
