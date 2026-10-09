"""Tests for PolyHarness CLI commands."""

from pathlib import Path

from typer.testing import CliRunner

from polyharness.cli.main import app

runner = CliRunner()


def test_cli_info():
    result = runner.invoke(app, ["info"])
    assert result.exit_code == 0
    assert "PolyHarness" in result.output
    assert "openai" in result.output


def test_cli_validate():
    traj_file = Path(__file__).parent.parent / "examples" / "trajectories" / "web_ecommerce_search.json"
    result = runner.invoke(app, ["validate", str(traj_file)])
    assert result.exit_code == 0
    assert "ADP Trajectory Validation" in result.output


def test_cli_render():
    traj_file = Path(__file__).parent.parent / "examples" / "trajectories" / "web_ecommerce_search.json"
    result = runner.invoke(app, ["render", str(traj_file), "--harness", "openai"])
    assert result.exit_code == 0
    assert "search_products" in result.output


def test_cli_eval():
    result = runner.invoke(app, ["eval", "--profile", "polyharness_generalist"])
    assert result.exit_code == 0
    assert "PolyHarness Audit Scorecard" in result.output
    assert "SAFE FOR PRODUCTION" in result.output
