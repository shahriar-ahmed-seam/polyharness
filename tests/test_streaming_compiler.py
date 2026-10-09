"""Unit tests for StreamingDatasetCompiler, sharded partitioning, and token budgeting."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

from typer.testing import CliRunner

from polyharness.cli.main import app
from polyharness.schema.serialization import load_trajectory
from polyharness.synthesis.streaming import (
    StreamingDatasetCompiler,
    estimate_tokens,
)

runner = CliRunner()


def test_estimate_tokens():
    text = "Hello world! This is a test trajectory prompt."
    tokens = estimate_tokens(text)
    assert tokens > 0
    dict_payload = {"role": "user", "content": text}
    tokens_dict = estimate_tokens(dict_payload)
    assert tokens_dict >= tokens


def test_streaming_compile_and_sharding(tmp_path: Path):
    traj_dir = Path("examples/trajectories")
    compiler = StreamingDatasetCompiler(
        harness_distribution={"openai": 0.5, "anthropic": 0.5},
        seed=1337,
    )

    trajs = list(compiler.stream_trajectories_from_path(traj_dir))
    assert len(trajs) >= 3

    # Compile with max_records_per_shard=2 to force multiple shards
    out_dir = tmp_path / "sharded_out"
    summary = compiler.compile_to_sharded_files(
        trajectories=trajs,
        output_dir=out_dir,
        base_filename="test_shard",
        max_records_per_shard=2,
        compress=False,
    )

    assert summary.total_records == len(trajs)
    assert summary.total_shards == 2  # 3 records / 2 per shard = 2 shards
    assert len(summary.shard_files) == 2

    # Verify each shard content
    total_lines = 0
    for shard_name in summary.shard_files:
        shard_file = out_dir / shard_name
        assert shard_file.exists()
        with open(shard_file, encoding="utf-8") as f:
            lines = [json.loads(line) for line in f if line.strip()]
            total_lines += len(lines)
            for rec in lines:
                assert "_polyharness_meta" in rec
                assert rec["_polyharness_meta"]["harness"] in {"openai", "anthropic"}
                assert "estimated_tokens" in rec["_polyharness_meta"]

    assert total_lines == len(trajs)

    # Verify manifest
    manifest_file = out_dir / "test_shard_manifest.json"
    assert manifest_file.exists()
    with open(manifest_file, encoding="utf-8") as f:
        manifest = json.load(f)
    assert manifest["total_records"] == len(trajs)
    assert manifest["total_shards"] == 2
    assert "token_stats" in manifest


def test_streaming_gzip_compression(tmp_path: Path):
    traj_dir = Path("examples/trajectories")
    compiler = StreamingDatasetCompiler(seed=42)
    trajs = list(compiler.stream_trajectories_from_path(traj_dir))

    out_dir = tmp_path / "gzip_out"
    summary = compiler.compile_to_sharded_files(
        trajectories=trajs,
        output_dir=out_dir,
        base_filename="compressed_sft",
        max_records_per_shard=5,
        compress=True,
    )

    assert summary.total_shards == 1
    gz_file = out_dir / summary.shard_files[0]
    assert gz_file.name.endswith(".jsonl.gz")
    assert gz_file.exists()

    with gzip.open(gz_file, "rt", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    assert len(records) == len(trajs)


def test_token_budget_filtering(tmp_path: Path):
    traj = load_trajectory("examples/trajectories/web_ecommerce_search.json")
    # Set a tiny max_seq_len to trigger over-budget
    compiler = StreamingDatasetCompiler(max_seq_len=20, seed=42)

    # When skip_over_budget is True, it should drop over-budget items
    records = list(compiler.compile_stream([traj], skip_over_budget=True))
    assert len(records) == 0

    # When skip_over_budget is False, it should retain but flag
    out_dir = tmp_path / "budget_out"
    summary = compiler.compile_to_sharded_files(
        trajectories=[traj],
        output_dir=out_dir,
        max_records_per_shard=10,
        skip_over_budget=False,
    )
    assert summary.total_records == 1
    assert summary.token_stats.over_budget_count == 1


def test_cli_compile_stream(tmp_path: Path):
    out_dir = tmp_path / "cli_stream_out"
    result = runner.invoke(
        app,
        [
            "compile-stream",
            "examples/trajectories",
            "--output-dir",
            str(out_dir),
            "--max-records",
            "2",
        ],
    )
    assert result.exit_code == 0
    assert "Streaming Compilation Telemetry" in result.stdout
    assert "Harness Distribution Breakdown" in result.stdout
    assert (out_dir / "train_mixture_manifest.json").exists()
