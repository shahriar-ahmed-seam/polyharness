"""Streaming and Sharded Dataset Compiler for Enterprise Fine-Tuning.

Provides zero-copy, memory-bounded trajectory streaming, deterministic harness mixture
distribution, token budgeting, and multi-shard partitioned export (.jsonl / .jsonl.gz).
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from polyharness.adapters import get_adapter
from polyharness.schema.adp import Trajectory
from polyharness.schema.serialization import load_trajectory


@dataclass
class TokenBudgetStats:
    """Token length distribution statistics for compiled SFT datasets."""

    total_tokens: int = 0
    min_tokens: int = 0
    max_tokens: int = 0
    mean_tokens: float = 0.0
    p50_tokens: int = 0
    p95_tokens: int = 0
    over_budget_count: int = 0


@dataclass
class StreamingCompilationSummary:
    """Summary telemetry emitted by StreamingDatasetCompiler."""

    total_records: int
    total_shards: int
    harness_counts: dict[str, int]
    recovery_count: int
    recovery_ratio: float
    token_stats: TokenBudgetStats
    shard_files: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_records": self.total_records,
            "total_shards": self.total_shards,
            "harness_counts": self.harness_counts,
            "recovery_count": self.recovery_count,
            "recovery_ratio": round(self.recovery_ratio, 4),
            "token_stats": {
                "total_tokens": self.token_stats.total_tokens,
                "min_tokens": self.token_stats.min_tokens,
                "max_tokens": self.token_stats.max_tokens,
                "mean_tokens": round(self.token_stats.mean_tokens, 2),
                "p50_tokens": self.token_stats.p50_tokens,
                "p95_tokens": self.token_stats.p95_tokens,
                "over_budget_count": self.token_stats.over_budget_count,
            },
            "shard_files": self.shard_files,
        }


def estimate_tokens(obj: Any) -> int:
    """Estimate token count for an arbitrary SFT payload.

    Uses conservative ~3.7 characters per token heuristic for mixed code/JSON/text.
    """
    if isinstance(obj, str):
        text_len = len(obj)
    else:
        text_len = len(json.dumps(obj, ensure_ascii=False))
    return max(1, math.ceil(text_len / 3.7))


class StreamingDatasetCompiler:
    """High-throughput, memory-bounded streaming dataset compiler.

    Compiles arbitrarily large collections of ADP trajectories into sharded SFT records
    with deterministic mixture weighting, token auditing, and zero memory spikes.
    """

    def __init__(
        self,
        harness_distribution: dict[str, float] | None = None,
        max_seq_len: int | None = None,
        seed: int = 42,
    ):
        if harness_distribution is None:
            self.harness_distribution = {
                "openai": 0.35,
                "hermes": 0.30,
                "anthropic": 0.20,
                "smolagents": 0.15,
            }
        else:
            self.harness_distribution = harness_distribution

        # Normalize mixture weights
        total_w = sum(self.harness_distribution.values())
        if total_w <= 0:
            msg = "Sum of harness distribution weights must be positive."
            raise ValueError(msg)
        self.normalized_weights = {
            h: w / total_w for h, w in self.harness_distribution.items()
        }
        self.harnesses = sorted(self.normalized_weights.keys())
        self.cumulative_weights = []
        cum = 0.0
        for h in self.harnesses:
            cum += self.normalized_weights[h]
            self.cumulative_weights.append(cum)

        self.max_seq_len = max_seq_len
        self.seed = seed

    def _select_harness(self, trajectory_id: str) -> str:
        """Deterministically select target harness for a trajectory using SHA256 hash.

        Guarantees that identical trajectories receive the identical harness assignment
        across distributed nodes without inter-process coordination.
        """
        hasher = hashlib.sha256(f"{self.seed}:{trajectory_id}".encode("utf-8"))
        # Use first 8 bytes as an integer fraction in [0.0, 1.0)
        val = int(hasher.hexdigest()[:8], 16) / 0xFFFFFFFF
        for h, cum in zip(self.harnesses, self.cumulative_weights, strict=False):
            if val <= cum:
                return h
        return self.harnesses[-1]

    @staticmethod
    def stream_trajectories_from_path(
        path: str | Path,
    ) -> Iterator[Trajectory]:
        """Lazily stream Trajectory objects from a file or directory tree."""
        target = Path(path)
        if target.is_file():
            yield from StreamingDatasetCompiler._stream_from_single_file(target)
        elif target.is_dir():
            for file_path in sorted(target.rglob("*")):
                if file_path.is_file() and file_path.suffix in {".json", ".jsonl", ".gz"}:
                    yield from StreamingDatasetCompiler._stream_from_single_file(file_path)

    @staticmethod
    def _stream_from_single_file(path: Path) -> Iterator[Trajectory]:
        """Stream trajectories from a JSON file, JSONL file, or gzipped file."""
        if path.name.endswith(".jsonl.gz") or path.name.endswith(".gz"):
            with gzip.open(path, "rt", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        data = json.loads(line)
                        yield Trajectory.model_validate(data)
        elif path.suffix == ".jsonl":
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        data = json.loads(line)
                        yield Trajectory.model_validate(data)
        elif path.suffix == ".json":
            # Can be a single trajectory or a list of trajectories
            with open(path, encoding="utf-8") as f:
                content = json.load(f)
            if isinstance(content, list):
                for item in content:
                    yield Trajectory.model_validate(item)
            else:
                yield load_trajectory(path)

    def compile_stream(
        self,
        trajectories: Iterable[Trajectory],
        skip_over_budget: bool = False,
    ) -> Iterator[tuple[dict[str, Any], int, str]]:
        """Yield (sft_record, estimated_tokens, chosen_harness) tuple lazily for each trajectory."""
        for traj in trajectories:
            chosen_harness = self._select_harness(traj.id)
            adapter = get_adapter(chosen_harness)
            sft_rec = adapter.to_sft_record(traj)
            is_recovery = traj.recovery_turn_count() > 0

            sft_rec["_polyharness_meta"] = {
                "trajectory_id": traj.id,
                "harness": chosen_harness,
                "is_recovery": is_recovery,
            }

            tokens = estimate_tokens(sft_rec)
            sft_rec["_polyharness_meta"]["estimated_tokens"] = tokens

            if self.max_seq_len is not None and tokens > self.max_seq_len:
                if skip_over_budget:
                    continue

            yield sft_rec, tokens, chosen_harness

    def compile_to_sharded_files(
        self,
        trajectories: Iterable[Trajectory],
        output_dir: str | Path,
        base_filename: str = "train_mixture",
        max_records_per_shard: int = 5000,
        compress: bool = False,
        skip_over_budget: bool = False,
    ) -> StreamingCompilationSummary:
        """Stream trajectories and write partitioned shards directly to disk with minimal RAM footprint."""
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        shard_idx = 0
        current_records_in_shard = 0
        total_records = 0
        recovery_count = 0
        harness_counts = {h: 0 for h in self.harnesses}
        all_token_counts: list[int] = []
        shard_files: list[str] = []
        over_budget_count = 0

        current_file = None

        def open_new_shard(idx: int) -> tuple[Any, str]:
            ext = ".jsonl.gz" if compress else ".jsonl"
            filename = f"{base_filename}_part_{idx:05d}{ext}"
            file_dest = out_path / filename
            if compress:
                f_out = gzip.open(file_dest, "wt", encoding="utf-8")
            else:
                f_out = open(file_dest, "w", encoding="utf-8")
            return f_out, filename

        try:
            for sft_rec, tokens, harness in self.compile_stream(
                trajectories, skip_over_budget=skip_over_budget
            ):
                if current_file is None or current_records_in_shard >= max_records_per_shard:
                    if current_file is not None:
                        current_file.close()
                    current_file, fname = open_new_shard(shard_idx)
                    shard_files.append(fname)
                    shard_idx += 1
                    current_records_in_shard = 0

                # Write line immediately
                current_file.write(json.dumps(sft_rec, ensure_ascii=False) + "\n")
                current_records_in_shard += 1
                total_records += 1

                # Update stats
                harness_counts[harness] = harness_counts.get(harness, 0) + 1
                if sft_rec["_polyharness_meta"]["is_recovery"]:
                    recovery_count += 1

                all_token_counts.append(tokens)
                if self.max_seq_len is not None and tokens > self.max_seq_len:
                    over_budget_count += 1

        finally:
            if current_file is not None:
                current_file.close()

        # Compute token statistics
        if all_token_counts:
            sorted_tokens = sorted(all_token_counts)
            total_tokens = sum(sorted_tokens)
            min_t = sorted_tokens[0]
            max_t = sorted_tokens[-1]
            mean_t = total_tokens / len(sorted_tokens)
            p50_idx = int(0.50 * len(sorted_tokens))
            p95_idx = min(int(0.95 * len(sorted_tokens)), len(sorted_tokens) - 1)
            p50_t = sorted_tokens[p50_idx]
            p95_t = sorted_tokens[p95_idx]
        else:
            total_tokens = min_t = max_t = p50_t = p95_t = 0
            mean_t = 0.0

        token_stats = TokenBudgetStats(
            total_tokens=total_tokens,
            min_tokens=min_t,
            max_tokens=max_t,
            mean_tokens=mean_t,
            p50_tokens=p50_t,
            p95_tokens=p95_t,
            over_budget_count=over_budget_count,
        )

        recovery_ratio = recovery_count / total_records if total_records > 0 else 0.0

        summary = StreamingCompilationSummary(
            total_records=total_records,
            total_shards=len(shard_files),
            harness_counts=harness_counts,
            recovery_count=recovery_count,
            recovery_ratio=recovery_ratio,
            token_stats=token_stats,
            shard_files=shard_files,
        )

        # Write manifest.json
        manifest_path = out_path / f"{base_filename}_manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(summary.to_dict(), f, indent=2)

        return summary
