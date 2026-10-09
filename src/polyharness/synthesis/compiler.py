"""Multi-Harness SFT Dataset Compiler.

Compiles canonical ADP trajectories into multi-harness fine-tuning datasets
ready for training with HuggingFace TRL (SFTTrainer), Axolotl, Unsloth, or OpenAI Fine-Tuning.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from polyharness.adapters import get_adapter
from polyharness.schema.adp import Trajectory


class DatasetCompiler:
    """Compiles ADP trajectories into production-ready SFT datasets across multiple harnesses."""

    def __init__(self, seed: int | None = None):
        if seed is not None:
            random.seed(seed)

    def compile_single_harness(
        self,
        trajectories: list[Trajectory],
        target_harness: str = "openai",
    ) -> list[dict[str, Any]]:
        """Compile all trajectories into a single target harness SFT format."""
        adapter = get_adapter(target_harness)
        records = []
        for traj in trajectories:
            rec = adapter.to_sft_record(traj)
            rec["_polyharness_meta"] = {
                "trajectory_id": traj.id,
                "harness": target_harness,
                "is_recovery": traj.recovery_turn_count() > 0,
            }
            records.append(rec)
        return records

    def compile_multi_harness_mixture(
        self,
        trajectories: list[Trajectory],
        harness_distribution: dict[str, float] | None = None,
    ) -> list[dict[str, Any]]:
        """Compile trajectories into a multi-harness mixture dataset to prevent harness overfitting.

        Default distribution:
          - openai: 0.35
          - hermes: 0.30
          - anthropic: 0.20
          - react: 0.15
        """
        if harness_distribution is None:
            harness_distribution = {
                "openai": 0.35,
                "hermes": 0.30,
                "anthropic": 0.20,
                "react": 0.15,
            }

        # Normalize weights
        total_w = sum(harness_distribution.values())
        harnesses = list(harness_distribution.keys())
        weights = [harness_distribution[h] / total_w for h in harnesses]

        compiled_records = []
        for traj in trajectories:
            chosen_harness = random.choices(harnesses, weights=weights, k=1)[0]
            adapter = get_adapter(chosen_harness)
            rec = adapter.to_sft_record(traj)
            rec["_polyharness_meta"] = {
                "trajectory_id": traj.id,
                "harness": chosen_harness,
                "is_recovery": traj.recovery_turn_count() > 0,
            }
            compiled_records.append(rec)

        return compiled_records

    def export_to_file(
        self,
        records: list[dict[str, Any]],
        output_path: str | Path,
        format_type: str = "jsonl",
    ) -> Path:
        """Export compiled SFT records to JSONL or JSON."""
        dest = Path(output_path)
        dest.parent.mkdir(parents=True, exist_ok=True)

        if format_type.lower() == "jsonl":
            with open(dest, "w", encoding="utf-8") as f:
                f.writelines(json.dumps(rec) + "\n" for rec in records)
        else:
            with open(dest, "w", encoding="utf-8") as f:
                json.dump(records, f, indent=2)

        return dest
