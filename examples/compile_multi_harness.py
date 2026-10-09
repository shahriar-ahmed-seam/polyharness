#!/usr/bin/env python3
"""Example: Compiling ADP Trajectories into Multi-Harness SFT Datasets."""

from pathlib import Path

from polyharness.schema.serialization import load_trajectory
from polyharness.synthesis.compiler import DatasetCompiler


def main():
    traj_path = Path(__file__).parent / "trajectories" / "web_ecommerce_search.json"
    print(f"Loading canonical ADP trajectory from: {traj_path}")
    traj = load_trajectory(traj_path)

    compiler = DatasetCompiler(seed=42)

    # 1. Compile single target harness (OpenAI ChatML format)
    openai_records = compiler.compile_single_harness([traj], target_harness="openai")
    print(f"\n[+] Compiled single harness (openai): {len(openai_records)} record(s)")
    print(f"Sample keys: {list(openai_records[0].keys())}")

    # 2. Compile Nous-Hermes XML format
    hermes_records = compiler.compile_single_harness([traj], target_harness="hermes")
    print(f"[+] Compiled single harness (hermes): {len(hermes_records)} record(s)")

    # 3. Compile Multi-Harness Mixture (Inoculates against harness overfitting!)
    mixture_distribution = {
        "openai": 0.40,
        "hermes": 0.30,
        "anthropic": 0.20,
        "react": 0.10,
    }
    mixture_records = compiler.compile_multi_harness_mixture([traj], harness_distribution=mixture_distribution)
    print(f"[+] Compiled multi-harness mixture: {len(mixture_records)} record(s)")

    out_file = Path("exported_sft/mixture_train.jsonl")
    compiler.export_to_file(mixture_records, out_file)
    print(f"[✓] Saved compiled mixture dataset to: {out_file}")

if __name__ == "__main__":
    main()
