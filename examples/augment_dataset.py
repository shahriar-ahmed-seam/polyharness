#!/usr/bin/env python3
"""Example: Generating Anti-Overfitting Synthetic Variants (Syntax, Modality, CascadeGuard)."""

from pathlib import Path

from polyharness.schema.serialization import load_trajectory, save_trajectory
from polyharness.synthesis.cascade_guard import CascadeGuard
from polyharness.synthesis.observation import ObservationTransformer
from polyharness.synthesis.perturbation import SyntaxPerturber


def main():
    traj_path = Path(__file__).parent / "trajectories" / "web_ecommerce_search.json"
    traj = load_trajectory(traj_path)
    output_dir = Path("augmented_data")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Original Trajectory: {traj.id} (Steps: {len(traj.steps)})")

    # 1. Tool Syntax Perturbation (camelCase vs snake_case, parameter reordering)
    perturber = SyntaxPerturber(seed=42)
    p_traj = perturber.perturb_trajectory(traj)
    save_trajectory(p_traj, output_dir / f"{p_traj.id}.json")
    print(f"[+] Generated syntax-perturbed variant: {p_traj.id}")

    # 2. Multi-Representation Observation Enrichment
    transformer = ObservationTransformer()
    obs_traj = transformer.populate_multi_representations(traj)
    save_trajectory(obs_traj, output_dir / f"{obs_traj.id}_multiobs.json")
    print("[+] Generated multi-representation variant (HTML + AXTree + Markdown)")

    # 3. CascadeGuard (Off-distribution error state & recovery turn injection)
    guard = CascadeGuard(seed=42)
    c_traj = guard.inject_recovery_turn(traj)
    save_trajectory(c_traj, output_dir / f"{c_traj.id}.json")
    print(f"[+] Generated cascade-guarded variant with recovery turns (Steps: {len(c_traj.steps)})")

    print(f"\n[✓] All augmented variants saved in: {output_dir.resolve()}")

if __name__ == "__main__":
    main()
