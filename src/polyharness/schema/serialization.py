"""Serialization utilities for ADP trajectories."""

from __future__ import annotations

import json
from pathlib import Path

from polyharness.schema.adp import Trajectory


def load_trajectory(source: str | Path | dict) -> Trajectory:
    """Load a Trajectory instance from a JSON file path, JSON string, or dict."""
    if isinstance(source, dict):
        return Trajectory.model_validate(source)
    if isinstance(source, Path) or (isinstance(source, str) and (Path(source).exists() or source.endswith(".json"))):
        path = Path(source)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return Trajectory.model_validate(data)
    # Parse as JSON string
    data = json.loads(source)
    return Trajectory.model_validate(data)


def save_trajectory(trajectory: Trajectory, destination: str | Path, indent: int = 2) -> Path:
    """Save an ADP Trajectory to a JSON file."""
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(trajectory.model_dump_json(indent=indent))
    return path


def load_trajectories_jsonl(file_path: str | Path) -> list[Trajectory]:
    """Read a JSONL file containing multiple ADP trajectories."""
    path = Path(file_path)
    trajectories: list[Trajectory] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            trajectories.append(Trajectory.model_validate(data))
    return trajectories


def save_trajectories_jsonl(trajectories: list[Trajectory], file_path: str | Path) -> Path:
    """Export a list of ADP trajectories to a JSONL file."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(t.model_dump_json() + "\n" for t in trajectories)
    return path
