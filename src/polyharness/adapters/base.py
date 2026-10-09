"""Base Harness Adapter interface for PolyHarness."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from polyharness.schema.adp import Trajectory


class BaseAdapter(ABC):
    """Abstract adapter providing bidirectional conversion between ADP and target harness formats."""

    adapter_id: str
    description: str

    @abstractmethod
    def render(self, trajectory: Trajectory, **kwargs: Any) -> dict[str, Any]:
        """Convert a canonical ADP Trajectory into target harness prompt/payload format."""

    @abstractmethod
    def parse(self, raw_data: dict[str, Any], **kwargs: Any) -> Trajectory:
        """Parse raw harness output or recording into canonical ADP Trajectory."""

    @abstractmethod
    def to_sft_record(self, trajectory: Trajectory, **kwargs: Any) -> dict[str, Any]:
        """Convert trajectory into a supervised fine-tuning record (e.g. ChatML/messages list)."""
