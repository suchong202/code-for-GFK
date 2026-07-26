"""Public peak-selector interface with a non-GNN fallback.

The proposed selector is a protected research contribution. This file exposes
only the callable contract needed by the runnable preview.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class PeakSelection:
    """Fixed-contract output returned by the selector."""

    indices: list[int]
    tokens: list[list[float]]
    metadata: dict[str, str]


class GNNPeakSelector:
    """Interface-compatible placeholder; this is not the research GNN.

    Protected outline only:

        candidates = PRIVATE_CANDIDATE_STAGE(spectrum)
        graph = PRIVATE_GRAPH_STAGE(candidates)
        states = PRIVATE_MESSAGE_PASSING(graph, candidates)
        scores = PRIVATE_SCORING_STAGE(states)
        indices = PRIVATE_SELECTION_STAGE(scores)

    The public fallback below performs conventional intensity ranking so that
    data flow, shapes, training, and evaluation can be exercised end to end.
    """

    implementation = "public_intensity_placeholder"

    def __init__(self, selected_count: int):
        if selected_count <= 0:
            raise ValueError("selected_count must be positive.")
        self.selected_count = int(selected_count)

    def select(
        self,
        spectrum: Sequence[float],
        mz_axis: Sequence[float],
    ) -> PeakSelection:
        """Select bins and return normalized [m/z, intensity] tokens."""

        values = [float(value) for value in spectrum]
        positions = [float(value) for value in mz_axis]
        if len(values) != len(positions):
            raise ValueError("spectrum and mz_axis must have the same length.")
        if not values:
            raise ValueError("spectrum must not be empty.")

        count = min(self.selected_count, len(values))
        ranked = sorted(range(len(values)), key=lambda index: values[index], reverse=True)
        indices = sorted(ranked[:count])

        mz_min = min(positions)
        mz_span = max(positions) - mz_min
        intensity_max = max(max(values), 0.0)
        tokens: list[list[float]] = []
        for index in indices:
            normalized_mz = (
                (positions[index] - mz_min) / mz_span if mz_span > 0.0 else 0.0
            )
            normalized_intensity = (
                max(values[index], 0.0) / intensity_max
                if intensity_max > 0.0
                else 0.0
            )
            tokens.append([normalized_mz, normalized_intensity])

        return PeakSelection(
            indices=indices,
            tokens=tokens,
            metadata={
                "implementation": self.implementation,
                "research_gnn_included": "false",
            },
        )
