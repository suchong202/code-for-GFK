"""Feature extraction through the protected-module interface."""

from __future__ import annotations

from typing import Sequence

from .peak_selector import GNNPeakSelector


def extract_feature_vector(
    spectrum: Sequence[float],
    mz_axis: Sequence[float],
    selector: GNNPeakSelector,
) -> list[float]:
    """Flatten fixed-count peak tokens into a demo feature vector."""

    selection = selector.select(spectrum, mz_axis)
    features: list[float] = []
    for normalized_mz, normalized_intensity in selection.tokens:
        features.extend((normalized_mz, normalized_intensity))

    required_size = selector.selected_count * 2
    if len(features) < required_size:
        features.extend([0.0] * (required_size - len(features)))
    return features[:required_size]


def extract_feature_matrix(
    spectra: Sequence[Sequence[float]],
    mz_axis: Sequence[float],
    selector: GNNPeakSelector,
) -> list[list[float]]:
    return [
        extract_feature_vector(spectrum, mz_axis, selector)
        for spectrum in spectra
    ]
