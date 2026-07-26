"""Synthetic demo data and a generic CSV interface.

No research data, labels, sample identifiers, or split files are shipped.
"""

from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class SpectrumDataset:
    spectra: list[list[float]]
    labels: list[int]
    mz_axis: list[float]


def make_synthetic_dataset(
    *,
    n_samples: int,
    n_bins: int,
    noise_scale: float,
    signal_scale: float,
    seed: int,
) -> SpectrumDataset:
    """Create a deterministic toy dataset for end-to-end smoke testing."""

    if n_samples < 20:
        raise ValueError("The demo requires at least 20 samples.")
    if n_bins < 32:
        raise ValueError("The demo requires at least 32 spectrum bins.")

    rng = random.Random(seed)
    mz_axis = [float(i) / float(n_bins - 1) for i in range(n_bins)]
    spectra: list[list[float]] = []
    labels: list[int] = []

    class_centers = {
        0: (0.18, 0.43, 0.71),
        1: (0.29, 0.57, 0.84),
    }

    for sample_index in range(n_samples):
        label = sample_index % 2
        spectrum = [rng.random() * noise_scale for _ in range(n_bins)]
        for center in class_centers[label]:
            center_index = int(round(center * (n_bins - 1)))
            center_index += rng.choice((-1, 0, 0, 0, 1))
            amplitude = signal_scale * (0.85 + 0.30 * rng.random())
            for offset, multiplier in ((-1, 0.35), (0, 1.0), (1, 0.35)):
                bin_index = center_index + offset
                if 0 <= bin_index < n_bins:
                    spectrum[bin_index] += amplitude * multiplier

        max_value = max(spectrum) or 1.0
        spectra.append([value / max_value for value in spectrum])
        labels.append(label)

    order = list(range(n_samples))
    rng.shuffle(order)
    return SpectrumDataset(
        spectra=[spectra[index] for index in order],
        labels=[labels[index] for index in order],
        mz_axis=mz_axis,
    )


def load_csv_dataset(path: str | Path) -> SpectrumDataset:
    """Load a generic `label` + spectrum-bin CSV supplied by the caller."""

    csv_path = Path(path)
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        if "label" not in fieldnames:
            raise ValueError("CSV input must contain a 'label' column.")

        feature_names = [
            name for name in fieldnames if name not in {"label", "sample_id"}
        ]
        if not feature_names:
            raise ValueError("CSV input contains no spectrum columns.")

        spectra: list[list[float]] = []
        raw_labels: list[str] = []
        for row in reader:
            spectra.append([float(row[name]) for name in feature_names])
            raw_labels.append(str(row["label"]))

    if len(spectra) < 4:
        raise ValueError("CSV input must contain at least four rows.")
    label_values = sorted(set(raw_labels))
    if len(label_values) != 2:
        raise ValueError("The public demo currently expects binary labels.")

    label_map = {value: index for index, value in enumerate(label_values)}
    labels = [label_map[value] for value in raw_labels]
    n_bins = len(feature_names)
    mz_axis = [float(i) / float(max(1, n_bins - 1)) for i in range(n_bins)]
    return SpectrumDataset(spectra=spectra, labels=labels, mz_axis=mz_axis)


def subset(dataset: SpectrumDataset, indices: Sequence[int]) -> SpectrumDataset:
    return SpectrumDataset(
        spectra=[dataset.spectra[index] for index in indices],
        labels=[dataset.labels[index] for index in indices],
        mz_axis=list(dataset.mz_axis),
    )


def stratified_train_validation_indices(
    labels: Sequence[int],
    *,
    validation_fraction: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    """Build one deterministic, label-stratified train/validation split."""

    if not 0.05 <= validation_fraction <= 0.5:
        raise ValueError("validation_fraction must be between 0.05 and 0.5.")

    rng = random.Random(seed)
    by_label: dict[int, list[int]] = {}
    for index, label in enumerate(labels):
        by_label.setdefault(int(label), []).append(index)

    train_indices: list[int] = []
    validation_indices: list[int] = []
    for label_indices in by_label.values():
        shuffled = list(label_indices)
        rng.shuffle(shuffled)
        validation_count = max(1, int(round(len(shuffled) * validation_fraction)))
        validation_indices.extend(shuffled[:validation_count])
        train_indices.extend(shuffled[validation_count:])

    train_indices.sort()
    validation_indices.sort()
    return train_indices, validation_indices
