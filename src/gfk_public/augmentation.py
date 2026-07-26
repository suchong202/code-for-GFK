"""Shape-compatible public augmentation placeholder."""

from __future__ import annotations

import random
from typing import Sequence


class PublicAugmentationPlaceholder:
    """Apply generic intensity jitter, not the protected augmentation policy."""

    implementation = "public_jitter_placeholder"

    def __init__(self, jitter_std: float, seed: int):
        if jitter_std < 0:
            raise ValueError("jitter_std must be non-negative.")
        self.jitter_std = float(jitter_std)
        self.rng = random.Random(seed)

    def __call__(self, spectrum: Sequence[float]) -> list[float]:
        values = [float(value) for value in spectrum]
        if self.jitter_std == 0.0:
            return values

        augmented = [
            max(0.0, value + self.rng.gauss(0.0, self.jitter_std))
            for value in values
        ]
        max_value = max(augmented) if augmented else 0.0
        if max_value > 0.0:
            augmented = [value / max_value for value in augmented]
        return augmented
