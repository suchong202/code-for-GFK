"""Small dependency-free classifier used only by the public demo."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        exp_value = math.exp(-value)
        return 1.0 / (1.0 + exp_value)
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


@dataclass
class LogisticDemoClassifier:
    """Binary logistic regression trained with full-batch gradient descent."""

    weights: list[float]
    bias: float = 0.0

    @classmethod
    def create(cls, n_features: int) -> "LogisticDemoClassifier":
        if n_features <= 0:
            raise ValueError("n_features must be positive.")
        return cls(weights=[0.0] * n_features)

    def fit(
        self,
        features: Sequence[Sequence[float]],
        labels: Sequence[int],
        *,
        epochs: int,
        learning_rate: float,
        l2: float,
    ) -> None:
        if not features:
            raise ValueError("Training features must not be empty.")
        if len(features) != len(labels):
            raise ValueError("features and labels must have equal length.")

        sample_count = float(len(features))
        for _ in range(int(epochs)):
            weight_gradient = [0.0] * len(self.weights)
            bias_gradient = 0.0
            for row, label in zip(features, labels):
                probability = self.predict_probability(row)
                error = probability - float(label)
                for feature_index, value in enumerate(row):
                    weight_gradient[feature_index] += error * float(value)
                bias_gradient += error

            for feature_index in range(len(self.weights)):
                gradient = (
                    weight_gradient[feature_index] / sample_count
                    + float(l2) * self.weights[feature_index]
                )
                self.weights[feature_index] -= float(learning_rate) * gradient
            self.bias -= float(learning_rate) * bias_gradient / sample_count

    def predict_probability(self, row: Sequence[float]) -> float:
        if len(row) != len(self.weights):
            raise ValueError("Feature width does not match the checkpoint.")
        logit = self.bias + sum(
            weight * float(value) for weight, value in zip(self.weights, row)
        )
        return _sigmoid(logit)

    def predict_probabilities(
        self,
        features: Sequence[Sequence[float]],
    ) -> list[float]:
        return [self.predict_probability(row) for row in features]

    def to_dict(self) -> dict[str, object]:
        return {"weights": list(self.weights), "bias": float(self.bias)}

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "LogisticDemoClassifier":
        weights = [float(value) for value in payload["weights"]]  # type: ignore[index]
        return cls(weights=weights, bias=float(payload["bias"]))  # type: ignore[arg-type]
