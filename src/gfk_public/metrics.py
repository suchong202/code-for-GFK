"""Validation metrics used by the public workflow preview."""

from __future__ import annotations

from typing import Sequence


def binary_auc(labels: Sequence[int], probabilities: Sequence[float]) -> float:
    """Compute binary ROC AUC using average ranks for ties."""

    if len(labels) != len(probabilities) or not labels:
        raise ValueError("labels and probabilities must be non-empty and aligned.")

    ordered = sorted(
        enumerate(probabilities),
        key=lambda item: float(item[1]),
    )
    ranks = [0.0] * len(ordered)
    cursor = 0
    while cursor < len(ordered):
        end = cursor + 1
        while end < len(ordered) and ordered[end][1] == ordered[cursor][1]:
            end += 1
        average_rank = (cursor + 1 + end) / 2.0
        for position in range(cursor, end):
            original_index = ordered[position][0]
            ranks[original_index] = average_rank
        cursor = end

    positive_indices = [index for index, label in enumerate(labels) if int(label) == 1]
    negative_count = sum(1 for label in labels if int(label) == 0)
    positive_count = len(positive_indices)
    if positive_count == 0 or negative_count == 0:
        raise ValueError("AUC requires both classes.")

    positive_rank_sum = sum(ranks[index] for index in positive_indices)
    return (
        positive_rank_sum - positive_count * (positive_count + 1) / 2.0
    ) / float(positive_count * negative_count)


def accuracy(labels: Sequence[int], predictions: Sequence[int]) -> float:
    if len(labels) != len(predictions) or not labels:
        raise ValueError("labels and predictions must be non-empty and aligned.")
    correct = sum(int(target) == int(prediction) for target, prediction in zip(labels, predictions))
    return correct / float(len(labels))


def macro_f1(labels: Sequence[int], predictions: Sequence[int]) -> float:
    if len(labels) != len(predictions) or not labels:
        raise ValueError("labels and predictions must be non-empty and aligned.")

    class_scores: list[float] = []
    for class_value in (0, 1):
        true_positive = sum(
            int(target) == class_value and int(prediction) == class_value
            for target, prediction in zip(labels, predictions)
        )
        false_positive = sum(
            int(target) != class_value and int(prediction) == class_value
            for target, prediction in zip(labels, predictions)
        )
        false_negative = sum(
            int(target) == class_value and int(prediction) != class_value
            for target, prediction in zip(labels, predictions)
        )
        denominator = 2 * true_positive + false_positive + false_negative
        class_scores.append(
            0.0 if denominator == 0 else (2.0 * true_positive) / denominator
        )
    return sum(class_scores) / len(class_scores)


def validation_metrics(
    labels: Sequence[int],
    probabilities: Sequence[float],
    *,
    threshold: float,
) -> dict[str, float]:
    predictions = [int(float(value) >= threshold) for value in probabilities]
    return {
        "val_auc": binary_auc(labels, probabilities),
        "val_macro_f1": macro_f1(labels, predictions),
        "val_acc": accuracy(labels, predictions),
    }
