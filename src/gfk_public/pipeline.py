"""End-to-end training and evaluation for the limited public preview."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .augmentation import PublicAugmentationPlaceholder
from .data import (
    SpectrumDataset,
    load_csv_dataset,
    make_synthetic_dataset,
    stratified_train_validation_indices,
    subset,
)
from .features import extract_feature_matrix
from .metrics import validation_metrics
from .model import LogisticDemoClassifier
from .peak_selector import GNNPeakSelector


def _load_dataset(
    config: dict[str, Any],
    *,
    seed: int,
    data_path: str | None,
) -> SpectrumDataset:
    if data_path:
        return load_csv_dataset(data_path)
    demo = config["demo_data"]
    return make_synthetic_dataset(
        n_samples=int(demo["n_samples"]),
        n_bins=int(demo["n_bins"]),
        noise_scale=float(demo["noise_scale"]),
        signal_scale=float(demo["signal_scale"]),
        seed=seed,
    )


def _config_digest(config: dict[str, Any]) -> str:
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _split_dataset(
    dataset: SpectrumDataset,
    config: dict[str, Any],
    seed: int,
) -> tuple[SpectrumDataset, SpectrumDataset]:
    train_indices, validation_indices = stratified_train_validation_indices(
        dataset.labels,
        validation_fraction=float(config["demo_data"]["validation_fraction"]),
        seed=seed,
    )
    return subset(dataset, train_indices), subset(dataset, validation_indices)


def train_preview(
    config: dict[str, Any],
    *,
    seed: int,
    data_path: str | None,
    output_dir: str | Path,
) -> tuple[dict[str, float], Path]:
    dataset = _load_dataset(config, seed=seed, data_path=data_path)
    train_data, validation_data = _split_dataset(dataset, config, seed)

    selector = GNNPeakSelector(
        selected_count=int(config["selector"]["selected_count"])
    )
    train_spectra = train_data.spectra
    augmentation_config = config["augmentation"]
    if bool(augmentation_config.get("enabled", True)):
        augmenter = PublicAugmentationPlaceholder(
            jitter_std=float(augmentation_config["jitter_std"]),
            seed=seed + 1,
        )
        train_spectra = [augmenter(spectrum) for spectrum in train_spectra]

    train_features = extract_feature_matrix(
        train_spectra, train_data.mz_axis, selector
    )
    validation_features = extract_feature_matrix(
        validation_data.spectra, validation_data.mz_axis, selector
    )

    model = LogisticDemoClassifier.create(len(train_features[0]))
    training_config = config["training"]
    model.fit(
        train_features,
        train_data.labels,
        epochs=int(training_config["epochs"]),
        learning_rate=float(training_config["learning_rate"]),
        l2=float(training_config["l2"]),
    )

    probabilities = model.predict_probabilities(validation_features)
    metrics = validation_metrics(
        validation_data.labels,
        probabilities,
        threshold=float(config.get("decision_threshold", 0.5)),
    )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_path / "demo_checkpoint.json"
    checkpoint = {
        "release_type": "public_placeholder_demo",
        "contains_research_weights": False,
        "seed": int(seed),
        "config_sha256": _config_digest(config),
        "selector": {
            "implementation": selector.implementation,
            "selected_count": selector.selected_count,
        },
        "model": model.to_dict(),
    }
    with checkpoint_path.open("w", encoding="utf-8") as handle:
        json.dump(checkpoint, handle, indent=2, sort_keys=True)
        handle.write("\n")

    metrics_path = output_path / "train_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return metrics, checkpoint_path


def evaluate_preview(
    config: dict[str, Any],
    *,
    seed: int,
    data_path: str | None,
    checkpoint_path: str | Path,
    output_dir: str | Path,
) -> dict[str, float]:
    checkpoint_file = Path(checkpoint_path)
    with checkpoint_file.open("r", encoding="utf-8") as handle:
        checkpoint = json.load(handle)

    if checkpoint.get("release_type") != "public_placeholder_demo":
        raise ValueError("Refusing to load a non-preview checkpoint.")
    if bool(checkpoint.get("contains_research_weights", True)):
        raise ValueError("The public evaluator accepts demo checkpoints only.")
    if int(checkpoint["seed"]) != int(seed):
        raise ValueError("Evaluation seed must match the demo checkpoint seed.")
    if checkpoint["config_sha256"] != _config_digest(config):
        raise ValueError("Evaluation config does not match the demo checkpoint.")

    dataset = _load_dataset(config, seed=seed, data_path=data_path)
    _, validation_data = _split_dataset(dataset, config, seed)
    selector = GNNPeakSelector(
        selected_count=int(checkpoint["selector"]["selected_count"])
    )
    validation_features = extract_feature_matrix(
        validation_data.spectra, validation_data.mz_axis, selector
    )
    model = LogisticDemoClassifier.from_dict(checkpoint["model"])
    probabilities = model.predict_probabilities(validation_features)
    metrics = validation_metrics(
        validation_data.labels,
        probabilities,
        threshold=float(config.get("decision_threshold", 0.5)),
    )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    metrics_path = output_path / "evaluation_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return metrics
