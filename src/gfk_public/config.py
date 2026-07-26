"""Configuration helpers for the dependency-free public preview."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_config(path: str | Path) -> dict[str, Any]:
    """Load the JSON-compatible YAML used by the public demo.

    JSON is a valid subset of YAML. Keeping the preview config in that subset
    lets the repository run without adding a YAML package.
    """

    config_path = Path(path)
    try:
        with config_path.open("r", encoding="utf-8") as handle:
            config = json.load(handle)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Config file not found: {config_path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"{config_path} must use the JSON-compatible YAML subset: {exc}"
        ) from exc

    required = {"demo_data", "selector", "augmentation", "training"}
    missing = sorted(required.difference(config))
    if missing:
        raise ValueError(f"Missing config sections: {', '.join(missing)}")
    return config
