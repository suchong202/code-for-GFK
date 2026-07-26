#!/usr/bin/env python3
"""Train the dependency-free public workflow preview."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from gfk_public.config import load_config
from gfk_public.pipeline import train_preview


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/demo.yaml")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data", default=None)
    parser.add_argument("--output-dir", default="artifacts")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    metrics, checkpoint_path = train_preview(
        config,
        seed=args.seed,
        data_path=args.data,
        output_dir=args.output_dir,
    )
    print("Public placeholder training complete.")
    print(json.dumps(metrics, indent=2, sort_keys=True))
    print(f"Demo checkpoint: {checkpoint_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
