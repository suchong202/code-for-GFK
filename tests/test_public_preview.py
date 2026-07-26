"""Smoke tests for the intentionally limited public preview."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gfk_public.peak_selector import GNNPeakSelector


class PeakSelectorContractTest(unittest.TestCase):
    def test_placeholder_returns_sorted_fixed_count_indices(self) -> None:
        selector = GNNPeakSelector(selected_count=3)
        result = selector.select(
            spectrum=[0.1, 0.9, 0.2, 0.8, 0.7],
            mz_axis=[1.0, 2.0, 3.0, 4.0, 5.0],
        )
        self.assertEqual(result.indices, [1, 3, 4])
        self.assertEqual(len(result.tokens), 3)
        self.assertEqual(result.metadata["research_gnn_included"], "false")


class CommandLineSmokeTest(unittest.TestCase):
    def test_train_then_evaluate(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            train_command = [
                sys.executable,
                str(ROOT / "train.py"),
                "--config",
                str(ROOT / "configs" / "demo.yaml"),
                "--seed",
                "42",
                "--output-dir",
                temp_dir,
            ]
            subprocess.run(train_command, check=True, cwd=ROOT)

            checkpoint = Path(temp_dir) / "demo_checkpoint.json"
            self.assertTrue(checkpoint.exists())
            payload = json.loads(checkpoint.read_text(encoding="utf-8"))
            self.assertFalse(payload["contains_research_weights"])

            evaluate_command = [
                sys.executable,
                str(ROOT / "evaluate.py"),
                "--config",
                str(ROOT / "configs" / "demo.yaml"),
                "--seed",
                "42",
                "--checkpoint",
                str(checkpoint),
                "--output-dir",
                temp_dir,
            ]
            subprocess.run(evaluate_command, check=True, cwd=ROOT)
            self.assertTrue((Path(temp_dir) / "evaluation_metrics.json").exists())


if __name__ == "__main__":
    unittest.main()
