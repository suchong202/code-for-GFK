# code-for-GFK: public workflow preview

This repository is an intentionally limited, runnable preview of a
MALDI-TOF mass-spectrum classification workflow.

It is designed to demonstrate the experiment plumbing used for paper review:

1. load a spectrum dataset or generate deterministic synthetic demo data;
2. call a stable peak-selector input/output interface;
3. apply a public placeholder augmentation;
4. train a lightweight demonstration classifier;
5. report validation AUC, macro-F1, and accuracy; and
6. save and reload a demo checkpoint.

## Important scope statement

This preview **does not reproduce the paper's reported results**. The following
research assets are intentionally withheld while the manuscript is under
review:

- the GNN peak-selection implementation, graph construction, message passing,
  scoring rule, and diversity logic;
- the complete augmentation policy and its composition;
- paper-specific hyperparameters and selection criteria;
- trained research-model weights;
- raw, derived, or patient-level research data; and
- the full research model and training recipe.

The public selector is a clearly marked intensity-ranking placeholder. It
implements the same high-level input/output contract so that the rest of the
demo can run, but it is not the proposed GNN method.

See [PUBLIC_RELEASE_SCOPE.md](PUBLIC_RELEASE_SCOPE.md) for the disclosure
matrix and pseudocode boundary.

## Quick start

Python 3.10 or newer is sufficient. The demo has no third-party runtime
dependencies.

```bash
python train.py --config configs/demo.yaml --seed 42
python evaluate.py --config configs/demo.yaml --seed 42
```

The first command creates `artifacts/demo_checkpoint.json`. The second command
reloads it and evaluates the same deterministic validation split. Generated
artifacts are ignored by Git.

Run the smoke tests with:

```bash
python -m unittest discover -s tests -v
```

## Optional dataset interface

For interface testing with your own non-sensitive data:

```bash
python train.py --config configs/demo.yaml --seed 42 --data path/to/data.csv
```

The CSV must contain a `label` column and numeric spectrum columns. An optional
`sample_id` column is ignored. Data supplied by users is never included in this
repository.

## Peak-selector contract

```python
from gfk_public.peak_selector import GNNPeakSelector

selector = GNNPeakSelector(selected_count=12)
selection = selector.select(spectrum, mz_axis)

# selection.indices: selected bin indices, sorted by m/z position
# selection.tokens:  [normalized_mz, normalized_intensity] pairs
# selection.metadata: identifies the public implementation as a placeholder
```

The number above is a demo-only value and is not a paper hyperparameter.

## Research-use notice

This repository is a workflow preview for inspection and demonstration. No
claim is made that the placeholder modules reproduce the protected method or
the paper's numerical results. A fuller release may be considered after the
manuscript decision, subject to data-use, institutional, and intellectual
property constraints.

No license is granted by publication of this repository; see [NOTICE.md](NOTICE.md).
