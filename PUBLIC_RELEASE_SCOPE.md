# Public release scope

This file makes the boundary between runnable public code and withheld research
code explicit.

| Component | Public preview | Withheld research implementation |
| --- | --- | --- |
| Data path | Deterministic synthetic generator and generic CSV loader | Research cohorts, raw/derived spectra, labels, sample IDs, and split records |
| Peak selection | Stable input/output interface plus intensity-ranking fallback | Candidate construction, graph topology, edge functions, message passing, node scoring, selection constraints, and tuned values |
| Augmentation | Shape-preserving low-amplitude demo jitter | Full transformation set, ordering, probabilities, schedules, and tuned values |
| Model | Small dependency-free logistic demonstration model | Full research encoder, heads, objectives, calibration, and ensemble logic |
| Training | Deterministic validation-only demonstration loop | Complete optimization recipe, loss weights, training schedules, and paper configuration |
| Weights | Demo checkpoint generated locally | All trained research checkpoints |
| Evaluation | Validation AUC, macro-F1, and accuracy | Paper result files and any test-set outputs |

## Protected peak-selector pseudocode

Only the following non-operational outline is disclosed:

```text
INPUT: one binned spectrum and its m/z axis
candidate_nodes <- PRIVATE_CANDIDATE_STAGE(spectrum)
graph           <- PRIVATE_GRAPH_STAGE(candidate_nodes)
node_states     <- PRIVATE_MESSAGE_PASSING(graph, candidate_nodes)
scores          <- PRIVATE_SCORING_STAGE(node_states)
indices         <- PRIVATE_SELECTION_STAGE(scores)
OUTPUT: sorted indices and fixed-shape peak tokens
```

The public `GNNPeakSelector` class does not implement any `PRIVATE_*` stage.
It uses a conventional intensity ranking solely to keep the public pipeline
executable.

## Protected augmentation pseudocode

```text
INPUT: tokenized spectrum
view_a <- PRIVATE_COMPOSED_POLICY(tokens, random_state_a)
view_b <- PRIVATE_COMPOSED_POLICY(tokens, random_state_b)
OUTPUT: two shape-compatible stochastic views
```

The public preview applies only a small, generic intensity jitter and must not
be interpreted as the paper's augmentation method.

## Reproducibility claim

The repository reproduces the software flow and public API behavior of the
preview itself. It intentionally does not claim numerical reproduction of the
submitted manuscript.
