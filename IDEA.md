# Feature Genesis: Research Thesis

## One-sentence thesis

A learned feature should be studied as a developmental causal object, not merely as a direction discovered after training.

## Core question

Given a final interpretable feature, can we reconstruct:

1. when a measurable precursor first appeared,
2. how its selectivity and causal role changed,
3. whether it split, merged, or was repurposed,
4. which training examples and regions contributed to those transitions,
5. whether the same developmental story repeats across seeds?

## Proposed contribution

Feature Genesis combines four ideas that are normally studied separately:

1. exact deterministic optimization replay,
2. sparse feature discovery at many checkpoints,
3. temporal feature genealogy under non-identifiable dictionaries,
4. causal training-data intervention at the level of a selected feature.

The intended output for feature \(j\) is a causal genealogy:

\[
G_j = \left(V_j, E_j, A_j, C_j\right),
\]

where:

- \(V_j\) contains feature states at training checkpoints,
- \(E_j\) contains continue, split, merge, birth, and death hypotheses,
- \(A_j\) contains semantic, spatial, activation, and behavioral evidence,
- \(C_j\) contains exact counterfactual replay effects.

## Why the project can produce a strong paper

Checkpoint visualization by itself is descriptive. SAE labeling by itself is descriptive. Gradient attribution by itself is approximate. The paper becomes substantially stronger when these are connected by exact local retraining interventions.

A compelling result would have the form:

> A final visual feature begins as a broad texture precursor, becomes spatially tied to a bird part, gains species selectivity after a small set of training examples, and remains delayed or weakened when those examples are removed in exact replay. The effect generalizes across seeds and cannot be explained by an independently trained SAE artifact.

## Primary hypotheses

### H1: Feature development is structured

Semantic features will show reproducible phases such as emergence, specialization, stabilization, splitting, and consolidation rather than independent random drift.

### H2: Semantic identity and causal importance emerge at different times

A feature may become linearly or sparsely detectable before its ablation affects task behavior, or it may become causally important before it receives a clean semantic label.

### H3: A small amount of training evidence can have disproportionate developmental influence

Within a narrow replay interval, a small cluster of examples or factor edits will produce a larger feature-state change than matched random interventions.

### H4: Developmental histories are more stable than individual SAE indices

Exact latent indices will vary across SAE seeds, but aligned semantic and causal trajectories will show higher cross-seed agreement.

### H5: Some apparent feature births are measurement births

A feature may pre-exist in the model but only become separable by the SAE later. Agreement with probes, backward projection, causal intervention, and independent SAE runs is necessary to classify a model-feature birth.

## Dataset choice

CUB-200-2011 is the main benchmark because it combines manageable scale with unusually rich supervision:

- 200 fine-grained classes,
- 312 attributes,
- 15 part locations,
- bounding boxes,
- natural variation in pose, texture, color, shape, and background.

A scratch model avoids ImageNet overlap. The synthetic shapes benchmark supplies exact factor ground truth and fast falsification tests.

## Architecture choice

A GroupNorm ResNet18 is deliberately less fashionable than a giant pretrained foundation model. That is a strength for this question:

- the entire learning history is controlled,
- training is cheap enough to repeat,
- activations remain spatially localizable,
- optimizer replay is tractable,
- GroupNorm avoids cross-sample batch-statistic coupling,
- the model is still expressive enough to develop nontrivial visual abstractions.

## SAE strategy

No single SAE family is accepted as the truth.

- BatchTopK is the primary dictionary because it controls mean sparsity directly.
- JumpReLU is the strong L0-trained reconstruction baseline.
- Matryoshka tests hierarchical abstraction and feature absorption.
- Cosine-scored BatchTopK tests whether activation norm corrupts feature selection.
- Matching Pursuit tests conditionally orthogonal and hierarchical encoding, but is treated as diagnostic because expressive encoders may exploit reconstruction noise.

Methods are compared at matched realized L0 and matched model-level fidelity, not only reconstruction MSE.

## Main novelty boundary

The project does not claim that feature tracking across checkpoints is new by itself. The novelty target is the integrated causal method:

> temporally aligned sparse feature genealogy plus exact example-level and region-level optimization replay.

## What would falsify the central story

The project should report a negative result if:

- lineage events change arbitrarily across SAE seeds,
- final-feature backward projection disagrees with checkpoint SAEs,
- candidate influential batches do not outperform random matched batches under exact replay,
- semantic perturbations do not selectively move the claimed feature,
- feature ablation has no reproducible downstream effect,
- removing the alleged source examples changes many unrelated features equally,
- the developmental pattern disappears across model seeds.

A clean negative result would still be valuable because it would place a sharp limit on developmental claims made from post-hoc SAE analysis.
