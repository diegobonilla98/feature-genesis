# Paper-Grade Experiment Plan

## 1. Experimental phases

### Phase A: Proof and calibration

Run the synthetic shapes benchmark until all deterministic, sparse-coding, lineage, and causal controls pass.

Required outputs:

- exact replay hash equality,
- known-factor concept recovery,
- known-factor spatial localization,
- feature-coalition recovery for disjoint subtypes,
- null interventions with zero parameter difference,
- deliberately causal interventions with nonzero feature effects.

### Phase B: Main CUB model

Train three scratch GroupNorm ResNet18 seeds with the same data split and schedule. Do not select a seed by final accuracy. Analyze all seeds.

### Phase C: SAE frontier

At the final checkpoint, train each SAE family over matched sparsity settings:

- BatchTopK with L0 in \(\{8,16,32\}\),
- cosine BatchTopK with L0 in \(\{8,16,32\}\),
- Matryoshka with L0 in \(\{8,16,32\}\),
- JumpReLU over a calibrated sparsity-coefficient grid,
- Matching Pursuit with steps in \(\{8,16,32\}\).

Use three SAE seeds per point.

Select a primary operating point from a Pareto frontier over:

- realized L0,
- FVE,
- task loss recovered,
- held-out coalition score,
- perturbation alignment or synthetic factor selectivity,
- dead-feature fraction.

Do not select solely by reconstruction.

### Phase D: Developmental dictionaries

Train the selected primary SAE at all model checkpoints in two modes:

1. continual warm start,
2. independent initialization.

Compute signatures and pairwise lineages for both.

### Phase E: Feature selection

Select candidate final features using a preregistered score combining:

- sufficient activation frequency,
- high attribute or class selectivity,
- strong part localization,
- strong model intervention effect,
- stable identity across SAE seeds,
- a nontrivial developmental event.

Do not manually select only visually attractive examples.

### Phase F: Causal genealogy

For each selected feature:

1. project the final feature backward through aligned checkpoints,
2. identify the first stable precursor window,
3. rank training batches in a narrow interval by gradient alignment,
4. choose top candidates before exact replay,
5. replay candidate removals and matched controls,
6. refine from batch to sample and region interventions,
7. repeat across base-model seeds.

## 2. Main experiments

## E1. Do features have reproducible developmental phases?

For each feature lineage, estimate trajectories of:

- positive-response frequency,
- mean positive response,
- maximum response,
- class and attribute selectivity,
- part localization,
- task ablation effect,
- decoder and activation drift.

Fit change points without looking at candidate training examples. Compare phase timing across seeds.

Primary outcome: cross-seed agreement in normalized emergence and specialization times.

## E2. Does semantic detectability precede causal importance?

At every checkpoint, measure:

- supervised attribute probe score,
- SAE concept score,
- final-feature backward response,
- model loss change after feature ablation.

Primary outcome: distribution of lag between semantic detectability and behavioral relevance.

## E3. Which training examples contribute to feature formation?

Within a preregistered interval, compare exact replay effects for:

- top gradient-alignment samples,
- semantically similar but low-gradient samples,
- same-class random samples,
- random samples matched by loss and gradient norm,
- top samples for an unrelated feature.

Primary outcome:

\[
\Delta_j^{\text{top}} - \Delta_j^{\text{matched random}}.
\]

Secondary outcomes include specificity to feature \(j\), parameter distance, and task accuracy change.

## E4. Which image region causes the contribution?

For influential CUB images, replay:

- removal of the full example,
- occlusion of the nearest annotated part,
- occlusion of a matched random region,
- same-class replacement,
- background-only perturbation where possible.

Primary outcome: fraction of the full-example effect explained by the hypothesized part.

## E5. Birth, split, and merge validation

For every proposed lineage event, compare:

- continual SAE event,
- independent SAE event,
- backward projection trajectory,
- concept-profile change,
- feature ablation change.

A split is supported only if the descendants partition activation contexts or interventions in a way the precursor did not.

A merge is supported only if the target responds to the union of predecessor contexts and the relation survives independent SAE seeds.

## E6. Are SAE births model births?

Define four event labels:

- confirmed model birth,
- likely model birth,
- measurement birth,
- unresolved.

A confirmed model birth requires:

- no earlier supervised or backward-projected signal,
- a later signal across independent SAEs,
- a later causal intervention effect,
- robustness to alignment method.

## E7. Does the SAE family change the developmental story?

Repeat selected lineages with BatchTopK, JumpReLU, Matryoshka, and cosine BatchTopK at matched realized L0 and model fidelity.

Primary outcome: agreement of semantic trajectories after feature-set matching.

Matching Pursuit is reported separately because its richer encoder can exploit residual structure that may not correspond to ground-truth features.

## E8. Does the same evidence create the same feature across seeds?

Cluster influential examples by attributes, parts, and visual embeddings. Compare whether corresponding feature families across model seeds are driven by similar clusters even when exact samples differ.

Primary outcome: cross-seed overlap in influential semantic factors, not exact image IDs.

## 3. Negative controls

Every causal result should include:

- absent-index no-op,
- matched random example removal,
- matched same-class removal,
- unrelated-feature candidate removal,
- random region occlusion,
- shuffled attribute labels,
- random decoder directions,
- untrained SAE,
- independently trained SAE,
- replay interval with no proposed event,
- equivalent parameter-distance control where feasible.

## 4. SAE sanity controls

Report:

- L0 distribution, not only mean,
- FVE and normalized MSE,
- task loss recovered,
- dead-feature fraction,
- activation frequency histogram,
- mutual coherence,
- feature duplication rates,
- stability across SAE seeds,
- semantic precision and recall tradeoff,
- behavior under shuffled concepts,
- synthetic factor recovery.

## 5. Statistical analysis

The independent unit for model-development claims is the base-model seed, not the number of images or SAE latents.

Recommended reporting:

- per-seed values,
- mean and standard deviation,
- paired effects within each model seed,
- hierarchical bootstrap over model seeds, features, and intervention examples,
- randomization tests for candidate versus matched-control rankings,
- false-discovery correction across large feature panels.

Avoid treating thousands of SAE features from one model as thousands of independent scientific replications.

## 6. Preregistered success criteria

A candidate feature genealogy is headline-ready only if:

1. final feature identity is stable across at least two SAE seeds,
2. model substitution retains meaningful task information,
3. semantic evidence exceeds shuffled-label controls,
4. spatial evidence exceeds matched random-location controls,
5. the developmental event is visible through at least two measurement routes,
6. exact replay candidate effects exceed matched random effects,
7. the effect is selective for the target feature or feature family,
8. the direction of effect repeats in at least two base-model seeds.

## 7. Recommended first paper scope

Keep the first paper narrow:

- one model architecture,
- one main layer,
- three base-model seeds,
- BatchTopK as primary SAE,
- two serious SAE baselines,
- five to twenty deeply validated feature genealogies,
- broad aggregate statistics over all stable features,
- exact replay on a limited number of preregistered intervals.

A smaller number of rigorous causal genealogies is more convincing than thousands of auto-labeled feature movies.
