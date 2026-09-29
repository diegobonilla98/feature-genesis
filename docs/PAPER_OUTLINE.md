# Suggested Paper Outline

## Working title

**Feature Genesis: Causal Reconstruction of Learned Feature Development Through Deterministic Training Replay**

## Abstract structure

1. Post-hoc interpretability identifies final features but usually cannot explain how they formed.
2. Introduce deterministic feature genealogy across training checkpoints.
3. Combine temporal SAE alignment with exact training-example and region interventions.
4. Validate on synthetic factors and CUB-200-2011.
5. Report aggregate developmental patterns and several causal case studies.
6. State limitations from SAE non-identifiability and environment-specific exact replay.

## 1. Introduction

Motivation:

- final-model interpretation omits learning history,
- training-data attribution usually targets predictions rather than internal features,
- checkpoint feature tracking is mostly observational,
- exact local replay can connect examples to feature formation.

Contributions:

1. an exact deterministic replay system for feature-level interventions,
2. a multi-view temporal SAE genealogy method,
3. human-grounded vision evaluation using attributes and parts,
4. causal case studies and aggregate developmental findings,
5. an open proof-oriented benchmark and codebase.

## 2. Related work

Organize by problem rather than chronology:

- sparse autoencoders and direct L0 methods,
- hierarchical and conditional feature extraction,
- SAE evaluation and non-identifiability,
- temporal representation analysis,
- training-data attribution and influence methods,
- deterministic machine learning systems.

## 3. Problem formulation

Define:

- model trajectory \(\theta_0,\ldots,\theta_T\),
- layer activations \(h_t(x)\),
- checkpoint SAE dictionaries,
- feature state signatures,
- lineage graph,
- exact replay intervention effect.

Clearly distinguish:

- SAE feature birth,
- measurable representation birth,
- causal feature formation.

## 4. Method

### 4.1 Deterministic training and replay

Batch plans, keyed transforms, GroupNorm, optimizer anchors, environment contract.

### 4.2 Sparse feature extraction

Primary BatchTopK and matched baselines.

### 4.3 Temporal alignment and genealogy

Procrustes, multi-view signatures, continuation, split, merge, birth, death.

### 4.4 Candidate data attribution

Gradient-alignment ranking.

### 4.5 Exact causal replay

Fixed-denominator removal, same-class replacement, part occlusion.

### 4.6 Validation

Attributes, coalitions, parts, model substitution, ablation, steering, independent SAE seeds.

## 5. Experimental setup

### 5.1 Synthetic shapes

Known factors and exact controls.

### 5.2 CUB-200-2011

Scratch training, split, architecture, checkpoint schedule, layer choice.

### 5.3 SAE frontier

Matched L0 and task fidelity.

### 5.4 Statistical protocol

Base-model seeds as primary independent units.

## 6. Results

### 6.1 Exact replay validation

Report bitwise anchor recovery across several intervals and hardware configurations separately.

### 6.2 SAE benchmark

Show Pareto frontiers for FVE, realized L0, task loss recovered, and semantic metrics.

### 6.3 Aggregate developmental dynamics

Feature emergence-time distribution, specialization rates, stability, split and merge frequencies.

### 6.4 Semantic versus causal timing

Show lag between concept detectability and ablation relevance.

### 6.5 Training-example influence

Compare top candidates with matched random controls under exact replay.

### 6.6 Region-level causal evidence

Show whether annotated parts explain full-example effects.

### 6.7 Cross-seed replication

Compare semantic families and influential factors rather than requiring identical latent IDs.

## 7. Causal feature genealogy case studies

Use a consistent figure template:

1. final feature examples and label,
2. backward trajectory plot,
3. lineage graph,
4. top candidate batches,
5. exact replay effect plot,
6. part or factor perturbation,
7. model behavior effect,
8. seed replication.

Avoid selecting cases only for visual appeal. State the selection funnel.

## 8. Ablations

- BatchNorm versus GroupNorm,
- continual versus independent SAEs,
- decoder-only versus multi-view matching,
- aligned versus unaligned trajectories,
- full spatial grid versus sampled tokens,
- fixed denominator versus renormalized removal,
- interval length,
- SAE family,
- matched L0 versus unmatched comparisons.

## 9. Limitations

Include:

- non-canonical SAE units,
- one layer and architecture in the first paper,
- CUB label correlations,
- off-manifold activation interventions,
- local scope of exact replay effects,
- hardware-specific bitwise claims,
- possible multidimensional features.

## 10. Discussion

Discuss what a developmental explanation should require.

A useful hierarchy is:

1. observational precursor,
2. semantic specialization,
3. behavioral relevance,
4. training-evidence causation,
5. cross-seed reproducibility.

## 11. Conclusion

The main conclusion should concern method and evidence, not claim that all neural features have simple genealogies.

## Main figures

### Figure 1

Method overview from training trajectory to causal genealogy.

### Figure 2

Exact replay contract and bitwise verification.

### Figure 3

SAE Pareto frontier at matched L0 and model fidelity.

### Figure 4

Aggregate feature developmental phases.

### Figure 5

Semantic detectability versus causal importance timing.

### Figure 6

Example-level exact replay effects against matched controls.

### Figure 7

One complete region-grounded feature genealogy.

### Figure 8

Cross-seed replication and failure cases.

## Tables

### Table 1

Dataset, model, layer, and compute settings.

### Table 2

SAE family metrics at matched realized L0.

### Table 3

Lineage stability across SAE and model seeds.

### Table 4

Exact replay candidate versus control effects.

### Table 5

Negative results and unresolved genealogies.
