# Failure Modes and Threats to Validity

## 1. SAE feature birth is not model feature birth

A representation may exist before the SAE separates it. Conversely, an SAE can create an apparently clean latent from a continuously changing mixture.

Mitigation:

- independent SAE runs,
- final-feature backward projection,
- supervised probes,
- model ablation,
- semantic input interventions.

## 2. Sparse dictionaries are non-canonical

Feature indices can permute, split, merge, or disappear across initialization. Similar decoder directions can still have different activation conditions.

Mitigation:

- multi-view signatures,
- Hungarian matching,
- split and merge edges,
- SAE seed replication,
- report unresolved identity.

## 3. Warm starts can manufacture continuity

Initializing checkpoint \(t+1\)'s SAE from checkpoint \(t\) can preserve an old feature even if an independent optimum would reorganize the dictionary.

Mitigation:

- compare continual and independent SAEs,
- use temporal regularization only as an explicit ablation,
- inspect backward projection from the final SAE.

## 4. Alignment can hide nonlinear representational change

Orthogonal Procrustes assumes corresponding activation spaces differ mainly by rotation and translation. A high residual means this assumption is poor.

Mitigation:

- report residual ratio,
- test linear maps with held-out rows,
- compare unaligned concept signatures,
- reduce confidence when alignment is unstable,
- investigate subspace or nonlinear alignment only as a separate analysis.

## 5. L0 mismatch invalidates SAE comparisons

A method can reconstruct better simply because it activates more features. JumpReLU density can drift during training, and BatchTopK training L0 differs from thresholded inference L0.

Mitigation:

- report realized inference L0,
- build full sparsity frontiers,
- initialize JumpReLU near the target L0,
- compare at matched model fidelity as well as matched L0.

## 6. Reconstruction is not interpretability

A more expressive encoder can exploit residual noise or arbitrary basis refinements. Matching Pursuit is a specific risk because iterative residual fitting can improve reconstruction without recovering semantic ground truth.

Mitigation:

- synthetic factors,
- held-out concept coalitions,
- targeted perturbations,
- SAE seed stability,
- model-level interventions,
- do not rank methods by MSE alone.

## 7. Attribute labels are incomplete and correlated

CUB attributes are not a complete ontology. Species, colors, parts, and backgrounds are correlated. A feature may appear to represent one attribute while using another cue.

Mitigation:

- many-to-one feature coalitions,
- part localization,
- within-class and same-class controls,
- paired synthetic factor edits,
- background perturbations,
- conditional analyses.

## 8. Bounding-box crops can alter the learning problem

Object-focused crops reduce background shortcuts but also remove natural context and can make part locations easier to interpret than in an unconstrained classifier.

Mitigation:

- state this scope explicitly,
- run a full-image ablation on one seed,
- compare feature classes and causal examples between crop regimes.

## 9. Gradient alignment is local

The dot product between feature and batch gradients ignores curvature, momentum, later interactions, and feature reparameterization.

Mitigation:

- use it only to rank candidates,
- validate every main claim with exact replay,
- include gradient-norm matched controls.

## 10. Long replay intervals become chaotic

Removing one example can cause trajectories to diverge in many unrelated ways over thousands of steps. A final difference may be causal but mechanistically diffuse.

Mitigation:

- begin with narrow intervals near detected events,
- compare several interval lengths,
- measure target-feature specificity,
- report global parameter and task changes,
- avoid claiming a unique pathway from a very long replay.

## 11. Removing an example changes optimization magnitude

Even with a fixed denominator, the total gradient norm decreases when an example is masked. That is the intended direct effect but can still alter later momentum dynamics.

Mitigation:

- compare same-gradient-norm controls,
- optionally add a norm-matched replacement intervention,
- distinguish example-content effects from generic update-magnitude effects.

## 12. Same-class replacement is not neutral

A replacement preserves the class label but can change pose, background, attributes, and difficulty.

Mitigation:

- choose replacements matched on attributes or embeddings,
- report replacement metadata,
- use multiple replacements per source image.

## 13. Part occlusion can create artifacts

A constant square can be out of distribution and itself activate features.

Mitigation:

- compare multiple fill strategies,
- use local blur or inpainting as an ablation,
- include matched random locations,
- test direct response before training replay.

## 14. Feature ablation can leave the data manifold

Subtracting one SAE contribution may produce an activation that the downstream network never encountered.

Mitigation:

- report intervention magnitude,
- test partial ablations,
- compare steering in both directions,
- evaluate whether reconstruction residual grows abnormally,
- avoid equating a large off-manifold effect with normal feature use.

## 15. A feature may be a subspace or manifold

A one-dimensional latent may split a genuinely multidimensional concept, or several latents may jointly represent one factor.

Mitigation:

- coalition metrics,
- local subspace analysis,
- Matryoshka and Matching Pursuit diagnostics,
- descendant partition tests for apparent splits.

## 16. Final-model labels can bias historical interpretation

Using the final semantic label to inspect early checkpoints can make ambiguous precursors look more coherent than they were.

Mitigation:

- blind early-checkpoint labeling,
- preregister trajectory measures,
- show top examples without final labels,
- compare forward and backward analyses.

## 17. Feature selection can create a winner's curse

Choosing the feature with the prettiest lineage, strongest AP, and largest causal effect from thousands of candidates inflates the result.

Mitigation:

- separate discovery and confirmation seeds,
- freeze selection rules,
- test held-out checkpoints or model seeds,
- report the full candidate funnel.

## 18. CUB can overlap with ImageNet

Using an ImageNet-pretrained model would make feature-origin claims invalid because part of the feature history occurred before the recorded run.

Mitigation:

- train from scratch,
- do not use pretrained augmenters that contain trainable visual representations in the main causal path,
- state any auxiliary pretrained model used only for analysis.

## 19. Exact replay is environment-specific

Bitwise equality can fail after a PyTorch, CUDA, cuDNN, compiler, driver, or GPU change even with identical seeds and code.

Mitigation:

- store the full environment,
- reject critical mismatches by default,
- distinguish exact replay from cross-machine replication.

## 20. The model may be undertrained

A small scratch model on CUB may have lower accuracy than a modern pretrained system. Low task performance can produce unstable or shortcut-heavy features.

Mitigation:

- report learning curves and final accuracy,
- analyze feature stability as a function of training quality,
- improve the schedule before scaling the architecture,
- keep pretraining out of the main origin claim.
