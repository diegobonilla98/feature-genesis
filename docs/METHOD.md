# Method

## 1. Experimental object

Let a neural network with parameters \(\theta_t\) be trained by a deterministic update rule:

\[
\theta_{t+1} = U(\theta_t, o_t, B_t, \eta_t),
\]

where \(o_t\) is optimizer state, \(B_t\) is the exact training batch including deterministic augmentations, and \(\eta_t\) is the stateless learning rate. At selected steps \(t\), the activation at layer \(\ell\) is:

\[
h_t(x) = h_\ell(x; \theta_t) \in \mathbb{R}^{d \times H \times W}.
\]

The project studies how sparse features extracted from \(h_t\) evolve and which training evidence changes them.

## 2. Main benchmark

### 2.1 CUB-200-2011

The primary benchmark is CUB-200-2011. It contains 11,788 natural images, 200 bird species, 312 binary attributes, 15 part locations, and a bounding box for every image.

The official training split is partitioned deterministically into:

- training data,
- three validation images per class,
- the untouched official test split.

The validation split is selected by hashing class and image identifiers with a fixed split seed. This avoids dependence on mutable list order.

### 2.2 Synthetic shapes

The smoke benchmark renders circles, squares, and triangles with known color, size, horizontal position, and vertical position. Each factor has an exact binary annotation. Sample appearance noise is keyed by sample, split, epoch, and seed.

The synthetic benchmark serves three purposes:

1. execution verification,
2. exact-factor semantic tests,
3. fast negative controls before any CUB run.

## 3. Base model

The main model is a ResNet18 trained from scratch. Every BatchNorm layer is replaced by GroupNorm. The classifier has 200 outputs.

The default hook is `layer3`, whose output has 256 channels and a spatial grid of 10 x 10 for a 160 x 160 input.

### Why GroupNorm

For BatchNorm, the forward value for sample \(i\) depends on other samples in the batch through batch mean and variance. Removing one example therefore changes the activations and gradients of all remaining examples.

GroupNorm normalizes within each sample. With fixed batch order, masking the loss of sample \(i\) removes its direct gradient contribution without changing the forward path of sample \(j\). This makes example-level replay much easier to interpret.

## 4. Deterministic data path

### 4.1 Exact batch plan

Before training, every batch index and epoch number is generated and stored in `batch_plan.npz`. Replay never asks a random sampler to reconstruct the order.

### 4.2 Keyed augmentations

The transform seed for an image is:

\[
s_{i,e} = H(s_0, \text{split}, \text{image_id}_i, e),
\]

where \(H\) is a stable hash and \(e\) is the epoch stored in the batch plan.

Consequently, data-loader worker scheduling cannot change the crop, flip, or color jitter for a sample.

### 4.3 Bounding-box crop

Each object bounding box is expanded by a fixed margin, converted into a square contained in the image, and then sampled with deterministic scale and aspect-ratio noise during training. Evaluation uses the canonical square crop.

Mapped part annotations use the exact integer crop passed to torchvision. A part is marked invisible if the crop removes it.

## 5. Model training and anchors

The default optimizer is Nesterov SGD. The learning rate is a pure function of the global step:

\[
\eta_t = \operatorname{CosineWarmup}(t, T, \eta_{\max}, \eta_{\min}, T_w).
\]

Each anchor contains:

- model state,
- optimizer state,
- Python RNG state,
- NumPy RNG state,
- CPU torch RNG state,
- all CUDA RNG states,
- model SHA-256,
- run metadata and hashes.

The default CUB schedule records developmental checkpoints at:

\[
0, 25, 50, 100, 200, 400, 800, 1600, 3200, 4800, 6400, 7000.
\]

Frequent optimizer anchors are stored every 500 steps, and every declared developmental analysis step is also saved as an anchor. A requested intermediate state can still be reconstructed exactly from the nearest preceding anchor.

## 6. Activation dataset

At each checkpoint, the project evaluates a stable image subset with canonical transforms. Spatial tokens are stored in correspondence across checkpoints.

For full CUB defaults:

\[
N_{\text{rows}} = 3000 \times 10 \times 10 = 300{,}000.
\]

Each row stores:

- the activation vector,
- image ID,
- training-dataset index,
- class label,
- spatial grid coordinate,
- image attributes,
- mapped part locations,
- model prediction.

The random spatial-token option selects the same token coordinates at every checkpoint because selection is keyed by image ID rather than checkpoint.

## 7. Sparse autoencoders

For an activation \(x \in \mathbb{R}^d\), each SAE learns a nonnegative sparse code \(z \in \mathbb{R}^m_{\ge 0}\) and reconstruction:

\[
\hat{x} = s \left(Dz + b_D\right),
\]

where \(s\) is the stored dataset RMS scale and every decoder row is constrained to unit norm.

Initialization uses a robust center, dataset RMS scaling, principal directions, and additional orthogonal blocks for the overcomplete frame.

### 7.1 BatchTopK

The primary SAE selects the largest \(N k\) positive preactivations across a batch of \(N\) rows. This fixes mean training L0 at \(k\) while allowing different samples to use different numbers of features.

After training, a deterministic calibration sample is scored without BatchTopK competition. The inference threshold is set to the exact order statistic that yields the requested aggregate activation count on that sample. This gives sample-independent inference near the target mean L0 while preserving the exact global BatchTopK budget during training.

### 7.2 JumpReLU

JumpReLU applies a learned threshold per feature and optimizes an explicit L0 term through the paper's straight-through estimators. Initial thresholds are calibrated from an activation quantile so the run begins near the target mean L0.

### 7.3 Matryoshka

A single global BatchTopK code is computed over the full dictionary. Nested prefixes of that same code reconstruct the activation independently:

\[
\mathcal{L}_{\text{mat}} = \sum_{r=1}^{R} w_r
\left\|x - D_{1:m_r} z_{1:m_r}\right\|_2^2.
\]

This encourages early dictionary prefixes to retain broad features while later prefixes add specificity.

### 7.4 Cosine-scored BatchTopK

The cosine encoder replaces the standard inner-product score with:

\[
a_i(x) = \exp(b_i) \|x_c\|^{\alpha_i}
\cos(x_c, w_i) + b_{E,i},
\]

where \(x_c\) is centered input and \(\alpha_i\) is learned. It can interpolate between pure cosine scoring and inner-product-like norm dependence.

This method is experimental in this repository because it was introduced recently and its advantage is not universal across tasks or layers.

### 7.5 Matching Pursuit

The encoder repeatedly selects the positive decoder direction most aligned with the current residual. The residual is updated after each selected contribution. This tests conditionally orthogonal and hierarchical feature extraction.

It is treated as a diagnostic rather than automatically preferred. More expressive encoders can improve reconstruction while learning worse semantic units.

## 8. SAE training protocol

Every SAE run uses:

- deterministic initialization and activation-batch order,
- AdamW,
- cosine learning-rate decay,
- unit decoder normalization after every update,
- decoder-gradient projection onto the tangent space of the unit sphere,
- gradient clipping,
- dead-feature auxiliary reconstruction for TopK-derived models,
- stored hyperparameters inside every checkpoint.

Loading an SAE reconstructs its architecture from saved metadata, not from the caller's current configuration.

## 9. Comparing SAE families

A lower reconstruction error at higher density is not a meaningful win. The project builds frontiers over:

- realized mean L0,
- fraction of variance explained,
- normalized reconstruction error,
- dead-feature fraction,
- mutual dictionary coherence,
- model task loss recovered after activation substitution,
- semantic concept metrics,
- intervention selectivity.

The primary comparison rule is Pareto performance at matched realized L0 and comparable model fidelity.

## 10. Feature signatures

Each feature receives a multi-view signature:

\[
s_j(t) = \left[d_j(t), a_j(t), c_j(t), f_j(t), \mu_j(t)\right].
\]

The components are:

- normalized decoder direction \(d_j\),
- random-projection embedding of responses on fixed probe rows \(a_j\),
- standardized class and attribute effect profile \(c_j\),
- activation frequency \(f_j\),
- mean activation when active \(\mu_j\).

Top activating images are retained for qualitative inspection.

## 11. Activation-space alignment

Raw activation spaces can rotate during training. For corresponding rows \(X_t\) and \(X_{t'}\), the project fits an orthogonal Procrustes map:

\[
R^* = \arg\min_{R^T R = I}
\left\|(X_t - \bar{X}_t)R - (X_{t'} - \bar{X}_{t'})\right\|_F^2.
\]

The residual ratio is reported. A large residual is evidence that a pure rotation is inadequate and lineage confidence should be reduced.

## 12. Pairwise lineage

After alignment, source and target features are compared by a weighted combination of:

- decoder cosine similarity,
- activation-signature cosine similarity,
- concept-profile cosine similarity.

Hungarian matching gives one-to-one continuation candidates. Secondary high-score edges propose splits and merges. A feature is labeled a birth or death only if it has no continuation, split, or merge relation and is active above the configured frequency floor.

The graph is exported as JSON and GraphML.

## 13. Independent temporal checks

Warm-started SAEs make lineage easier to estimate but can impose artificial continuity. The paper protocol therefore compares:

1. continual SAEs,
2. independently initialized SAEs,
3. final SAE directions projected backward through aligned activation spaces,
4. supervised probes for known attributes,
5. causal model interventions.

A model-feature birth requires agreement across more than one measurement route.

## 14. Gradient screening

For a smooth feature-strength objective \(F_j(\theta)\), the local effect of one SGD update from batch \(B_t\) is approximated by:

\[
F_j(\theta_t - \eta_t \nabla_\theta L_{B_t}) - F_j(\theta_t)
\approx
-\eta_t \nabla_\theta F_j(\theta_t)^T \nabla_\theta L_{B_t}(\theta_t).
\]

The implementation evaluates each batch at its actual evolving baseline state, includes the step-specific learning rate, applies the original optimizer update, and continues to the next step. The feature gradient is refreshed every configured number of steps to control cost. This makes the ranking trajectory faithful to the original SGD path between refreshes, while the feature-gradient reuse remains a stated first-order approximation.

This score ranks candidate batches. It is only a screening method. The final claim comes from exact replay.

The feature objective uses mean positive preactivation over a fixed validation probe set. This gives a differentiable measure of precursor strength even before a BatchTopK feature crosses its inference threshold.

## 15. Exact counterfactual replay

For a candidate example set \(S\) and replay interval \([t_0,t_1)\), the baseline and intervention runs begin from the identical anchor at \(t_0\).

Example removal uses:

\[
L_t^{(-S)} = \frac{1}{|B_t|}
\sum_{i \in B_t} w_i \ell_i,
\qquad
w_i = \mathbf{1}[i \notin S].
\]

The denominator remains the original batch size. This prevents the remaining examples from being implicitly upweighted.

Available interventions are:

- zeroing selected example losses,
- replacing an image with a deterministic same-class example,
- occluding a selected annotated part while preserving the label.

The causal feature effect is:

\[
\Delta_j(S; t_0,t_1) =
F_j(\theta_{t_1}^{(-S)}) - F_j(\theta_{t_1}).
\]

Parameter L2 distance, maximum parameter difference, and both model hashes are stored.

## 16. Semantic and behavioral validation

The repository evaluates:

- best-feature average precision for each attribute,
- held-out OR coalitions of multiple features for each attribute,
- spatial distance from peak feature location to annotated parts,
- reconstruction substitution into the original model,
- zero-ablation baseline,
- single-feature ablation,
- single-feature steering,
- class-probability shifts and prediction flips.

Natural-language feature labels remain metadata. They do not enter the causal proof.
