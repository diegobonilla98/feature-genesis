# Feature Genesis v3: QuickDraw visual microscope

The CUB run passed reconstruction, sparsity, replay, and cross-seed gates but failed the visual-semantic gate. Its accepted labels were too dependent on color, species-correlated attributes, coarse response maps, and crowded contact sheets. That run is retired rather than treated as successful scientific evidence.

## Confirmed defects in v2

1. The numerical feature score used `sae.encode(..., inference=True)`, but the response visualization used `relu(sae.preactivations(...))`. BatchTopK suppressed many of those positive preactivations. The displayed response could therefore highlight a location where the sparse feature was actually zero.
2. Every response overlay was normalized to its own maximum. A tiny residual response could look as visually strong as a true top activation.
3. Feature discovery used top examples, minimum nonzero examples, and median active examples. It did not sample uniformly across activation intervals, so polysemantic behavior at weaker activation levels was easy to miss.
4. Gemini received a dense contact sheet as one image. Small response regions were difficult to see, image identities were encoded in rendered text, and the model could not inspect each piece of evidence at useful resolution.
5. Candidate ranking rewarded CUB attribute average precision. Since CUB attributes are correlated with species, this favored human-nameable class shortcuts rather than an unbiased sample of the SAE dictionary.
6. The activation audit used only 3,000 CUB images for 4,096 SAE features. The resulting evidence tail was shallow and visually repetitive.
7. Gradient screening ranked whole batches. All images in a candidate batch inherited the same influence score, so the pipeline could not support an image-level claim about which example built or broke a feature.
8. Backward feature trajectories also used positive preactivation instead of the actual sparse code.
9. Held-out Gemini validation showed activation-derived response overlays. The new validator sees original images only and must predict activation from the proposed explanation without target leakage.

## v3 experiment

- Dataset: local QuickDraw parquet corpus at `F:/QuickDraw`.
- Training images: 4,500,000.
- Validation images: 250,000.
- Test images: 250,000.
- Classes: 345.
- Input: 64 × 64 grayscale drawings.
- Model: GroupNorm QuickDraw CNN trained from scratch.
- Hook: `block2`, a 128-channel 32 × 32 spatial representation.
- Primary SAE: 2,048 BatchTopK features with `k=8`.
- Analysis checkpoints: 0, 25, 50, 100, 200, 400, 800, 1,600, 3,200, 6,400, 10,000, 15,000, and 20,000 steps.
- Run root: `F:/QuickDraw/feature_genesis_runs/quickdraw_cnn_seed1729_v3`.

The SAE-training cache uses a fixed train subset with 96 spatial tokens per image. The final semantic-evidence cache is a separate validation subset with every 32 × 32 location preserved. Gemini never labels from the SAE-training examples.

## Feature discovery

Each feature is screened for:

- sufficient active and inactive images;
- activation frequency;
- dynamic range;
- activation-mass class entropy;
- effective number of classes;
- modal-class share among the top 32 images;
- decoder-direction redundancy;
- repeatability across SAE seeds.

Candidates with fewer than three effective classes or more than 50% top-tail modal-class share are excluded before Gemini. Ranking no longer uses ground-truth class names or attribute AP as a positive signal.

Following Anthropic’s feature-analysis methodology, discovery examples cover evenly spaced activation intervals rather than only the maximum tail. Strong examples are class-diversified, and no class contributes more than the configured number of examples unless the candidate pool is exhausted.

References:

- [Towards Monosemanticity](https://transformer-circuits.pub/2023/monosemantic-features/index.html)
- [Scaling Monosemanticity](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html)
- [Anthropic feature browser](https://transformer-circuits.pub/2024/scaling-monosemanticity/features/index.html)

## Visual evidence

Discovery evidence is no longer a contact sheet or independently normalized heatmap.

For every positive example, the pipeline writes:

1. a 256 × 256 drawing with a visible magenta box around the strongest connected component of the actual thresholded sparse response;
2. a separate 256 × 256 nearest-neighbor zoom of exactly that region.

Matched inactive controls are separate original images. Each file is sent to Gemini as a separate multimodal part preceded by its exact evidence filename. Validation images are separate originals with randomized tile IDs and contain no box, response, activation, or target.

Gemini returns:

- a short canonical label;
- a plain-English explanation;
- atomic visual primitives;
- trigger and non-trigger conditions;
- activation-spectrum behavior;
- class-confounding assessment;
- counterevidence;
- edit tests and falsifiers.

A label is rejected when quantitative class confounding is too high, Gemini judges confounding high, held-out activation prediction is weak, or the explanation is ambiguous.

## Training-image causality

Batch influence remains a first-order screen:

\[
\Delta F \approx -\eta \nabla_\theta F^\top \nabla_\theta \ell_B.
\]

Within every batch, v3 estimates each example’s contribution using a centered directional derivative of its individual loss along the normalized feature-objective gradient. This requires two functional forwards and does not mutate the replayed model state:

\[
\nabla_\theta \ell_i^\top \hat g_F
\approx
\frac{\ell_i(\theta+\epsilon\hat g_F)-\ell_i(\theta-\epsilon\hat g_F)}
{2\epsilon}.
\]

The exact counterfactual replay then masks only the selected individual drawings rather than the entire batch. Every rendered builder or breaker records its training step, dataset index, augmentation epoch, class, batch influence, and per-example influence.

The lineage graph maps a final feature to its earliest matched checkpoint feature. Attribution defaults to the interval immediately preceding that mapped birth instead of always analyzing only the final training interval.

## Execution

All control values remain editable constants at the top of each script.

```powershell
& "C:\Users\diego\anaconda3\envs\python311\python.exe" "scripts\00_validate_quickdraw.py"
& "C:\Users\diego\anaconda3\envs\python311\python.exe" "scripts\01_train_model.py"
& "C:\Users\diego\anaconda3\envs\python311\python.exe" "scripts\02_verify_replay.py"
& "C:\Users\diego\anaconda3\envs\python311\python.exe" "scripts\03_cache_activations.py"
& "C:\Users\diego\anaconda3\envs\python311\python.exe" "scripts\04_train_saes.py"
& "C:\Users\diego\anaconda3\envs\python311\python.exe" "scripts\04_validate_saes.py"
& "C:\Users\diego\anaconda3\envs\python311\python.exe" "scripts\05_compute_signatures.py"
& "C:\Users\diego\anaconda3\envs\python311\python.exe" "scripts\06_build_lineage.py"
& "C:\Users\diego\anaconda3\envs\python311\python.exe" "scripts\07_score_concepts.py"
& "C:\Users\diego\anaconda3\envs\python311\python.exe" "scripts\15_evaluate_features_gemini.py"
```

For an accepted feature, set `FEATURE_ID` in scripts 08, 09, 13, and 17:

```powershell
& "C:\Users\diego\anaconda3\envs\python311\python.exe" "scripts\13_backward_feature_trajectories.py"
& "C:\Users\diego\anaconda3\envs\python311\python.exe" "scripts\08_rank_training_batches.py"
& "C:\Users\diego\anaconda3\envs\python311\python.exe" "scripts\17_render_training_influence.py"
& "C:\Users\diego\anaconda3\envs\python311\python.exe" "scripts\09_counterfactual_replay.py"
```

The gradient screen is not causal proof. Only the exact replay result may be described causally.
