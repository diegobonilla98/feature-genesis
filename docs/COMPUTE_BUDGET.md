# Compute and Storage Budget

## Main configuration

The default CUB run uses:

- 5,394 training images after taking three validation images per class from the official training split,
- batch size 64,
- 7,000 optimization steps,
- 160 x 160 inputs,
- scratch GroupNorm ResNet18,
- layer3 activations with 256 channels and a 10 x 10 grid,
- 3,000 cached analysis images per checkpoint,
- 12 analysis checkpoints,
- a 4,096-feature SAE.

The exact training-set count should be read from run metadata because it depends on the configured validation split.

## Base-model image presentations

At batch size 64 and 7,000 steps:

\[
64 \times 7000 = 448{,}000
\]

image presentations are processed.

This is many orders of magnitude below a standard ImageNet-scale pretraining run. The project is designed for repeated research experiments rather than one expensive model.

## Activation-cache size

With all spatial tokens:

\[
3000 \text{ images} \times 100 \text{ tokens/image}
= 300{,}000 \text{ rows/checkpoint}.
\]

At 256 dimensions and float16:

\[
300{,}000 \times 256 \times 2
= 153{,}600{,}000 \text{ bytes}
\approx 146.5 \text{ MiB/checkpoint}.
\]

For 12 checkpoints, raw activation arrays are approximately:

\[
12 \times 146.5 \approx 1.72 \text{ GiB}.
\]

Annotations and indices add a relatively small amount.

Set:

```yaml
random_spatial_tokens_per_image: 25
```

to reduce activation storage and SAE training rows by roughly 4x while retaining stable corresponding spatial samples across checkpoints.

## Base-model anchors

A ResNet18 has roughly 11 million parameters. Approximate storage per full SGD anchor includes:

- model float32 tensors,
- momentum buffers,
- optimizer metadata,
- RNG state.

A practical estimate is 90 to 120 MiB per anchor. With anchors every 500 steps plus initialization and final state, expect roughly 1.5 to 2.0 GiB.

Exact size depends on serialization and classifier width.

## SAE parameters

For input dimension \(d=256\) and dictionary size \(m=4096\):

\[
md = 1{,}048{,}576
\]

weights exist in each of encoder and decoder.

At float32, encoder plus decoder require approximately:

\[
2 \times 1{,}048{,}576 \times 4
\approx 8.0 \text{ MiB}.
\]

AdamW adds first and second moments, so an SAE training checkpoint is typically several times the raw parameter size. A rough planning number is 30 to 45 MiB per checkpoint for standard SAEs, with small differences across variants.

## SAE activation-batch memory

For batch size 8,192 and dictionary size 4,096, a dense preactivation matrix contains:

\[
8192 \times 4096 = 33{,}554{,}432
\]

float values.

At float32 this is approximately 128 MiB for one tensor. Autograd retains several related tensors, so peak memory is much larger. On a modern GPU, 8,192 is reasonable but not mandatory.

For limited memory, reduce:

```yaml
sae:
  batch_size: 2048
```

BatchTopK still controls average L0, but changing batch size can affect which activations compete globally and should be reported as an experimental setting.

## Lineage signatures

Default signature memory is modest relative to activation caches:

- decoder: \(4096 \times 256\),
- activation embedding: \(4096 \times 256\),
- concept profile: \(4096 \times (312+200)\),
- image top examples and summary statistics.

In float32, this is on the order of tens of MiB per checkpoint.

## Procrustes alignment

With 50,000 rows and 256 dimensions, the sampled matrices each occupy about 49 MiB in float32 before conversion for the SVD. The SVD itself is only over a 256 x 256 cross-covariance matrix.

## Counterfactual replay cost

A replay from step 6,000 to 7,000 processes:

\[
1000 \times 64 = 64{,}000
\]

image presentations per trajectory.

Each causal test generally needs:

- one baseline replay, unless cached and already verified,
- one intervention replay,
- feature scoring on the probe set.

Use gradient ranking to reduce the number of exact replays, but never replace exact replay with the approximation for a headline claim.

## Recommended experiment scaling

### Development mode

- one model seed,
- 1,000 to 2,000 model steps,
- 1,000 activation images,
- 25 spatial tokens per image,
- 1,024-feature SAE,
- 500 to 1,000 SAE steps.

### Main single-seed run

Use `configs/cub_resnet18.yaml` as written.

### Paper run

Plan for:

- three base-model seeds,
- three SAE seeds for primary checkpoints,
- a smaller SAE frontier at the final checkpoint,
- five to twenty deeply tested feature genealogies,
- matched random replay controls.

Storage should remain comfortably below 100 GiB if intermediate SAE sweeps and duplicate activation caches are managed carefully.

## Included smoke run

The included CPU smoke run uses:

- 96 synthetic training images,
- 12 model steps,
- two activation checkpoints,
- 128 SAE features,
- target L0 4,
- 24 SAE steps per family.

It is intended to prove code paths, not to establish scientific performance.
