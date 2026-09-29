# Reproducibility Contract

## Goal

The project distinguishes three levels of reproducibility:

1. Configuration reproducibility: another run uses the same declared settings.
2. Numerical trajectory reproducibility: a run follows the same floating-point optimization path.
3. Scientific reproducibility: the qualitative result survives new model and SAE seeds.

The exact replay subsystem targets level 2 on the same critical software and hardware environment. The paper protocol targets level 3 through replication.

## Stored run state

`train_project` writes:

```text
resolved_config.yaml
run_metadata.json
batch_plan.npz
anchors/anchor_step_XXXXXXXX.pt
telemetry.jsonl
result.json
```

Every configured developmental analysis step is saved as an anchor, even when it is not a multiple of the periodic anchor interval. Each anchor contains:

- model tensors,
- optimizer tensors and momentum buffers,
- Python RNG state,
- NumPy RNG state,
- torch CPU RNG state,
- every torch CUDA RNG state,
- model SHA-256,
- run metadata.

## Training configuration hash

The replay hash includes all fields that can affect training:

- project seed and requested device,
- deterministic settings except the verification toggle,
- data split, batch, transform, and synthetic-data settings,
- model architecture and hook definition,
- optimizer, schedule, loss, checkpoint, and clipping settings.

The dataset root path is excluded because data may be moved. The dataset contents are checked separately by signature.

SAE, lineage, attribution, and evaluation settings do not enter the model-training hash. Changing them must not invalidate reconstruction of the trained model.

## Dataset signature

For CUB, the signature includes SHA-256 hashes of all annotation and split metadata files plus a hash of the complete image inventory and file sizes.

For synthetic shapes, the signature includes split sizes and seed.

The CUB image inventory currently avoids hashing every image byte to keep startup practical. For archival publication, an optional full image-content hash can be generated externally and stored with the release.

## Batch-plan signature

The plan stores a rectangular index matrix, one epoch value per step, actual batch sizes, and the plan seed. Its metadata includes separate hashes for indices, epochs, sizes, and a combined plan hash.

Replay rejects a changed plan before loading the optimizer anchor.

## Environment signature

The original run records:

- full Python version,
- operating system and platform,
- machine architecture,
- torch version,
- torchvision version,
- NumPy version,
- Pillow version,
- CUDA runtime,
- cuDNN version,
- GPU identity,
- deterministic algorithm state,
- TF32 state,
- CUBLAS workspace configuration,
- complete `pip freeze`,
- git commit when available.

Replay compares the critical runtime fields before proceeding. Set `reproducibility.verify_environment: false` only for exploratory loading when bitwise equality is not required.

## Deterministic operations

The code configures:

```text
PYTHONHASHSEED
CUBLAS_WORKSPACE_CONFIG=:4096:8
random.seed
numpy.random.seed
torch.manual_seed
torch.cuda.manual_seed_all
torch.use_deterministic_algorithms
torch.backends.cudnn.deterministic
torch.backends.cudnn.benchmark = false
TF32 disabled
float32 matmul precision = highest
fixed CPU thread count
```

If a selected operation has no deterministic implementation, strict mode raises rather than silently continuing.

## Data-loader independence

The realized sample sequence comes from `ExactBatchPlan`. Random transforms are keyed by sample ID and epoch. As a result, changing worker completion order cannot change the augmentation attached to an example.

The run hash still includes `num_workers` to enforce a conservative exact configuration contract.

## Stateless learning-rate schedule

The learning rate is computed from `global_step` and total planned steps. No hidden scheduler state is required.

## Replay verification

Run:

```bash
python scripts/02_verify_replay.py
```

The script:

1. selects a stored target anchor,
2. loads the nearest earlier anchor,
3. restores optimizer and RNG state,
4. replays every intervening batch,
5. computes the reconstructed model SHA-256,
6. compares all parameters with the stored target.

A valid exact replay has:

```json
{
  "bitwise_equal": true,
  "max_abs_difference": 0.0
}
```

## Counterfactual replay invariants

A causal replay preserves:

- starting model and optimizer state,
- batch order,
- batch membership,
- per-sample augmentation,
- batch denominator,
- labels unless explicitly edited,
- learning-rate schedule,
- weight decay,
- momentum,
- all non-intervened inputs,
- all future random states.

The only intended difference is the explicit loss or image intervention.

## No-op controls

The test suite verifies that:

- masking a dataset index that never appears in the interval is bitwise identical,
- occluding an absent dataset index is bitwise identical,
- changing only SAE settings does not invalidate model replay,
- changing a training learning rate does invalidate replay,
- gradient screening advances the model and optimizer to the exact ordinary-training endpoint,
- SAE evaluation does not mutate dead-feature counters or inference-threshold state.

## Cross-machine limitation

PyTorch deterministic algorithms promise the same output for the same input on the same software and hardware. PyTorch does not guarantee bitwise identity across releases, platforms, or devices.

For a different machine, use the stored environment to reproduce the setup as closely as possible, but treat the result as a new scientific replication rather than an exact trajectory reconstruction.

## Publication replication plan

For each headline finding:

- train at least three independent base-model seeds,
- train at least three independent SAEs per relevant checkpoint and family,
- rerun the selected counterfactual from at least two preceding anchor widths,
- include matched random example and random region controls,
- report all seeds, including failed lineages,
- report effect size distributions and bootstrap confidence intervals,
- preregister feature-selection and interval-selection rules before final causal testing.

## Archival checklist

Before release:

1. Save the git commit.
2. Export the full environment.
3. Store all resolved configurations.
4. Verify every headline target anchor from an earlier anchor.
5. Hash the source tree and result artifacts.
6. Freeze selected feature IDs and causal candidates before exact replay.
7. Preserve raw output, not only figures.
8. Record CUB acquisition date and archive MD5.
9. State the exact hardware for every model and SAE run.
10. Keep the synthetic smoke proof runnable on CPU.
