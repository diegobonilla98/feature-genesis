# Executed Proof Artifacts

These artifacts were produced by the included CPU smoke experiment. The smoke dataset is a tiny deterministic shapes benchmark, not CUB-200-2011, and its numerical metrics are implementation checks rather than scientific results.

## Commands

```bash
python -m compileall -q src scripts tests
pytest -q
python scripts/smoke_all.py
```

## Verified properties

- 24 tests pass.
- Replaying optimization steps 6 through 12 from the saved step-6 anchor produces a bitwise-identical model.
- The expected and replayed model hashes are identical.
- The maximum absolute parameter difference is exactly zero.
- BatchTopK realizes its exact batch sparsity budget.
- Matryoshka reconstructions use prefixes of one shared global sparse code.
- JumpReLU threshold gradients and initial L0 calibration behave as specified.
- Matching Pursuit residual norms are monotone and selected atoms satisfy the tested conditional-orthogonality property.
- Procrustes alignment recovers a known rotation.
- Lineage matching detects controlled continuation, birth, and split cases.
- SAE checkpoint loading restores the architecture-defining hyperparameters stored with the checkpoint.
- Feature ablation correctly converts normalized SAE contributions back to raw activation units.
- Known binary concept coalitions can be recovered by the evaluation procedure.
- Every declared developmental analysis step is saved as an exact replay anchor.
- Post-training BatchTopK threshold calibration reaches the requested mean L0 on distinct calibration scores.
- Evaluation forwards do not mutate SAE tracking buffers.
- Gradient screening follows the evolving baseline optimizer trajectory and reaches the exact ordinary-training endpoint.

## Included files

- `pytest.txt`: final test output.
- `compileall.txt`: syntax compilation output.
- `smoke_stdout.txt`: full smoke command output.
- `smoke_result.json`: aggregated smoke measurements.
- `replay_verification.json`: exact replay result.
- `environment.json`: critical execution environment and installed-package fingerprint.
- `report.html`: generated smoke report.
- `MANIFEST.json`: SHA-256 hashes for every proof artifact.

The full smoke run, including anchors, compact SAE checkpoints, signatures, lineage files, and activation caches, is retained under `runs/smoke/` in the archive.
