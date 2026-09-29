# Circuit Genesis

This stage extends Feature Genesis from one `layer2` sparse feature to a causally tested graph spanning `layer1`, `layer2`, `layer3`, `layer4`, and the selected output logit. It reuses the supervised checkpoints and the validated `layer2` SAE sequence. The classifier is not retrained.

## Scientific object

A node is a Top-K SAE feature at a named ResNet stage and spatial position distribution. A directed edge from source feature `i` to downstream feature `j` is screened with sparse Cross-Layer Attribution:

`E[i→j] = mean_x Σ_p z_i(x,p) <∂a_j(x)/∂h_i(x,p), s_i d_i>`

where `z_i` is the sparse source activation, `h_i` is the original ResNet activation, `d_i` is the unit decoder direction, and `s_i` is the SAE input scale. The computation is performed in the original model. Bootstrap intervals and across-batch sign agreement quantify screening uncertainty.

The final validated image cohort for each target feature is frozen before the temporal sweep. The downstream logit selected by target-feature ablation at the final checkpoint is also frozen. Every checkpoint therefore measures the same images, target feature identity, and output behavior.

## Causal validation

Source-node ablation tests the sign of important feature-to-feature and feature-to-logit edges. Joint graph sufficiency keeps the selected sparse contributions and removes non-circuit SAE contributions while preserving the exact SAE reconstruction residual. Joint necessity removes the selected contributions from the original activation.

The circuit gate requires:

- sufficient reconstruction of the model behavior;
- causal edge signs consistent with the attribution graph;
- a nonzero downstream effect from the original validated target feature;
- stable cross-checkpoint SAE alignment;
- persistence across consecutive checkpoints.

Candidate graphs are screened with up to 16 nodes per layer. Validation evaluates progressively larger subgraphs with 2, 4, 8, 12, and 16 nodes per layer and retains the smallest subgraph that reaches the faithfulness requirement. Failing to reach the threshold stops the pilot.

## Emergence

Circuit birth is the first persistent checkpoint where the causal gate passes, the graph sufficiently resembles the final graph, and the target affects the frozen downstream logit. The reported birth is an interval bounded by the previous observed checkpoint and the first persistent qualifying checkpoint.

## Training-data attribution

Candidate training batches are ranked by gradient alignment with the final circuit-only output. Exact replay masks only the selected examples at the selected training update, rebuilds the model deterministically, and measures the change in the circuit score. Proxy-ranked examples are not called builders or breakers until exact replay confirms their effect.

## Execution and manual gate

Run from `/home/boni/projects/feature_genesis` after activating `/home/boni/ai/envs/dgx-dl/bin/activate`:

1. `python scripts/22_cache_circuit_activations.py`
2. `python scripts/23_train_circuit_saes.py`
3. `python scripts/24_pilot_sparse_circuits.py`
4. Inspect `runs/quickdraw_resnet34_gn_seed1729_v4/circuits/v1_sparse_visual/pilot_gate.json`. Do not continue unless it passes and the three pilot graphs are visually coherent.
5. `python scripts/24_discover_sparse_circuits.py`
6. `python scripts/25_validate_sparse_circuits.py`
7. `python scripts/26_build_circuit_lineage.py`
8. `python scripts/27_rank_circuit_training_batches.py`
9. `python scripts/28_counterfactual_circuit_replay.py`
10. `python scripts/31_label_circuit_nodes_gemini.py`
11. Repeat step 10 while its printed `remaining` count is nonzero. Each invocation is resumable and capped by both node count and dollar budget.
12. `python scripts/29_make_circuit_report.py`
13. `python scripts/30_validate_circuit_artifacts.py`

The Gemini stage labels evidence-backed nodes and performs collision refinement. It does not select edges or decide causal validity.
