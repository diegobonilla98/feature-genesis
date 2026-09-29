import json
import shutil
from pathlib import Path

import numpy as np

from feature_genesis.activations import ActivationCache, cache_checkpoint_activations
from feature_genesis.attribution.counterfactual import LossMask, counterfactual_replay
from feature_genesis.attribution.spatial import spatial_feature_evidence
from feature_genesis.config import load_config
from feature_genesis.data.batch_plan import ExactBatchPlan
from feature_genesis.data.factory import build_datasets, eval_loader
from feature_genesis.determinism import resolve_device
from feature_genesis.evaluation.concepts import heldout_binary_coalition_metrics, image_max_codes, many_to_one_concept_metrics
from feature_genesis.evaluation.fidelity import evaluate_sae_substitution
from feature_genesis.evaluation.report import write_research_report
from feature_genesis.hashing import stable_seed
from feature_genesis.lineage.alignment import fit_cache_alignment
from feature_genesis.lineage.graph import build_lineage_graph, save_lineage_graph
from feature_genesis.lineage.matching import match_feature_sets
from feature_genesis.lineage.signatures import compute_feature_signatures
from feature_genesis.progress import progress
from feature_genesis.sae.io import latest_sae_path, load_sae, sae_analysis_root
from feature_genesis.sae.trainer import train_continual_sae_sequence, train_sae
from feature_genesis.sae.validation import validate_stage_four
from feature_genesis.training.replay import reconstruct_state, verify_anchor_replay
from feature_genesis.training.trainer import train_project

CONFIG_PATH = Path("configs/smoke.yaml")
OVERWRITE = True


def main():
    config = load_config(CONFIG_PATH)
    run_dir = Path(config.project.run_dir)
    if OVERWRITE and run_dir.exists():
        shutil.rmtree(run_dir)
    training_result = train_project(config, overwrite=False)
    replay_result = verify_anchor_replay(config, 12)
    (run_dir / "replay_verification.json").write_text(json.dumps(replay_result, indent=2, sort_keys=True), encoding="utf-8")
    activation_roots = {}
    for step in progress(config.activations.checkpoint_steps, desc="cache activations"):
        activation_roots[step] = cache_checkpoint_activations(config, step)
    continual = []
    for sae_seed in config.sae.seeds:
        continual.extend(
            train_continual_sae_sequence(
                config,
                activation_roots,
                "batch_topk",
                sae_seed=sae_seed,
            )
        )
    benchmarks = []
    for sae_seed in config.sae.seeds:
        benchmarks.append(
            train_sae(
                config,
                activation_roots[12],
                12,
                "batch_topk",
                sae_seed=sae_seed,
                role="independent",
            )
        )
    for kind in [
        "cosine_batch_topk",
        "jumprelu",
        "matryoshka",
        "matching_pursuit",
    ]:
        benchmarks.append(
            train_sae(
                config,
                activation_roots[12],
                12,
                kind,
                sae_seed=config.sae.primary_seed,
                role="independent",
            )
        )
    stage_four_gate = validate_stage_four(config)
    if not stage_four_gate["passed"]:
        raise RuntimeError("Smoke stage 4 validation gate failed")
    device = resolve_device(config.project.device)
    signatures = {}
    for step in progress(config.activations.checkpoint_steps, desc="compute signatures"):
        cache = ActivationCache.load(activation_roots[step])
        sae, _ = load_sae(config, latest_sae_path(config, step, "batch_topk"), device)
        signature = compute_feature_signatures(
            sae,
            cache,
            device,
            config.activations.batch_size,
            config.project.seed,
            probe_rows=128,
            projection_dim=32,
            top_images=config.evaluation.top_examples,
        )
        signature.save(
            sae_analysis_root(config, "signatures", step, "batch_topk")
        )
        signatures[step] = signature
    source_cache = ActivationCache.load(activation_roots[6])
    target_cache = ActivationCache.load(activation_roots[12])
    alignment = fit_cache_alignment(
        source_cache,
        target_cache,
        config.lineage.procrustes_samples,
        stable_seed(config.project.seed, "smoke-alignment"),
    )
    pair = match_feature_sets(signatures[6], signatures[12], config.lineage, alignment, device)
    graph = build_lineage_graph([6, 12], signatures, {(6, 12): pair})
    lineage_root = save_lineage_graph(
        graph,
        sae_analysis_root(config, "lineage", kind="batch_topk"),
    )
    lineage_summary = {
        "alignment_residual_ratio": alignment.residual_ratio,
        "alignment_heldout_residual_ratio": alignment.heldout_residual_ratio,
        "continues": sum(match.relation == "continue" for match in pair.matches),
        "births": len(pair.births),
        "deaths": len(pair.deaths),
        "splits": len(pair.splits),
        "merges": len(pair.merges),
    }
    (lineage_root / "summary.json").write_text(json.dumps(lineage_summary, indent=2, sort_keys=True), encoding="utf-8")
    final_sae, _ = load_sae(config, latest_sae_path(config, 12, "batch_topk"), device)
    feature_id = int(np.argmax(signatures[12].frequency))
    selected_features = np.argsort(signatures[12].frequency)[::-1][: config.evaluation.max_features]
    codes = image_max_codes(final_sae, target_cache, device, config.activations.batch_size, selected_features)
    concept_metrics = many_to_one_concept_metrics(
        codes,
        np.asarray(target_cache.image_attributes, dtype=np.int64),
        selected_features,
        target_cache.metadata["attribute_names"],
        minimum_positives=2,
    )
    evaluation_root = sae_analysis_root(
        config,
        "evaluation",
        12,
        "batch_topk",
    )
    evaluation_root.mkdir(parents=True, exist_ok=True)
    (evaluation_root / "concept_metrics.json").write_text(json.dumps(concept_metrics, indent=2, sort_keys=True), encoding="utf-8")
    coalition_metrics = heldout_binary_coalition_metrics(
        codes,
        np.asarray(target_cache.image_attributes, dtype=np.int64),
        selected_features,
        target_cache.metadata["attribute_names"],
        seed=config.project.seed,
        max_features_per_concept=4,
        minimum_positives=2,
    )
    (evaluation_root / "coalition_metrics.json").write_text(
        json.dumps(coalition_metrics, indent=2, sort_keys=True), encoding="utf-8"
    )
    model, _, fidelity_bundle, _ = reconstruct_state(config, 12)
    fidelity = evaluate_sae_substitution(
        model,
        final_sae,
        eval_loader(fidelity_bundle.validation, config),
        config.model.hook_layer,
        device,
    )
    (evaluation_root / "fidelity.json").write_text(json.dumps(fidelity, indent=2, sort_keys=True), encoding="utf-8")
    spatial = spatial_feature_evidence(
        final_sae,
        target_cache,
        feature_id,
        device,
        config.data.image_size,
        config.evaluation.part_radius_fraction,
        config.evaluation.top_examples,
    )
    (evaluation_root / "spatial_evidence.json").write_text(json.dumps(spatial, indent=2, sort_keys=True), encoding="utf-8")
    plan = ExactBatchPlan.load(run_dir / "batch_plan.npz")
    masked_index = plan.batch(6)[0].index
    bundle = build_datasets(config, include_annotations=False)
    counterfactual = counterfactual_replay(
        config,
        final_sae,
        feature_id,
        6,
        12,
        LossMask(frozenset({masked_index}), frozenset({6})),
        bundle,
    )
    attribution_root = (
        sae_analysis_root(config, "attribution", kind="batch_topk")
        / f"feature_{feature_id:05d}"
    )
    attribution_root.mkdir(parents=True, exist_ok=True)
    (attribution_root / "counterfactual_replay.json").write_text(json.dumps(counterfactual, indent=2, sort_keys=True), encoding="utf-8")
    smoke_result = {
        "training": training_result,
        "replay": replay_result,
        "sae_continual": continual,
        "sae_benchmarks": benchmarks,
        "stage_four_gate": stage_four_gate,
        "lineage": lineage_summary,
        "selected_feature": feature_id,
        "concept_mean_best_ap": concept_metrics["mean_best_ap"],
        "coalition_mean_test_f1": coalition_metrics["mean_test_f1"],
        "sae_fidelity": fidelity,
        "part_hit_rate": spatial["part_hit_rate"],
        "counterfactual": counterfactual,
    }
    (run_dir / "smoke_result.json").write_text(json.dumps(smoke_result, indent=2, sort_keys=True), encoding="utf-8")
    write_research_report(
        run_dir / config.evaluation.report_dir,
        "Feature Genesis Smoke Report",
        [
            ("Deterministic replay", replay_result),
            ("SAE benchmark", benchmarks),
            ("Lineage", lineage_summary),
            ("Concept evaluation", {key: value for key, value in concept_metrics.items() if key != "concepts"}),
            ("Feature coalition evaluation", {key: value for key, value in coalition_metrics.items() if key != "concepts"}),
            ("SAE model-level fidelity", fidelity),
            ("Counterfactual replay", counterfactual),
        ],
    )
    print(json.dumps(smoke_result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
