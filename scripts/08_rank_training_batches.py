import json
from pathlib import Path

from feature_genesis.activations import ActivationCache
from feature_genesis.attribution.gradient import (
    influences_to_json,
    rank_candidate_batches_multi,
)
from feature_genesis.config import load_config
from feature_genesis.data.factory import build_datasets
from feature_genesis.hashing import stable_seed
from feature_genesis.hashing import state_dict_hash
from feature_genesis.lineage.alignment import fit_cache_alignment
from feature_genesis.lineage.ancestry import resolve_birth_target
from feature_genesis.sae.io import latest_sae_path, load_sae, sae_analysis_root
from feature_genesis.sae.validation import require_stage_four_gate
from feature_genesis.training.replay import reconstruct_state

CONFIG_PATH = Path("configs/quickdraw_dgx_v4.yaml")
SAE_KIND = "topk"
TARGET_STEP = None
START_STEP = None
REUSE_EXISTING = True


def main():
    config = load_config(CONFIG_PATH)
    require_stage_four_gate(config)
    training_bundle = build_datasets(config, include_annotations=False)
    groups = {}
    for feature_id in accepted_feature_ids(config):
        spec = feature_spec(config, feature_id)
        if REUSE_EXISTING and spec["path"].exists():
            print(spec["path"])
            continue
        groups.setdefault((spec["start_step"], spec["target_step"]), []).append(spec)
    for specs in groups.values():
        rank_feature_group(config, training_bundle, specs)


def feature_spec(config, feature_id):
    root = (
        sae_analysis_root(config, "attribution", kind=SAE_KIND)
        / f"feature_{feature_id:05d}"
    )
    root.mkdir(parents=True, exist_ok=True)
    path = root / "gradient_ranked_batches.json"
    target_step, target_feature_id, ancestry = resolve_birth_target(
        config,
        SAE_KIND,
        feature_id,
    )
    if TARGET_STEP is not None:
        target_step = TARGET_STEP
        target_feature_id = feature_id
    available_sources = [
        step for step in config.activations.checkpoint_steps if step < target_step
    ]
    start_step = START_STEP if START_STEP is not None else max(available_sources)
    return {
        "feature_id": feature_id,
        "target_feature_id": target_feature_id,
        "ancestry": ancestry,
        "start_step": start_step,
        "target_step": target_step,
        "path": path,
    }


def rank_feature_group(config, training_bundle, specs):
    start_step = specs[0]["start_step"]
    target_step = specs[0]["target_step"]
    model, optimizer, _, plan = reconstruct_state(config, start_step, bundle=training_bundle)
    sae, _ = load_sae(
        config,
        latest_sae_path(config, target_step, SAE_KIND),
        next(model.parameters()).device,
    )
    source_cache = ActivationCache.load(
        Path(config.project.run_dir) / config.activations.cache_dir / f"step_{start_step:08d}_{config.activations.split}"
    )
    target_cache = ActivationCache.load(
        Path(config.project.run_dir) / config.activations.cache_dir / f"step_{target_step:08d}_{config.activations.split}"
    )
    alignment = fit_cache_alignment(
        source_cache,
        target_cache,
        config.lineage.procrustes_samples,
        stable_seed(config.project.seed, "attribution-alignment", start_step, target_step),
    )
    target_feature_ids = [spec["target_feature_id"] for spec in specs]
    if len(target_feature_ids) != len(set(target_feature_ids)):
        raise RuntimeError("Validated features share an attribution target feature")
    values_by_feature = rank_candidate_batches_multi(
        model,
        optimizer,
        sae,
        training_bundle,
        plan,
        config,
        target_feature_ids,
        start_step,
        target_step,
        alignment,
    )
    replayed_model_sha256 = state_dict_hash(model.state_dict())
    expected_model, _, _, _ = reconstruct_state(
        config,
        target_step,
        bundle=training_bundle,
    )
    expected_model_sha256 = state_dict_hash(expected_model.state_dict())
    if replayed_model_sha256 != expected_model_sha256:
        raise RuntimeError(
            "Attribution replay diverged from the authoritative target checkpoint"
        )
    for spec in specs:
        values = values_by_feature[spec["target_feature_id"]]
        count = config.attribution.candidate_batches
        builders = values[:count]
        breakers = sorted(values, key=lambda item: item.influence)[:count]
        payload = {
            "feature_id": spec["feature_id"],
            "target_feature_id": spec["target_feature_id"],
            "start_step": start_step,
            "target_step": target_step,
            "strongest_ancestry": spec["ancestry"],
            "influence_method": "clipped_gradient_alignment_proxy",
            "example_selection_method": "loss_weighted_within_ranked_batch_proxy",
            "replayed_model_sha256": replayed_model_sha256,
            "expected_model_sha256": expected_model_sha256,
            "builders": influences_to_json(builders),
            "breakers": influences_to_json(breakers),
        }
        spec["path"].write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        print(spec["path"])


def accepted_feature_ids(config):
    step = max(config.activations.checkpoint_steps)
    root = sae_analysis_root(config, "feature_evaluation", step, SAE_KIND) / "gemini"
    feature_ids = []
    for path in sorted(root.glob("feature_*_evaluation.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        if row["validated_status"] == "accepted":
            feature_ids.append(int(row["feature_id"]))
    return feature_ids


if __name__ == "__main__":
    main()
