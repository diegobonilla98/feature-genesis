import json
from pathlib import Path

from feature_genesis.activations import ActivationCache
from feature_genesis.config import load_config
from feature_genesis.determinism import resolve_device
from feature_genesis.hashing import stable_seed
from feature_genesis.lineage.alignment import fit_cache_alignment
from feature_genesis.lineage.ancestry import resolve_birth_target
from feature_genesis.lineage.backward import backward_feature_trajectories
from feature_genesis.sae.io import latest_sae_path, load_sae, sae_analysis_root
from feature_genesis.sae.validation import require_stage_four_gate

CONFIG_PATH = Path("configs/quickdraw_dgx_v4.yaml")
SAE_KIND = "topk"
FEATURE_IDS = None


def main():
    config = load_config(CONFIG_PATH)
    require_stage_four_gate(config)
    device = resolve_device(config.project.device)
    steps = sorted(config.activations.checkpoint_steps)
    caches = {
        step: ActivationCache.load(
            Path(config.project.run_dir) / config.activations.cache_dir / f"step_{step:08d}_{config.activations.split}"
        )
        for step in steps
    }
    final_step = steps[-1]
    feature_ids = FEATURE_IDS
    if feature_ids is None:
        quality_path = (
            sae_analysis_root(config, "evaluation", final_step, SAE_KIND)
            / "feature_quality.json"
        )
        quality = json.loads(quality_path.read_text(encoding="utf-8"))
        feature_ids = sorted(int(row["feature_id"]) for row in quality["features"])
    final_sae, _ = load_sae(config, latest_sae_path(config, final_step, SAE_KIND), device)
    alignments = {final_step: None}
    for step in steps[:-1]:
        alignments[step] = fit_cache_alignment(
            caches[step],
            caches[final_step],
            config.lineage.procrustes_samples,
            stable_seed(config.project.seed, "backward-alignment", step, final_step),
        )
    result = backward_feature_trajectories(
        final_sae,
        caches,
        alignments,
        feature_ids,
        device,
        config.activations.batch_size,
        config.lineage.maximum_alignment_residual_ratio,
    )
    result["strongest_ancestry"] = {}
    for feature_id in feature_ids:
        birth_step, birth_feature_id, ancestry = resolve_birth_target(
            config,
            SAE_KIND,
            feature_id,
        )
        result["strongest_ancestry"][str(feature_id)] = {
            "structural_birth_step": int(ancestry["nodes"][0]["step"]),
            "structural_birth_feature_id": int(ancestry["nodes"][0]["feature"]),
            "attribution_target_step": birth_step,
            "attribution_target_feature_id": birth_feature_id,
            "nodes": ancestry["nodes"],
            "edges": ancestry["edges"],
        }
    root = sae_analysis_root(config, "lineage", kind=SAE_KIND)
    root.mkdir(parents=True, exist_ok=True)
    path = root / "backward_feature_trajectories.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
