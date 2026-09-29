import json
from pathlib import Path

from feature_genesis.activations import ActivationCache
from feature_genesis.config import load_config
from feature_genesis.determinism import resolve_device
from feature_genesis.hashing import stable_seed
from feature_genesis.lineage.alignment import fit_cache_alignment
from feature_genesis.lineage.graph import build_lineage_graph, save_lineage_graph
from feature_genesis.lineage.matching import match_feature_sets
from feature_genesis.lineage.signatures import FeatureSignatures
from feature_genesis.progress import progress
from feature_genesis.sae.io import sae_analysis_root
from feature_genesis.sae.validation import require_stage_four_gate

CONFIG_PATH = Path("configs/quickdraw_dgx_v4.yaml")
SAE_KIND = "topk"


def main():
    config = load_config(CONFIG_PATH)
    require_stage_four_gate(config)
    device = resolve_device(config.project.device)
    steps = sorted(config.activations.checkpoint_steps)
    signatures = {
        step: FeatureSignatures.load(
            sae_analysis_root(config, "signatures", step, SAE_KIND)
        )
        for step in steps
    }
    pairwise = {}
    summary = []
    pairs = list(zip(steps[:-1], steps[1:], strict=True))
    for source_step, target_step in progress(pairs, desc="build lineage"):
        source_cache = ActivationCache.load(
            Path(config.project.run_dir) / config.activations.cache_dir / f"step_{source_step:08d}_{config.activations.split}"
        )
        target_cache = ActivationCache.load(
            Path(config.project.run_dir) / config.activations.cache_dir / f"step_{target_step:08d}_{config.activations.split}"
        )
        alignment = fit_cache_alignment(
            source_cache,
            target_cache,
            config.lineage.procrustes_samples,
            stable_seed(config.project.seed, "alignment", source_step, target_step),
        )
        result = match_feature_sets(signatures[source_step], signatures[target_step], config.lineage, alignment, device)
        pairwise[(source_step, target_step)] = result
        summary.append(
            {
                "source_step": source_step,
                "target_step": target_step,
                "alignment_residual_ratio": alignment.residual_ratio,
                "alignment_heldout_residual_ratio": alignment.heldout_residual_ratio,
                "continues": sum(match.relation == "continue" for match in result.matches),
                "births": len(result.births),
                "deaths": len(result.deaths),
                "splits": len(result.splits),
                "merges": len(result.merges),
            }
        )
    graph = build_lineage_graph(steps, signatures, pairwise)
    root = save_lineage_graph(
        graph,
        sae_analysis_root(config, "lineage", kind=SAE_KIND),
    )
    (root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(root)


if __name__ == "__main__":
    main()
