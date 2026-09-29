from pathlib import Path

from feature_genesis.activations import ActivationCache
from feature_genesis.config import load_config
from feature_genesis.data.factory import build_datasets
from feature_genesis.determinism import resolve_device
from feature_genesis.labeling.cards import (
    add_blind_validation_tiles,
    build_feature_card,
    render_individual_discovery_evidence,
    render_individual_validation_evidence,
    write_feature_card,
)
from feature_genesis.progress import progress
from feature_genesis.sae.io import latest_sae_path, load_sae, sae_analysis_root
from feature_genesis.sae.validation import require_stage_four_gate

CONFIG_PATH = Path("configs/quickdraw_dgx_v4.yaml")
SAE_KIND = "topk"
MODEL_STEP = None
FEATURE_IDS = [0]


def main():
    config = load_config(CONFIG_PATH)
    require_stage_four_gate(config)
    step = max(config.activations.checkpoint_steps) if MODEL_STEP is None else MODEL_STEP
    device = resolve_device(config.project.device)
    cache = ActivationCache.load(
        Path(config.project.run_dir)
        / config.activations.cache_dir
        / f"step_{step:08d}_{config.activations.evaluation_split}"
    )
    sae, _ = load_sae(config, latest_sae_path(config, step, SAE_KIND), device)
    bundle = build_datasets(config, include_annotations=True, canonical_train=True)
    root = sae_analysis_root(config, "feature_cards", step, SAE_KIND)
    for feature_id in progress(FEATURE_IDS, desc="export feature cards"):
        card = add_blind_validation_tiles(
            config,
            build_feature_card(config, sae, cache, feature_id, device),
        )
        discovery = render_individual_discovery_evidence(
            config,
            bundle,
            cache,
            sae,
            card,
            root,
            device,
        )
        validation = render_individual_validation_evidence(
            config,
            bundle,
            card,
            root,
        )
        print(write_feature_card(card, root))
        for path in discovery + validation:
            print(path)


if __name__ == "__main__":
    main()
