import json
import shutil
from pathlib import Path

import numpy as np

from feature_genesis.activations import ActivationCache, cache_checkpoint_activations
from feature_genesis.config import load_config
from feature_genesis.data.factory import build_datasets
from feature_genesis.determinism import resolve_device
from feature_genesis.evaluation.concepts import image_max_codes
from feature_genesis.labeling.cards import (
    add_blind_validation_tiles,
    build_feature_card,
    render_individual_discovery_evidence,
    render_individual_validation_evidence,
    write_feature_card,
)
from feature_genesis.sae.io import latest_sae_path, load_sae
from feature_genesis.sae.trainer import train_continual_sae_sequence
from feature_genesis.training.trainer import train_project


CONFIG_PATH = Path("configs/quickdraw_smoke.yaml")
OVERWRITE = True


def main():
    config = load_config(CONFIG_PATH)
    run_root = Path(config.project.run_dir)
    if OVERWRITE and run_root.exists():
        shutil.rmtree(run_root)
    training = train_project(config)
    activation_roots = {}
    for step in config.activations.checkpoint_steps:
        activation_roots[step] = cache_checkpoint_activations(
            config,
            step,
            config.activations.split,
            config.activations.max_images,
            config.activations.random_spatial_tokens_per_image,
        )
    final_step = max(config.activations.checkpoint_steps)
    evaluation_root = cache_checkpoint_activations(
        config,
        final_step,
        config.activations.evaluation_split,
        config.activations.evaluation_max_images,
        0,
    )
    train_continual_sae_sequence(
        config,
        activation_roots,
        "batch_topk",
        sae_seed=config.sae.primary_seed,
    )
    device = resolve_device(config.project.device)
    sae, _ = load_sae(
        config,
        latest_sae_path(config, final_step, "batch_topk"),
        device,
    )
    cache = ActivationCache.load(evaluation_root)
    codes = image_max_codes(
        sae,
        cache,
        device,
        config.activations.batch_size,
    )
    active_counts = (codes > 0).sum(axis=0)
    eligible = np.flatnonzero(
        (active_counts >= 8)
        & ((len(codes) - active_counts) >= 4)
    )
    if len(eligible) == 0:
        raise RuntimeError("QuickDraw smoke SAE produced no explorable feature")
    feature_id = int(eligible[np.argmax(codes[:, eligible].max(axis=0))])
    bundle = build_datasets(config, include_annotations=True, canonical_train=True)
    card = add_blind_validation_tiles(
        config,
        build_feature_card(config, sae, cache, feature_id, device),
    )
    evidence_root = run_root / "smoke_feature_evidence"
    discovery = render_individual_discovery_evidence(
        config,
        bundle,
        cache,
        sae,
        card,
        evidence_root,
        device,
    )
    validation = render_individual_validation_evidence(
        config,
        bundle,
        card,
        evidence_root,
    )
    card_path = write_feature_card(card, evidence_root)
    result = {
        "status": "passed",
        "training": training,
        "selected_feature": feature_id,
        "card": str(card_path),
        "discovery_images": [str(path) for path in discovery],
        "validation_images": [str(path) for path in validation],
    }
    path = run_root / "quickdraw_v3_smoke_result.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
