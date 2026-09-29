import json
from pathlib import Path

import numpy as np

from feature_genesis.activations import ActivationCache
from feature_genesis.config import load_config
from feature_genesis.determinism import resolve_device
from feature_genesis.evaluation.concepts import image_max_codes
from feature_genesis.lineage.signatures import FeatureSignatures
from feature_genesis.sae.io import latest_sae_path, load_sae, sae_analysis_root
from feature_genesis.sae.validation import require_stage_four_gate
from feature_genesis.training.checkpoints import read_json


CONFIG_PATH = Path("configs/quickdraw_dgx_v4.yaml")
SAE_KIND = "topk"
MODEL_STEP = None
TOP_TAIL_IMAGES = 32


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
    signatures = FeatureSignatures.load(
        sae_analysis_root(config, "signatures", step, SAE_KIND)
    )
    gate = read_json(
        Path(config.project.run_dir)
        / "sae_validation"
        / config.sae.artifact_version
        / "stage_04_gate.json"
    )
    replicated = np.asarray(
        gate["cross_seed_stability"]["replicated_primary_feature_ids"],
        dtype=np.int64,
    )
    ranked = np.argsort(signatures.frequency)[::-1]
    feature_ids = ranked[np.isin(ranked, replicated)][: config.evaluation.max_features]
    codes = image_max_codes(
        sae,
        cache,
        device,
        config.activations.batch_size,
        feature_ids,
    )
    labels = np.asarray(cache.labels[cache.image_offsets[:-1]], dtype=np.int64)
    metrics = feature_quality_metrics(
        codes,
        labels,
        feature_ids,
        cache.metadata["class_names"],
    )
    root = sae_analysis_root(config, "evaluation", step, SAE_KIND)
    root.mkdir(parents=True, exist_ok=True)
    path = root / "feature_quality.json"
    path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    print(path)


def feature_quality_metrics(
    codes: np.ndarray,
    labels: np.ndarray,
    feature_ids: np.ndarray,
    class_names: list[str],
) -> dict:
    rows = []
    for local_index, feature_id in enumerate(feature_ids):
        values = np.maximum(codes[:, local_index], 0)
        active = values > 0
        class_mass = np.bincount(
            labels,
            weights=values,
            minlength=len(class_names),
        ).astype(np.float64)
        probability = class_mass / max(class_mass.sum(), 1e-12)
        nonzero = probability > 0
        effective_classes = float(
            np.exp(-(probability[nonzero] * np.log(probability[nonzero])).sum())
        )
        top_count = min(TOP_TAIL_IMAGES, len(values))
        top = np.argsort(values)[::-1][:top_count]
        top_counts = np.bincount(labels[top], minlength=len(class_names))
        modal_label = int(np.argmax(top_counts))
        modal_share = float(top_counts[modal_label] / max(1, top_count))
        active_values = values[active]
        median = float(np.median(active_values)) if len(active_values) else 0.0
        p99 = float(np.quantile(active_values, 0.99)) if len(active_values) else 0.0
        frequency = float(active.mean())
        diversity = np.log1p(effective_classes) / np.log1p(max(2, len(class_names)))
        score = (
            np.log1p(max(p99, 0))
            * np.sqrt(max(frequency * (1 - frequency), 0))
            * diversity
            * (1 - modal_share)
        )
        rows.append(
            {
                "feature_id": int(feature_id),
                "candidate_score": float(score),
                "active_image_frequency": frequency,
                "active_images": int(active.sum()),
                "median_when_active": median,
                "p99_when_active": p99,
                "effective_classes_by_activation_mass": effective_classes,
                "top_32_modal_class": class_names[modal_label],
                "top_32_modal_class_share": modal_share,
                "top_32_unique_classes": int(len(np.unique(labels[top]))),
            }
        )
    rows.sort(key=lambda row: row["candidate_score"], reverse=True)
    return {
        "feature_count": len(rows),
        "features": rows,
        "selection_policy": {
            "uses_class_labels_as_names": False,
            "penalizes_class_collapse": True,
            "requires_three_seed_replication": True,
            "maximum_modal_class_share": 0.5,
            "minimum_effective_classes": 3.0,
        },
    }


if __name__ == "__main__":
    main()
