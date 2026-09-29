import json
from pathlib import Path

import numpy as np

from feature_genesis.activations import ActivationCache
from feature_genesis.config import load_config
from feature_genesis.data.factory import build_datasets
from feature_genesis.determinism import resolve_device
from feature_genesis.evaluation.concepts import image_max_codes
from feature_genesis.labeling.cards import (
    add_blind_validation_tiles,
    build_feature_card,
    build_feature_card_from_scores,
    render_individual_discovery_evidence,
    render_individual_validation_evidence,
    write_feature_card,
)
from feature_genesis.labeling.gemini import evaluate_feature_cards
from feature_genesis.sae.io import (
    latest_sae_path,
    load_sae,
    sae_analysis_root,
    sae_root,
)
from feature_genesis.sae.validation import require_stage_four_gate
from feature_genesis.training.checkpoints import read_json


CONFIG_PATH = Path("configs/quickdraw_dgx_v4.yaml")
SAE_KIND = "topk"
MODEL_STEP = None
USE_FEATURE_QUALITY_CANDIDATES = True
REUSE_EXISTING_CARDS = True


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
    sae_path = latest_sae_path(
        config,
        step,
        SAE_KIND,
        config.sae.primary_seed,
        "continual",
    )
    sae, _ = load_sae(config, sae_path, device)
    sae.eval()
    feature_ids = (
        quality_feature_ids(config, step, SAE_KIND)
        if USE_FEATURE_QUALITY_CANDIDATES
        else selected_feature_ids(
            config,
            step,
            SAE_KIND,
            sae,
            cache,
            device,
        )
    )
    bundle = build_datasets(config, include_annotations=True, canonical_train=True)
    root = sae_analysis_root(config, "feature_evaluation", step, SAE_KIND)
    card_root = root / "cards"
    evidence_root = root / "evidence"
    feature_inputs_by_id = {}
    missing_feature_ids = []
    for feature_id in feature_ids:
        feature_id = int(feature_id)
        card_path = card_root / f"feature_{feature_id:05d}.json"
        cached = (
            cached_feature_input(card_path)
            if REUSE_EXISTING_CARDS and card_path.exists()
            else None
        )
        if cached is None:
            missing_feature_ids.append(feature_id)
        else:
            feature_inputs_by_id[feature_id] = cached
    if missing_feature_ids:
        scores = image_max_codes(
            sae,
            cache,
            device,
            config.activations.batch_size,
            np.asarray(missing_feature_ids, dtype=np.int64),
        )
    else:
        scores = np.empty((0, 0), dtype=np.float32)
    for local_index, feature_id in enumerate(missing_feature_ids):
        card = build_feature_card_from_scores(
            config,
            sae,
            cache,
            feature_id,
            scores[:, local_index],
        )
        card = add_blind_validation_tiles(config, card)
        card["trajectory_uid"] = trajectory_uid(
            config,
            step,
            SAE_KIND,
            int(feature_id),
        )
        card["supplemental_evidence"] = supplemental_evidence(
            config,
            step,
            SAE_KIND,
            int(feature_id),
        )
        discovery_images = render_individual_discovery_evidence(
            config,
            bundle,
            cache,
            sae,
            card,
            evidence_root,
            device,
        )
        validation_images = render_individual_validation_evidence(
            config,
            bundle,
            card,
            evidence_root,
        )
        card_path = write_feature_card(card, card_root)
        feature_inputs_by_id[feature_id] = {
            "card": card,
            "card_path": card_path,
            "discovery_images": discovery_images,
            "validation_images": validation_images,
        }
    feature_inputs = [feature_inputs_by_id[int(feature_id)] for feature_id in feature_ids]
    manifest = evaluate_feature_cards(config, feature_inputs, root / "gemini")
    if manifest.get("status") != "complete":
        raise RuntimeError(
            f"Gemini evaluation did not complete: {manifest.get('reason', manifest['path'])}"
        )
    if int(manifest.get("features", -1)) != len(feature_inputs):
        raise RuntimeError(
            f"Gemini evaluated {manifest.get('features')} of {len(feature_inputs)} features"
        )
    print(manifest["path"])


def quality_feature_ids(config, model_step: int, kind: str) -> np.ndarray:
    path = (
        sae_analysis_root(config, "evaluation", model_step, kind)
        / "feature_quality.json"
    )
    payload = read_json(path)
    return np.asarray(
        [int(row["feature_id"]) for row in payload["features"]],
        dtype=np.int64,
    )


def cached_feature_input(card_path: Path) -> dict | None:
    card = read_json(card_path)
    discovery_rows = card.get("individual_evidence", {}).get("discovery", [])
    validation_rows = card.get("individual_evidence", {}).get("validation", [])
    discovery_images = [Path(row["paths"][0]) for row in discovery_rows]
    validation_images = [Path(row["path"]) for row in validation_rows]
    paths = discovery_images + validation_images
    if not discovery_images or not validation_images or not all(path.exists() for path in paths):
        return None
    return {
        "card": card,
        "card_path": card_path,
        "discovery_images": discovery_images,
        "validation_images": validation_images,
    }


def selected_feature_ids(
    config,
    model_step: int,
    kind: str,
    sae,
    cache,
    device,
) -> np.ndarray:
    if config.evaluation.gemini_feature_ids:
        return np.asarray(config.evaluation.gemini_feature_ids, dtype=np.int64)
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
    summary = read_json(
        sae_root(
            config,
            model_step,
            kind,
            config.sae.primary_seed,
            "continual",
        )
        / "summary.json"
    )
    frequency = np.asarray(
        summary["metrics"]["feature_frequency"],
        dtype=np.float32,
    )
    eligible = np.flatnonzero(
        np.isfinite(frequency)
    )
    eligible = np.intersect1d(eligible, replicated, assume_unique=False)
    codes = image_max_codes(
        sae,
        cache,
        device,
        config.activations.batch_size,
        eligible,
    )
    active = codes > 0
    active_images = active.sum(axis=0)
    image_frequency = active.mean(axis=0)
    labels = np.asarray(
        cache.labels[cache.image_offsets[:-1]],
        dtype=np.int64,
    )
    required_positive = (
        config.evaluation.gemini_top_positives
        + config.evaluation.gemini_borderline_positives
        + config.evaluation.gemini_validation_positives
    )
    required_negative = (
        config.evaluation.gemini_hard_negatives
        + config.evaluation.gemini_validation_negatives
    )
    valid = (
        (active_images >= required_positive)
        & ((len(codes) - active_images) >= required_negative)
        & (
            image_frequency
            <= config.evaluation.gemini_maximum_image_frequency
        )
    )
    modal_share, effective_classes = class_confounding_metrics(codes, labels)
    valid &= modal_share <= config.evaluation.gemini_max_modal_class_share
    valid &= effective_classes >= config.evaluation.gemini_minimum_effective_classes
    eligible = eligible[valid]
    codes = codes[:, valid]
    active = active[:, valid]
    image_frequency = image_frequency[valid]
    modal_share = modal_share[valid]
    effective_classes = effective_classes[valid]
    mean_when_active = np.asarray(
        [
            codes[:, index][active[:, index]].mean()
            for index in range(len(eligible))
        ],
        dtype=np.float32,
    )
    class_diversity = np.log1p(effective_classes) / np.log1p(
        max(2, len(cache.metadata.get("class_names", [])))
    )
    informativeness = (
        mean_when_active
        * np.sqrt(image_frequency * (1 - image_frequency))
        * (1 - image_frequency)
        * class_diversity
        * (1 - modal_share)
    )
    directions = (
        sae.decoder_directions()
        .detach()
        .cpu()
        .numpy()[eligible]
    )
    order = diverse_order(
        informativeness,
        directions,
        config.evaluation.gemini_diversity_penalty,
    )
    return eligible[order[: config.evaluation.gemini_max_features]]


def class_confounding_metrics(
    codes: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    feature_count = codes.shape[1]
    class_count = int(labels.max()) + 1
    mass = np.zeros((class_count, feature_count), dtype=np.float64)
    for label in range(class_count):
        mass[label] = np.maximum(codes[labels == label], 0).sum(axis=0)
    probability = mass / np.maximum(mass.sum(axis=0, keepdims=True), 1e-12)
    entropy = -(probability * np.log(np.maximum(probability, 1e-12))).sum(axis=0)
    effective_classes = np.exp(entropy).astype(np.float32)
    top_count = min(32, len(codes))
    top_indices = np.argpartition(codes, -top_count, axis=0)[-top_count:]
    modal_share = np.zeros(feature_count, dtype=np.float32)
    for feature_index in range(feature_count):
        top_labels = labels[top_indices[:, feature_index]]
        modal_share[feature_index] = np.bincount(top_labels).max() / top_count
    return modal_share, effective_classes


def concept_average_precision(
    config,
    model_step: int,
    kind: str,
    feature_ids: np.ndarray,
) -> np.ndarray:
    path = (
        sae_analysis_root(config, "evaluation", model_step, kind)
        / "concept_metrics.json"
    )
    if not path.exists():
        return np.zeros(len(feature_ids), dtype=np.float32)
    metrics = read_json(path)
    selected = np.asarray(metrics["feature_ids"], dtype=np.int64)
    values = np.asarray(
        metrics["single_feature"]["feature_best_ap"],
        dtype=np.float32,
    )
    lookup = {
        int(feature_id): float(values[index])
        for index, feature_id in enumerate(selected)
    }
    return np.asarray(
        [lookup.get(int(feature_id), 0.0) for feature_id in feature_ids],
        dtype=np.float32,
    )


def diverse_order(
    scores: np.ndarray,
    directions: np.ndarray,
    penalty: float,
) -> np.ndarray:
    if len(scores) == 0:
        return np.empty(0, dtype=np.int64)
    span = float(scores.max() - scores.min())
    normalized = (
        np.ones_like(scores)
        if span <= 1e-12
        else (scores - scores.min()) / span
    )
    similarities = np.maximum(0, directions @ directions.T)
    selected = []
    remaining = set(range(len(scores)))
    while remaining:
        if not selected:
            best = int(np.argmax(normalized))
        else:
            candidates = np.asarray(sorted(remaining), dtype=np.int64)
            redundancy = similarities[np.ix_(candidates, selected)].max(axis=1)
            values = normalized[candidates] - penalty * redundancy
            best = int(candidates[int(np.argmax(values))])
        selected.append(best)
        remaining.remove(best)
    return np.asarray(selected, dtype=np.int64)


def supplemental_evidence(
    config,
    model_step: int,
    kind: str,
    feature_id: int,
) -> dict:
    evidence = {}
    concept_path = (
        sae_analysis_root(config, "evaluation", model_step, kind)
        / "concept_metrics.json"
    )
    if concept_path.exists():
        concept_metrics = read_json(concept_path)
        selected = concept_metrics.get("feature_ids", [])
        if feature_id in selected:
            local_index = selected.index(feature_id)
            best_concept = concept_metrics["single_feature"]["feature_best_concept"][local_index]
            best_ap = concept_metrics["single_feature"]["feature_best_ap"][local_index]
            concept_rows = {
                row["concept_id"]: row
                for row in concept_metrics["single_feature"]["concepts"]
            }
            evidence["heldout_attribute_retrieval"] = {
                "concept_id": best_concept,
                "concept_name": concept_rows.get(best_concept, {}).get("name"),
                "average_precision": best_ap,
            }
    attribution_root = (
        sae_analysis_root(config, "attribution", kind=kind)
        / f"feature_{feature_id:05d}"
    )
    counterfactuals = {}
    if attribution_root.exists():
        for path in sorted(attribution_root.glob("counterfactual_*.json")):
            counterfactuals[path.stem] = json.loads(path.read_text(encoding="utf-8"))
    if counterfactuals:
        evidence["counterfactual_training_replay"] = counterfactuals
    return evidence


def trajectory_uid(
    config,
    model_step: int,
    kind: str,
    feature_id: int,
) -> str | None:
    path = sae_analysis_root(config, "lineage", kind=kind) / "lineage.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    target = f"s{model_step}:f{feature_id}"
    for node in data.get("nodes", []):
        if node.get("id") == target:
            return node.get("trajectory_uid")
    return None


if __name__ == "__main__":
    main()
