from __future__ import annotations

from typing import Any

import numpy as np
import torch
from sklearn.metrics import average_precision_score

from feature_genesis.activations import ActivationCache
from feature_genesis.progress import progress
from feature_genesis.sae.core import SparseAutoencoder


@torch.no_grad()
def image_max_codes(
    sae: SparseAutoencoder,
    cache: ActivationCache,
    device: torch.device,
    batch_size: int,
    feature_ids: np.ndarray | None = None,
) -> np.ndarray:
    image_count = len(cache.image_offsets) - 1
    selected = np.arange(sae.dictionary_size) if feature_ids is None else np.asarray(feature_ids, dtype=np.int64).copy()
    output = np.zeros((image_count, len(selected)), dtype=np.float32)
    tokens_per_image = max(1, int(cache.metadata["tokens_per_image"]))
    images_per_group = max(1, batch_size // tokens_per_image)
    for image_start in progress(
        range(0, image_count, images_per_group),
        desc="image max codes",
    ):
        image_end = min(image_count, image_start + images_per_group)
        row_start = int(cache.image_offsets[image_start])
        row_end = int(cache.image_offsets[image_end])
        inputs = torch.from_numpy(
            np.array(
                cache.activations[row_start:row_end],
                dtype=np.float32,
                copy=True,
            )
        ).to(device)
        codes = sae.encode(inputs, inference=True)[:, selected]
        for image_index in range(image_start, image_end):
            local_start = int(cache.image_offsets[image_index]) - row_start
            local_end = int(cache.image_offsets[image_index + 1]) - row_start
            output[image_index] = codes[local_start:local_end].amax(dim=0).cpu().numpy()
    return output


def many_to_one_concept_metrics(
    image_codes: np.ndarray,
    attributes: np.ndarray,
    feature_ids: np.ndarray,
    attribute_names: list[str],
    minimum_positives: int = 5,
    attribute_certainty: np.ndarray | None = None,
) -> dict[str, Any]:
    image_codes = np.asarray(image_codes, dtype=np.float32)
    attributes = np.asarray(attributes, dtype=np.int64)
    certainty = (
        None
        if attribute_certainty is None
        else np.asarray(attribute_certainty, dtype=np.float32) / 4.0
    )
    concept_rows = []
    feature_best_ap = np.zeros(image_codes.shape[1], dtype=np.float32)
    feature_best_concept = np.full(image_codes.shape[1], -1, dtype=np.int32)
    for concept in progress(range(attributes.shape[1]), desc="concept ap"):
        targets = attributes[:, concept]
        positives = int(targets.sum())
        if positives < minimum_positives or positives == len(targets):
            continue
        weights = None if certainty is None else certainty[:, concept]
        scores = np.asarray(
            [
                average_precision_score(
                    targets,
                    image_codes[:, feature],
                    sample_weight=weights,
                )
                for feature in range(image_codes.shape[1])
            ]
        )
        best = int(scores.argmax())
        baseline = (
            positives / len(targets)
            if weights is None
            else float((weights * targets).sum() / max(weights.sum(), 1e-12))
        )
        concept_rows.append(
            {
                "concept_id": concept,
                "name": attribute_names[concept] if concept < len(attribute_names) else str(concept),
                "positives": positives,
                "baseline_ap": baseline,
                "best_feature": int(feature_ids[best]),
                "best_ap": float(scores[best]),
                "normalized_ap_gain": float((scores[best] - baseline) / max(1 - baseline, 1e-12)),
            }
        )
        improved = scores > feature_best_ap
        feature_best_ap[improved] = scores[improved]
        feature_best_concept[improved] = concept
    concept_rows.sort(key=lambda row: row["normalized_ap_gain"], reverse=True)
    return {
        "concept_count": len(concept_rows),
        "attribute_certainty_weighted": certainty is not None,
        "mean_best_ap": float(np.mean([row["best_ap"] for row in concept_rows])) if concept_rows else 0.0,
        "mean_normalized_ap_gain": float(np.mean([row["normalized_ap_gain"] for row in concept_rows])) if concept_rows else 0.0,
        "concept_coverage_ap_0_5": float(np.mean([row["best_ap"] >= 0.5 for row in concept_rows])) if concept_rows else 0.0,
        "concepts": concept_rows,
        "feature_best_concept": feature_best_concept.tolist(),
        "feature_best_ap": feature_best_ap.tolist(),
    }


def heldout_binary_coalition_metrics(
    image_codes: np.ndarray,
    attributes: np.ndarray,
    feature_ids: np.ndarray,
    attribute_names: list[str],
    seed: int,
    max_features_per_concept: int = 6,
    minimum_positives: int = 5,
    train_fraction: float = 0.7,
) -> dict[str, Any]:
    image_codes = np.asarray(image_codes, dtype=np.float32)
    attributes = np.asarray(attributes, dtype=np.int64)
    active = image_codes > 0
    generator = np.random.default_rng(seed)
    rows = []
    for concept in progress(range(attributes.shape[1]), desc="concept coalitions"):
        targets = attributes[:, concept].astype(bool)
        positive = np.flatnonzero(targets)
        negative = np.flatnonzero(~targets)
        if len(positive) < minimum_positives or len(negative) < minimum_positives:
            continue
        generator.shuffle(positive)
        generator.shuffle(negative)
        positive_cut = min(max(1, int(round(len(positive) * train_fraction))), len(positive) - 1)
        negative_cut = min(max(1, int(round(len(negative) * train_fraction))), len(negative) - 1)
        train_indices = np.concatenate([positive[:positive_cut], negative[:negative_cut]])
        test_indices = np.concatenate([positive[positive_cut:], negative[negative_cut:]])
        selected = []
        train_prediction = np.zeros(len(train_indices), dtype=bool)
        best_train_f1 = 0.0
        for _ in range(max_features_per_concept):
            candidate_feature = None
            candidate_prediction = None
            candidate_f1 = best_train_f1
            for local_feature in range(active.shape[1]):
                if local_feature in selected:
                    continue
                prediction = train_prediction | active[train_indices, local_feature]
                value = _binary_f1(targets[train_indices], prediction)
                if value > candidate_f1 + 1e-12:
                    candidate_f1 = value
                    candidate_feature = local_feature
                    candidate_prediction = prediction
            if candidate_feature is None or candidate_prediction is None:
                break
            selected.append(candidate_feature)
            train_prediction = candidate_prediction
            best_train_f1 = candidate_f1
        if selected:
            test_prediction = active[test_indices][:, selected].any(axis=1)
        else:
            test_prediction = np.zeros(len(test_indices), dtype=bool)
        test_targets = targets[test_indices]
        rows.append(
            {
                "concept_id": concept,
                "name": attribute_names[concept] if concept < len(attribute_names) else str(concept),
                "selected_features": [int(feature_ids[index]) for index in selected],
                "coalition_size": len(selected),
                "train_f1": best_train_f1,
                "test_f1": _binary_f1(test_targets, test_prediction),
                "test_precision": _binary_precision(test_targets, test_prediction),
                "test_recall": _binary_recall(test_targets, test_prediction),
                "test_prevalence": float(test_targets.mean()),
            }
        )
    rows.sort(key=lambda row: row["test_f1"], reverse=True)
    return {
        "concept_count": len(rows),
        "mean_test_f1": float(np.mean([row["test_f1"] for row in rows])) if rows else 0.0,
        "median_test_f1": float(np.median([row["test_f1"] for row in rows])) if rows else 0.0,
        "coverage_test_f1_0_5": float(np.mean([row["test_f1"] >= 0.5 for row in rows])) if rows else 0.0,
        "concepts": rows,
    }


def _binary_precision(targets: np.ndarray, predictions: np.ndarray) -> float:
    true_positive = int(np.logical_and(targets, predictions).sum())
    predicted_positive = int(predictions.sum())
    return true_positive / predicted_positive if predicted_positive else 0.0


def _binary_recall(targets: np.ndarray, predictions: np.ndarray) -> float:
    true_positive = int(np.logical_and(targets, predictions).sum())
    positive = int(targets.sum())
    return true_positive / positive if positive else 0.0


def _binary_f1(targets: np.ndarray, predictions: np.ndarray) -> float:
    precision = _binary_precision(targets, predictions)
    recall = _binary_recall(targets, predictions)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0
