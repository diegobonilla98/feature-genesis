from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from feature_genesis.activations import ActivationCache


@dataclass
class OrthogonalAlignment:
    rotation: np.ndarray
    source_mean: np.ndarray
    target_mean: np.ndarray
    residual_ratio: float
    heldout_residual_ratio: float | None = None

    def transform_rows(self, rows: np.ndarray) -> np.ndarray:
        return (rows - self.source_mean) @ self.rotation + self.target_mean

    def transform_directions(self, directions: np.ndarray) -> np.ndarray:
        return directions @ self.rotation


def orthogonal_procrustes(source: np.ndarray, target: np.ndarray, center: bool = True) -> OrthogonalAlignment:
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if source.shape != target.shape:
        raise ValueError(f"Shape mismatch: {source.shape} vs {target.shape}")
    source_mean = source.mean(axis=0, keepdims=True) if center else np.zeros((1, source.shape[1]))
    target_mean = target.mean(axis=0, keepdims=True) if center else np.zeros((1, target.shape[1]))
    source_centered = source - source_mean
    target_centered = target - target_mean
    left, _, right = np.linalg.svd(source_centered.T @ target_centered, full_matrices=False)
    rotation = left @ right
    transformed = source_centered @ rotation
    numerator = np.square(transformed - target_centered).sum()
    denominator = np.square(target_centered).sum()
    return OrthogonalAlignment(
        rotation=rotation.astype(np.float32),
        source_mean=source_mean.reshape(-1).astype(np.float32),
        target_mean=target_mean.reshape(-1).astype(np.float32),
        residual_ratio=float(numerator / max(denominator, 1e-12)),
    )


def fit_cache_alignment(
    source: ActivationCache,
    target: ActivationCache,
    max_rows: int,
    seed: int,
) -> OrthogonalAlignment:
    _verify_correspondence(source, target)
    count = min(len(source.activations), len(target.activations), max_rows)
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(min(len(source.activations), len(target.activations)), generator=generator)[:count].numpy()
    source_rows = np.asarray(source.activations[indices], dtype=np.float32)
    target_rows = np.asarray(target.activations[indices], dtype=np.float32)
    if count < 10:
        return orthogonal_procrustes(source_rows, target_rows)
    fit_count = max(1, int(round(count * 0.8)))
    alignment = orthogonal_procrustes(
        source_rows[:fit_count],
        target_rows[:fit_count],
    )
    transformed = alignment.transform_rows(source_rows[fit_count:])
    target_centered = target_rows[fit_count:] - alignment.target_mean
    transformed_centered = transformed - alignment.target_mean
    numerator = np.square(transformed_centered - target_centered).sum()
    denominator = np.square(target_centered).sum()
    alignment.heldout_residual_ratio = float(
        numerator / max(denominator, 1e-12)
    )
    return alignment


def _verify_correspondence(source: ActivationCache, target: ActivationCache) -> None:
    count = min(len(source.image_ids), len(target.image_ids))
    if len(source.image_ids) != len(target.image_ids):
        raise ValueError("Activation caches have different row counts")
    fields = [
        (source.image_ids[:count], target.image_ids[:count], "image_ids"),
        (source.spatial_y[:count], target.spatial_y[:count], "spatial_y"),
        (source.spatial_x[:count], target.spatial_x[:count], "spatial_x"),
    ]
    for left, right, name in fields:
        if not np.array_equal(left, right):
            raise ValueError(f"Activation cache correspondence failed for {name}")
