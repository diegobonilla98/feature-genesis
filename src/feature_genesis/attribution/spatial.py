from __future__ import annotations

from typing import Any

import numpy as np
import torch

from feature_genesis.activations import ActivationCache
from feature_genesis.progress import progress
from feature_genesis.sae.core import SparseAutoencoder


@torch.no_grad()
def spatial_feature_evidence(
    sae: SparseAutoencoder,
    cache: ActivationCache,
    feature_id: int,
    device: torch.device,
    image_size: int,
    part_radius_fraction: float,
    top_images: int,
    batch_size: int = 8192,
) -> dict[str, Any]:
    image_count = len(cache.image_offsets) - 1
    height = int(cache.metadata["spatial_height"])
    width = int(cache.metadata["spatial_width"])
    maxima = np.zeros(image_count, dtype=np.float32)
    locations = np.full((image_count, 2), np.nan, dtype=np.float32)
    nearest_parts = np.full(image_count, -1, dtype=np.int32)
    normalized_distances = np.full(image_count, np.nan, dtype=np.float32)
    tokens_per_image = max(1, int(cache.metadata["tokens_per_image"]))
    images_per_group = max(1, batch_size // tokens_per_image)
    for image_start in progress(
        range(0, image_count, images_per_group),
        desc="spatial evidence",
    ):
        image_end = min(image_count, image_start + images_per_group)
        row_start = int(cache.image_offsets[image_start])
        row_end = int(cache.image_offsets[image_end])
        inputs = torch.from_numpy(
            np.array(cache.activations[row_start:row_end], dtype=np.float32, copy=True)
        ).to(device)
        values = sae.encode(inputs, inference=True)[:, feature_id]
        for image_index in range(image_start, image_end):
            local_start = int(cache.image_offsets[image_index]) - row_start
            local_end = int(cache.image_offsets[image_index + 1]) - row_start
            local_values = values[local_start:local_end]
            local_index = int(local_values.argmax().cpu())
            maxima[image_index] = float(local_values[local_index].cpu())
            row_index = int(cache.image_offsets[image_index]) + local_index
            grid_y = int(cache.spatial_y[row_index])
            grid_x = int(cache.spatial_x[row_index])
            if grid_y >= 0 and grid_x >= 0:
                x = (grid_x + 0.5) * image_size / width
                y = (grid_y + 0.5) * image_size / height
                locations[image_index] = (x, y)
                visible = np.asarray(cache.image_part_visible[image_index], dtype=bool)
                parts = np.asarray(cache.image_parts[image_index], dtype=np.float32)
                valid_indices = np.flatnonzero(
                    visible & np.isfinite(parts).all(axis=1)
                )
                if len(valid_indices):
                    distances = np.linalg.norm(
                        parts[valid_indices] - locations[image_index],
                        axis=1,
                    )
                    best = int(distances.argmin())
                    nearest_parts[image_index] = int(valid_indices[best])
                    normalized_distances[image_index] = float(
                        distances[best] / image_size
                    )
    radius = part_radius_fraction
    active = maxima > 0
    active_indices = np.flatnonzero(active)
    order = active_indices[np.argsort(maxima[active_indices])[::-1]][: min(top_images, len(active_indices))]
    localized = active & np.isfinite(normalized_distances)
    hit_rate = float((normalized_distances[localized] <= radius).mean()) if localized.any() else 0.0
    attributes = np.asarray(cache.image_attributes, dtype=np.float32)
    scale = max(maxima.std(), 1e-6)
    if cache.image_attribute_certainty is None:
        certainty = np.ones_like(attributes, dtype=np.float32)
        certainty_available = False
    else:
        certainty = np.asarray(
            cache.image_attribute_certainty,
            dtype=np.float32,
        ) / 4.0
        certainty_available = True
    positive_weights = attributes * certainty
    negative_weights = (1 - attributes) * certainty
    positive_count = positive_weights.sum(axis=0).clip(min=1)
    negative_count = negative_weights.sum(axis=0).clip(min=1)
    attribute_effect = (
        positive_weights.T @ maxima / positive_count
        - negative_weights.T @ maxima / negative_count
    ) / scale
    top_attributes = np.argsort(attribute_effect)[::-1][: min(20, len(attribute_effect))]
    active_parts = nearest_parts[active & (nearest_parts >= 0)]
    part_counts = np.bincount(active_parts, minlength=cache.image_parts.shape[1])
    top_parts = np.argsort(part_counts)[::-1][: min(10, len(part_counts))]
    image_ids = np.asarray(cache.image_ids[cache.image_offsets[:-1]], dtype=np.int64)
    dataset_indices = np.asarray(cache.dataset_indices[cache.image_offsets[:-1]], dtype=np.int64)
    labels = np.asarray(cache.labels[cache.image_offsets[:-1]], dtype=np.int64)
    return {
        "feature_id": feature_id,
        "activation_frequency_per_image": float(active.mean()),
        "active_images": int(active.sum()),
        "localized_active_images": int(localized.sum()),
        "attribute_certainty_available": certainty_available,
        "part_hit_rate": hit_rate,
        "median_nearest_part_distance": float(np.nanmedian(normalized_distances[localized])) if localized.any() else None,
        "top_images": [
            {
                "image_index": int(index),
                "image_id": int(image_ids[index]),
                "dataset_index": int(dataset_indices[index]),
                "label": int(labels[index]),
                "activation": float(maxima[index]),
                "location": locations[index].tolist(),
                "nearest_part": int(nearest_parts[index]),
                "normalized_part_distance": None if not np.isfinite(normalized_distances[index]) else float(normalized_distances[index]),
            }
            for index in order
        ],
        "top_attributes": [
            {
                "attribute_id": int(index),
                "name": cache.metadata.get("attribute_names", [str(value) for value in range(len(attribute_effect))])[index],
                "standardized_effect": float(attribute_effect[index]),
            }
            for index in top_attributes
        ],
        "top_parts": [
            {
                "part_id": int(index),
                "name": cache.metadata.get("part_names", [str(value) for value in range(len(part_counts))])[index],
                "count": int(part_counts[index]),
                "share_of_localized_active_images": float(part_counts[index] / max(1, localized.sum())),
            }
            for index in top_parts
        ],
    }


def occlude_square(images: torch.Tensor, centers: torch.Tensor, radius: int, fill: float = 0.0) -> torch.Tensor:
    output = images.clone()
    height, width = output.shape[-2:]
    for index, center in enumerate(centers):
        x = int(round(float(center[0])))
        y = int(round(float(center[1])))
        x0 = max(0, x - radius)
        x1 = min(width, x + radius + 1)
        y0 = max(0, y - radius)
        y1 = min(height, y + radius + 1)
        output[index, :, y0:y1, x0:x1] = fill
    return output
