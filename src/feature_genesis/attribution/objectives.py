from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from feature_genesis.lineage.alignment import OrthogonalAlignment
from feature_genesis.models.hooks import ActivationCapture, spatial_to_rows
from feature_genesis.progress import progress
from feature_genesis.sae.core import SparseAutoencoder


def feature_objective(
    model: nn.Module,
    sae: SparseAutoencoder,
    batches: Iterable[dict[str, torch.Tensor]],
    layer: str,
    feature_id: int,
    device: torch.device,
    max_images: int,
    alignment: OrthogonalAlignment | None = None,
    differentiable: bool = True,
) -> torch.Tensor:
    model.eval()
    sae.eval()
    values = []
    seen = 0
    with ActivationCapture(model, layer, detach=not differentiable) as capture:
        for batch in progress(batches, desc="feature objective"):
            images = batch["image"].to(device)
            remaining = max_images - seen
            if remaining <= 0:
                break
            images = images[:remaining]
            model(images)
            if capture.value is None:
                raise RuntimeError("Activation hook did not fire")
            rows, shape = spatial_to_rows(capture.value)
            rows = align_rows_torch(rows, alignment)
            preactivation = sae.preactivations(rows)[:, feature_id]
            sparse_values = sae.encode(rows, inference=True)[:, feature_id]
            if differentiable:
                feature_values = F.relu(preactivation) * sparse_values.gt(0).detach()
            else:
                feature_values = sparse_values
            if len(shape) == 4:
                batch_size, _, height, width = shape
                feature_values = feature_values.reshape(batch_size, height * width).amax(dim=-1)
            values.append(feature_values)
            seen += images.shape[0]
    if not values:
        raise ValueError("No probe images were evaluated")
    return torch.cat(values).mean()


@torch.no_grad()
def feature_score(
    model: nn.Module,
    sae: SparseAutoencoder,
    batches: Iterable[dict[str, torch.Tensor]],
    layer: str,
    feature_id: int,
    device: torch.device,
    max_images: int,
    alignment: OrthogonalAlignment | None = None,
) -> float:
    value = feature_objective(model, sae, batches, layer, feature_id, device, max_images, alignment, False)
    return float(value.cpu())


def align_rows_torch(rows: torch.Tensor, alignment: OrthogonalAlignment | None) -> torch.Tensor:
    if alignment is None:
        return rows
    source_mean = torch.from_numpy(alignment.source_mean).to(rows)
    target_mean = torch.from_numpy(alignment.target_mean).to(rows)
    rotation = torch.from_numpy(alignment.rotation).to(rows)
    return (rows - source_mean) @ rotation + target_mean
