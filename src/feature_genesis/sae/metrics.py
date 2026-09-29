from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from feature_genesis.activations import ActivationCache
from feature_genesis.progress import progress
from feature_genesis.sae.core import SparseAutoencoder


@torch.no_grad()
def evaluate_sae(
    sae: SparseAutoencoder,
    cache: ActivationCache,
    batch_size: int,
    device: torch.device,
    max_rows: int | None = None,
    row_indices: np.ndarray | None = None,
    include_feature_metrics: bool = False,
) -> dict[str, Any]:
    indices = _resolved_indices(len(cache.activations), max_rows, row_indices)
    count = len(indices)
    if count == 0:
        raise ValueError("SAE evaluation requires at least one row")
    squared_error = 0.0
    squared_signal = 0.0
    l0_total = 0.0
    active_counts = torch.zeros(sae.dictionary_size, dtype=torch.long, device=device)
    activation_sum = torch.zeros(sae.dictionary_size, dtype=torch.float64, device=device)
    activation_max = torch.zeros(sae.dictionary_size, dtype=torch.float32, device=device)
    image_active_counts = torch.zeros(sae.dictionary_size, dtype=torch.long, device=device)
    image_sample_count = 0
    image_metrics_valid = True
    tokens_per_image = int(cache.metadata.get("tokens_per_image", 1))
    sample_count = 0
    mean = torch.from_numpy(np.array(cache.activations[indices], dtype=np.float32, copy=True)).mean(dim=0).to(device)
    centered_energy = 0.0
    for start in progress(range(0, count, batch_size), desc="evaluate sae inference"):
        batch_indices = indices[start : start + batch_size]
        inputs = torch.from_numpy(np.array(cache.activations[batch_indices], dtype=np.float32, copy=True)).to(device)
        codes = sae.encode(inputs, inference=True)
        reconstruction = sae.decode(codes)
        residual = reconstruction - inputs
        squared_error += float(residual.float().pow(2).sum().cpu())
        squared_signal += float(inputs.float().pow(2).sum().cpu())
        centered_energy += float((inputs - mean).float().pow(2).sum().cpu())
        active = codes.ne(0)
        l0_total += float(active.sum().cpu())
        active_counts += active.sum(dim=0)
        activation_sum += codes.double().sum(dim=0)
        activation_max = torch.maximum(activation_max, codes.float().amax(dim=0))
        batch_image_ids = np.asarray(cache.image_ids[batch_indices], dtype=np.int64)
        if (
            image_metrics_valid
            and len(batch_indices) % tokens_per_image == 0
            and np.all(
                batch_image_ids.reshape(-1, tokens_per_image)
                == batch_image_ids.reshape(-1, tokens_per_image)[:, :1]
            )
        ):
            image_active_counts += active.reshape(
                -1,
                tokens_per_image,
                sae.dictionary_size,
            ).any(dim=1).sum(dim=0)
            image_sample_count += len(batch_indices) // tokens_per_image
        else:
            image_metrics_valid = False
        sample_count += inputs.shape[0]
    frequency = active_counts.float() / sample_count
    mean_when_active = activation_sum.float() / active_counts.clamp_min(1)
    mse = squared_error / (sample_count * sae.input_dim)
    signal_mse = squared_signal / (sample_count * sae.input_dim)
    directions = sae.decoder_directions().detach()
    coherence = mutual_coherence(directions)
    positive_decoder_cosine = maximum_positive_decoder_cosine(directions)
    metrics = {
        "rows": sample_count,
        "inference_mode": "sample_threshold_or_native_encode",
        "mse": mse,
        "nmse": mse / max(signal_mse, 1e-12),
        "fraction_variance_explained": 1 - squared_error / max(centered_energy, 1e-12),
        "mean_l0": l0_total / sample_count,
        "dead_fraction": float((active_counts == 0).float().mean().cpu()),
        "mutual_coherence": coherence,
        "maximum_positive_decoder_cosine": positive_decoder_cosine,
        "frequency_mean": float(frequency.mean().cpu()),
        "frequency_median": float(frequency.median().cpu()),
        "frequency_max": float(frequency.max().cpu()),
        "activation_mean_when_active": float(mean_when_active[active_counts > 0].mean().cpu()) if (active_counts > 0).any() else 0.0,
        "activation_max": float(activation_max.max().cpu()),
    }
    if image_metrics_valid and image_sample_count > 0:
        image_frequency = image_active_counts.float() / image_sample_count
        metrics.update(
            {
                "images": image_sample_count,
                "image_frequency_mean": float(image_frequency.mean().cpu()),
                "image_frequency_median": float(image_frequency.median().cpu()),
                "image_frequency_max": float(image_frequency.max().cpu()),
            }
        )
    if include_feature_metrics:
        metrics["feature_frequency"] = frequency.cpu().tolist()
        metrics["feature_mean_when_active"] = mean_when_active.cpu().tolist()
        metrics["feature_activation_max"] = activation_max.cpu().tolist()
        if image_metrics_valid and image_sample_count > 0:
            metrics["feature_image_frequency"] = image_frequency.cpu().tolist()
    return metrics


@torch.no_grad()
def evaluate_reconstruction_loss(
    sae: SparseAutoencoder,
    cache: ActivationCache,
    row_indices: np.ndarray,
    batch_size: int,
    device: torch.device,
    inference: bool,
) -> float:
    indices = np.asarray(row_indices, dtype=np.int64)
    if len(indices) == 0:
        raise ValueError("Reconstruction evaluation requires at least one row")
    squared_error = 0.0
    value_count = 0
    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start : start + batch_size]
        inputs = torch.from_numpy(np.array(cache.activations[batch_indices], dtype=np.float32, copy=True)).to(device)
        codes = sae.encode(inputs, inference=inference)
        reconstruction = sae.decode(codes)
        squared_error += float((reconstruction - inputs).float().pow(2).sum().cpu())
        value_count += inputs.numel()
    return squared_error / value_count


def _resolved_indices(
    row_count: int,
    max_rows: int | None,
    row_indices: np.ndarray | None,
) -> np.ndarray:
    if row_indices is None:
        indices = np.arange(row_count, dtype=np.int64)
    else:
        indices = np.asarray(row_indices, dtype=np.int64)
        if len(indices) and (indices.min() < 0 or indices.max() >= row_count):
            raise IndexError("SAE evaluation row index is out of bounds")
    if max_rows is not None:
        indices = indices[:max_rows]
    return indices


@torch.no_grad()
def mutual_coherence(directions: torch.Tensor, block_size: int = 512) -> float:
    directions = F.normalize(directions.float(), dim=-1)
    maximum = directions.new_zeros(())
    count = directions.shape[0]
    for start in range(0, count, block_size):
        block = directions[start : start + block_size]
        similarities = block @ directions.T
        row_indices = torch.arange(start, min(start + block_size, count), device=directions.device)
        local_indices = torch.arange(len(row_indices), device=directions.device)
        similarities[local_indices, row_indices] = 0
        maximum = torch.maximum(maximum, similarities.abs().max())
    return float(maximum.cpu())


@torch.no_grad()
def maximum_positive_decoder_cosine(
    directions: torch.Tensor,
    block_size: int = 512,
) -> float:
    directions = F.normalize(directions.float(), dim=-1)
    maximum = directions.new_zeros(())
    count = directions.shape[0]
    for start in range(0, count, block_size):
        block = directions[start : start + block_size]
        similarities = block @ directions.T
        row_indices = torch.arange(start, min(start + block_size, count), device=directions.device)
        local_indices = torch.arange(len(row_indices), device=directions.device)
        similarities[local_indices, row_indices] = 0
        maximum = torch.maximum(maximum, similarities.max())
    return float(maximum.cpu())
