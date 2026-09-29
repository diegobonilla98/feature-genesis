from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from feature_genesis.models.hooks import ActivationModifier, rows_to_spatial, spatial_to_rows
from feature_genesis.progress import progress
from feature_genesis.sae.core import SparseAutoencoder


@torch.no_grad()
def evaluate_sae_substitution(
    model: nn.Module,
    sae: SparseAutoencoder,
    loader,
    layer: str,
    device: torch.device,
    max_images: int | None = None,
) -> dict[str, Any]:
    model.eval()
    sae.eval()
    baseline_loss = 0.0
    reconstructed_loss = 0.0
    zero_loss = 0.0
    baseline_correct = 0
    reconstructed_correct = 0
    zero_correct = 0
    kl_total = 0.0
    logit_mse_total = 0.0
    count = 0

    def reconstruct_activation(activation: torch.Tensor) -> torch.Tensor:
        rows, shape = spatial_to_rows(activation)
        codes = sae.encode(rows, inference=True)
        reconstructed = sae.decode(codes).to(rows.dtype)
        return rows_to_spatial(reconstructed, shape)

    for batch in progress(loader, desc="sae fidelity"):
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        if max_images is not None:
            remaining = max_images - count
            if remaining <= 0:
                break
            images = images[:remaining]
            labels = labels[:remaining]
        baseline_logits = model(images)
        with ActivationModifier(model, layer, reconstruct_activation):
            reconstructed_logits = model(images)
        with ActivationModifier(model, layer, torch.zeros_like):
            zero_logits = model(images)
        baseline_loss += float(F.cross_entropy(baseline_logits, labels, reduction="sum").cpu())
        reconstructed_loss += float(F.cross_entropy(reconstructed_logits, labels, reduction="sum").cpu())
        zero_loss += float(F.cross_entropy(zero_logits, labels, reduction="sum").cpu())
        baseline_correct += int((baseline_logits.argmax(dim=-1) == labels).sum().cpu())
        reconstructed_correct += int((reconstructed_logits.argmax(dim=-1) == labels).sum().cpu())
        zero_correct += int((zero_logits.argmax(dim=-1) == labels).sum().cpu())
        log_probabilities = F.log_softmax(reconstructed_logits.float(), dim=-1)
        probabilities = F.softmax(baseline_logits.float(), dim=-1)
        kl_total += float(F.kl_div(log_probabilities, probabilities, reduction="sum").cpu())
        logit_mse_total += float((baseline_logits.float() - reconstructed_logits.float()).pow(2).sum().cpu())
        count += labels.numel()
    if count == 0:
        raise ValueError("No samples were evaluated")
    baseline_loss /= count
    reconstructed_loss /= count
    zero_loss /= count
    denominator = max(zero_loss - baseline_loss, 1e-12)
    return {
        "images": count,
        "baseline_loss": baseline_loss,
        "reconstructed_loss": reconstructed_loss,
        "zero_ablation_loss": zero_loss,
        "delta_task_loss": reconstructed_loss - baseline_loss,
        "loss_recovered": 1 - (reconstructed_loss - baseline_loss) / denominator,
        "baseline_accuracy": baseline_correct / count,
        "reconstructed_accuracy": reconstructed_correct / count,
        "zero_ablation_accuracy": zero_correct / count,
        "accuracy_change": (reconstructed_correct - baseline_correct) / count,
        "mean_kl_baseline_to_reconstruction": kl_total / count,
        "mean_logit_mse": logit_mse_total / count,
    }
