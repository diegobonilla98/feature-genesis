from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from feature_genesis.models.hooks import ActivationModifier
from feature_genesis.models.interventions import ablate_sae_feature, steer_sae_feature
from feature_genesis.progress import progress
from feature_genesis.sae.core import SparseAutoencoder


@torch.no_grad()
def evaluate_feature_intervention(
    model: nn.Module,
    sae: SparseAutoencoder,
    loader,
    layer: str,
    feature_id: int,
    device: torch.device,
    mode: str,
    amount: float,
    max_images: int | None = None,
) -> dict[str, Any]:
    model.eval()
    sae.eval()
    baseline_loss = 0.0
    intervened_loss = 0.0
    baseline_correct = 0
    intervened_correct = 0
    flips = 0
    logit_l2 = 0.0
    probability_shift = None
    count = 0

    def modify(activation: torch.Tensor) -> torch.Tensor:
        if mode == "ablate":
            return ablate_sae_feature(activation, sae, feature_id, amount)
        if mode == "steer":
            return steer_sae_feature(activation, sae, feature_id, amount)
        raise ValueError(f"Unknown intervention mode: {mode}")

    for batch in progress(loader, desc=f"intervention {mode}"):
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        if max_images is not None:
            remaining = max_images - count
            if remaining <= 0:
                break
            images = images[:remaining]
            labels = labels[:remaining]
        baseline_logits = model(images)
        with ActivationModifier(model, layer, modify):
            intervened_logits = model(images)
        baseline_loss += float(F.cross_entropy(baseline_logits, labels, reduction="sum").cpu())
        intervened_loss += float(F.cross_entropy(intervened_logits, labels, reduction="sum").cpu())
        baseline_predictions = baseline_logits.argmax(dim=-1)
        intervened_predictions = intervened_logits.argmax(dim=-1)
        baseline_correct += int((baseline_predictions == labels).sum().cpu())
        intervened_correct += int((intervened_predictions == labels).sum().cpu())
        flips += int((baseline_predictions != intervened_predictions).sum().cpu())
        logit_l2 += float((baseline_logits.float() - intervened_logits.float()).pow(2).sum(dim=-1).sqrt().sum().cpu())
        shift = F.softmax(intervened_logits.float(), dim=-1) - F.softmax(baseline_logits.float(), dim=-1)
        current = shift.sum(dim=0).cpu()
        probability_shift = current if probability_shift is None else probability_shift + current
        count += labels.numel()
    if count == 0 or probability_shift is None:
        raise ValueError("No samples were evaluated")
    mean_shift = probability_shift / count
    positive = torch.topk(mean_shift, min(10, len(mean_shift)))
    negative = torch.topk(-mean_shift, min(10, len(mean_shift)))
    return {
        "feature_id": feature_id,
        "mode": mode,
        "amount": amount,
        "images": count,
        "baseline_loss": baseline_loss / count,
        "intervened_loss": intervened_loss / count,
        "delta_task_loss": (intervened_loss - baseline_loss) / count,
        "baseline_accuracy": baseline_correct / count,
        "intervened_accuracy": intervened_correct / count,
        "prediction_flip_rate": flips / count,
        "mean_logit_l2": logit_l2 / count,
        "largest_probability_increases": [
            {"class_id": int(index), "mean_shift": float(value)}
            for value, index in zip(positive.values, positive.indices, strict=True)
        ],
        "largest_probability_decreases": [
            {"class_id": int(index), "mean_shift": float(-value)}
            for value, index in zip(negative.values, negative.indices, strict=True)
        ],
    }
