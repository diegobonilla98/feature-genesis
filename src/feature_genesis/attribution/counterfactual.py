from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from feature_genesis.config import Config
from feature_genesis.data.factory import DatasetBundle, eval_loader
from feature_genesis.hashing import state_dict_hash
from feature_genesis.lineage.alignment import OrthogonalAlignment
from feature_genesis.sae.core import SparseAutoencoder
from feature_genesis.training.replay import reconstruct_state
from feature_genesis.attribution.objectives import feature_score


@dataclass(frozen=True)
class LossMask:
    dataset_indices: frozenset[int]
    steps: frozenset[int] | None = None

    def weights(self, global_step: int, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        indices = batch["dataset_index"]
        if self.steps is not None and global_step not in self.steps:
            return torch.ones_like(indices, dtype=torch.float32)
        mask = torch.ones_like(indices, dtype=torch.float32)
        for dataset_index in self.dataset_indices:
            mask = mask * indices.ne(dataset_index)
        return mask


def counterfactual_replay(
    config: Config,
    sae: SparseAutoencoder,
    feature_id: int,
    start_step: int,
    end_step: int,
    mask: LossMask | None = None,
    bundle: DatasetBundle | None = None,
    alignment: OrthogonalAlignment | None = None,
    image_intervention: PartOcclusion | SameClassReplacement | None = None,
) -> dict[str, Any]:
    baseline_model, _, resolved_bundle, _ = reconstruct_state(
        config,
        end_step,
        bundle=bundle,
        force_anchor_step=start_step,
    )
    counterfactual_model, _, _, _ = reconstruct_state(
        config,
        end_step,
        bundle=resolved_bundle,
        force_anchor_step=start_step,
        loss_weight_function=None if mask is None else mask.weights,
        image_function=None if image_intervention is None else image_intervention.images,
    )
    device = next(baseline_model.parameters()).device
    baseline_loader = eval_loader(
        resolved_bundle.validation,
        config,
        config.attribution.probe_images,
        config.data.eval_batch_size,
    )
    counterfactual_loader = eval_loader(
        resolved_bundle.validation,
        config,
        config.attribution.probe_images,
        config.data.eval_batch_size,
    )
    baseline_score = feature_score(
        baseline_model,
        sae,
        baseline_loader,
        config.model.hook_layer,
        feature_id,
        device,
        config.attribution.probe_images,
        alignment,
    )
    counterfactual_score = feature_score(
        counterfactual_model,
        sae,
        counterfactual_loader,
        config.model.hook_layer,
        feature_id,
        device,
        config.attribution.probe_images,
        alignment,
    )
    parameter_l2 = 0.0
    parameter_max = 0.0
    for baseline, counterfactual in zip(baseline_model.parameters(), counterfactual_model.parameters(), strict=True):
        difference = baseline.detach() - counterfactual.detach()
        parameter_l2 += float(difference.float().pow(2).sum().cpu())
        parameter_max = max(parameter_max, float(difference.float().abs().max().cpu()))
    return {
        "start_step": start_step,
        "end_step": end_step,
        "feature_id": feature_id,
        "masked_dataset_indices": [] if mask is None else sorted(mask.dataset_indices),
        "masked_steps": None if mask is None or mask.steps is None else sorted(mask.steps),
        "image_intervention": intervention_metadata(image_intervention),
        "baseline_feature_score": baseline_score,
        "counterfactual_feature_score": counterfactual_score,
        "feature_score_change": counterfactual_score - baseline_score,
        "relative_feature_score_change": (counterfactual_score - baseline_score) / max(abs(baseline_score), 1e-12),
        "baseline_model_sha256": state_dict_hash(baseline_model.state_dict()),
        "counterfactual_model_sha256": state_dict_hash(counterfactual_model.state_dict()),
        "parameter_l2": parameter_l2**0.5,
        "parameter_max_abs": parameter_max,
    }


@dataclass(frozen=True)
class PartOcclusion:
    dataset_indices: frozenset[int]
    part_index: int
    radius: int
    steps: frozenset[int] | None = None
    fill: float = 0.0

    def images(self, global_step: int, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        images = batch["image"].clone()
        if self.steps is not None and global_step not in self.steps:
            return images
        if "parts" not in batch or "part_visible" not in batch:
            raise ValueError("Part occlusion requires an annotation-enabled training dataset")
        for batch_index, dataset_index in enumerate(batch["dataset_index"].tolist()):
            if int(dataset_index) not in self.dataset_indices:
                continue
            if not bool(batch["part_visible"][batch_index, self.part_index]):
                continue
            center = batch["parts"][batch_index, self.part_index]
            x = int(round(float(center[0])))
            y = int(round(float(center[1])))
            x0 = max(0, x - self.radius)
            x1 = min(images.shape[-1], x + self.radius + 1)
            y0 = max(0, y - self.radius)
            y1 = min(images.shape[-2], y + self.radius + 1)
            images[batch_index, :, y0:y1, x0:x1] = self.fill
        return images


class SameClassReplacement:
    def __init__(
        self,
        dataset,
        replacements: dict[int, int],
        steps: frozenset[int] | None = None,
    ):
        self.dataset = dataset
        self.replacements = replacements
        self.steps = steps
        for source, target in replacements.items():
            source_label = int(dataset[source]["label"])
            target_label = int(dataset[target]["label"])
            if source_label != target_label:
                raise ValueError(f"Replacement {source}->{target} changes the class label")

    def images(self, global_step: int, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        from feature_genesis.data.batch_plan import SampleKey

        images = batch["image"].clone()
        if self.steps is not None and global_step not in self.steps:
            return images
        for batch_index, dataset_index in enumerate(batch["dataset_index"].tolist()):
            source = int(dataset_index)
            if source not in self.replacements:
                continue
            epoch = int(batch["epoch"][batch_index])
            replacement = self.dataset[SampleKey(self.replacements[source], epoch)]["image"].to(images)
            images[batch_index] = replacement
        return images


def intervention_metadata(intervention: PartOcclusion | SameClassReplacement | None) -> dict[str, Any] | None:
    if intervention is None:
        return None
    if isinstance(intervention, PartOcclusion):
        return {
            "kind": "part_occlusion",
            "dataset_indices": sorted(intervention.dataset_indices),
            "part_index": intervention.part_index,
            "radius": intervention.radius,
            "steps": None if intervention.steps is None else sorted(intervention.steps),
            "fill": intervention.fill,
        }
    return {
        "kind": "same_class_replacement",
        "replacements": {str(source): target for source, target in sorted(intervention.replacements.items())},
        "steps": None if intervention.steps is None else sorted(intervention.steps),
    }
