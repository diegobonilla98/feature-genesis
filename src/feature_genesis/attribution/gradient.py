from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.func import functional_call

from feature_genesis.attribution.objectives import feature_objective
from feature_genesis.config import Config
from feature_genesis.data.batch_plan import ExactBatchPlan
from feature_genesis.data.factory import DatasetBundle, eval_loader, exact_train_loader
from feature_genesis.lineage.alignment import OrthogonalAlignment
from feature_genesis.models.factory import parameter_scope
from feature_genesis.progress import progress
from feature_genesis.sae.core import SparseAutoencoder
from feature_genesis.training.schedule import cosine_learning_rate
from feature_genesis.training.trainer import autocast_context


@dataclass
class BatchInfluence:
    step: int
    influence: float
    cosine: float
    learning_rate: float
    feature_gradient_norm: float
    batch_gradient_norm: float
    dataset_indices: list[int]
    labels: list[int]
    example_builders: list[dict[str, Any]]
    example_breakers: list[dict[str, Any]]


def rank_candidate_batches_multi(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    sae: SparseAutoencoder,
    bundle: DatasetBundle,
    plan: ExactBatchPlan,
    config: Config,
    feature_ids: list[int],
    start_step: int,
    end_step: int,
    alignment: OrthogonalAlignment | None = None,
) -> dict[int, list[BatchInfluence]]:
    device = next(model.parameters()).device
    names_and_parameters = parameter_scope(
        model,
        config.model.hook_layer,
        config.attribution.parameter_scope,
    )
    parameters = [parameter for _, parameter in names_and_parameters]
    if not parameters:
        raise ValueError("The selected parameter scope is empty")
    probe_loader = eval_loader(
        bundle.validation,
        config,
        config.attribution.probe_images,
        config.data.eval_batch_size,
    )
    loader = exact_train_loader(bundle.train, plan, config, start_step, end_step)
    feature_gradients = {}
    feature_norms = {}
    output = {feature_id: [] for feature_id in feature_ids}
    refresh = max(1, config.attribution.gradient_refresh_steps)
    step_range = range(start_step, end_step)
    for step, batch in progress(
        zip(step_range, loader, strict=True),
        total=end_step - start_step,
        desc=f"rank batches for {len(feature_ids)} features",
    ):
        if not feature_gradients or (step - start_step) % refresh == 0:
            for feature_id in feature_ids:
                model.zero_grad(set_to_none=True)
                objective = feature_objective(
                    model,
                    sae,
                    probe_loader,
                    config.model.hook_layer,
                    feature_id,
                    device,
                    config.attribution.probe_images,
                    alignment,
                    True,
                )
                values = torch.autograd.grad(
                    objective,
                    parameters,
                    retain_graph=False,
                    allow_unused=True,
                )
                gradients = tuple(
                    None if value is None else value.detach().clone()
                    for value in values
                )
                feature_gradients[feature_id] = gradients
                feature_norms[feature_id] = gradient_norm(gradients)
        learning_rate = cosine_learning_rate(
            step,
            len(plan),
            config.train.learning_rate,
            config.train.min_learning_rate,
            config.train.warmup_steps,
        )
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        model.train()
        optimizer.zero_grad(set_to_none=True)
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        if config.train.channels_last and images.ndim == 4:
            images = images.contiguous(memory_format=torch.channels_last)
        with autocast_context(config, device):
            losses = F.cross_entropy(
                model(images),
                labels,
                reduction="none",
                label_smoothing=config.train.label_smoothing,
            )
        loss = losses.sum() / losses.numel()
        loss.backward()
        gradient_scale = 1.0
        if config.train.gradient_clip_norm is not None:
            total_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                config.train.gradient_clip_norm,
            )
            gradient_scale = min(
                1.0,
                config.train.gradient_clip_norm / max(float(total_norm), 1e-12),
            )
        gradients = tuple(parameter.grad for parameter in parameters)
        current_norm = gradient_norm(gradients)
        example_templates = loss_weighted_example_templates(
            losses,
            batch["dataset_index"],
            batch["label"],
            batch["epoch"],
            config.attribution.per_example_candidates,
        )
        for feature_id in feature_ids:
            feature_gradient = feature_gradients[feature_id]
            feature_norm = feature_norms[feature_id]
            dot = gradient_dot(feature_gradient, gradients)
            cosine = dot / max(feature_norm * current_norm, 1e-12)
            influence = float(-learning_rate * dot)
            example_builders, example_breakers = scaled_example_proxies(
                example_templates,
                influence,
            )
            example_count = config.attribution.per_example_candidates
            output[feature_id].append(
                BatchInfluence(
                    step=step,
                    influence=influence,
                    cosine=float(-cosine),
                    learning_rate=float(learning_rate),
                    feature_gradient_norm=float(feature_norm),
                    batch_gradient_norm=float(current_norm),
                    dataset_indices=[],
                    labels=[],
                    example_builders=example_builders[:example_count],
                    example_breakers=example_breakers[:example_count],
                )
            )
        optimizer.step()
    return {
        feature_id: sorted(rows, key=lambda item: item.influence, reverse=True)
        for feature_id, rows in output.items()
    }


def loss_weighted_example_templates(
    losses,
    dataset_indices,
    labels,
    epochs,
    count,
):
    weights = losses.detach().float().clamp_min(0)
    weights = weights / weights.sum().clamp_min(1e-12)
    count = min(count, len(weights))
    high_indices = torch.topk(weights, count).indices
    low_indices = torch.topk(-weights, count).indices

    def rows(indices):
        indices_list = indices.tolist()
        selected_weights = weights[indices].cpu().tolist()
        return [
            {
                "dataset_index": int(dataset_indices[index]),
                "label": int(labels[index]),
                "epoch": int(epochs[index]),
                "weight": float(weight),
            }
            for index, weight in zip(indices_list, selected_weights, strict=True)
        ]

    return rows(high_indices), rows(low_indices)


def scaled_example_proxies(templates, batch_influence):
    high, low = templates

    def scale(rows):
        return [
            {
                "dataset_index": row["dataset_index"],
                "label": row["label"],
                "epoch": row["epoch"],
                "influence": row["weight"] * batch_influence,
            }
            for row in rows
        ]

    if batch_influence >= 0:
        return scale(high), scale(low)
    return scale(low), scale(high)


def rank_candidate_batches(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    sae: SparseAutoencoder,
    bundle: DatasetBundle,
    plan: ExactBatchPlan,
    config: Config,
    feature_id: int,
    start_step: int,
    end_step: int,
    alignment: OrthogonalAlignment | None = None,
) -> list[BatchInfluence]:
    device = next(model.parameters()).device
    names_and_parameters = parameter_scope(model, config.model.hook_layer, config.attribution.parameter_scope)
    parameters = [parameter for _, parameter in names_and_parameters]
    if not parameters:
        raise ValueError("The selected parameter scope is empty")
    initial_model_state = {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }
    initial_optimizer_state = copy.deepcopy(optimizer.state_dict())
    probe_loader = eval_loader(bundle.validation, config, config.attribution.probe_images, config.data.eval_batch_size)
    loader = exact_train_loader(bundle.train, plan, config, start_step, end_step)
    feature_gradients: tuple[torch.Tensor | None, ...] | None = None
    feature_norm = 0.0
    output = []
    refresh = max(1, config.attribution.gradient_refresh_steps)
    step_range = range(start_step, end_step)
    for step, batch in progress(zip(step_range, loader, strict=True), total=end_step - start_step, desc="rank batches"):
        if feature_gradients is None or (step - start_step) % refresh == 0:
            model.zero_grad(set_to_none=True)
            objective = feature_objective(
                model,
                sae,
                probe_loader,
                config.model.hook_layer,
                feature_id,
                device,
                config.attribution.probe_images,
                alignment,
                True,
            )
            values = torch.autograd.grad(objective, parameters, retain_graph=False, allow_unused=True)
            feature_gradients = tuple(None if value is None else value.detach().clone() for value in values)
            feature_norm = gradient_norm(feature_gradients)
        learning_rate = cosine_learning_rate(
            step,
            len(plan),
            config.train.learning_rate,
            config.train.min_learning_rate,
            config.train.warmup_steps,
        )
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        model.train()
        optimizer.zero_grad(set_to_none=True)
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        if config.train.channels_last and images.ndim == 4:
            images = images.contiguous(memory_format=torch.channels_last)
        with autocast_context(config, device):
            losses = F.cross_entropy(
                model(images),
                labels,
                reduction="none",
                label_smoothing=config.train.label_smoothing,
            )
        loss = losses.sum() / losses.numel()
        loss.backward()
        gradient_scale = 1.0
        if config.train.gradient_clip_norm is not None:
            total_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                config.train.gradient_clip_norm,
            )
            gradient_scale = min(
                1.0,
                config.train.gradient_clip_norm / max(float(total_norm), 1e-12),
            )
        gradients = tuple(parameter.grad for parameter in parameters)
        dot = gradient_dot(feature_gradients, gradients)
        current_norm = gradient_norm(gradients)
        cosine = dot / max(feature_norm * current_norm, 1e-12)
        influence = float(-learning_rate * dot)
        output.append(
            BatchInfluence(
                step=step,
                influence=influence,
                cosine=float(-cosine),
                learning_rate=float(learning_rate),
                feature_gradient_norm=float(feature_norm),
                batch_gradient_norm=float(current_norm),
                dataset_indices=[int(value) for value in batch["dataset_index"].tolist()],
                labels=[int(value) for value in batch["label"].tolist()],
                example_builders=[],
                example_breakers=[],
            )
        )
        optimizer.step()
    ranked = sorted(output, key=lambda item: item.influence, reverse=True)
    candidate_count = config.attribution.candidate_batches
    selected = ranked[:candidate_count] + ranked[-candidate_count:]
    enrich_candidate_examples(
        model,
        optimizer,
        sae,
        bundle,
        plan,
        config,
        feature_id,
        start_step,
        end_step,
        alignment,
        initial_model_state,
        initial_optimizer_state,
        selected,
    )
    return ranked


def enrich_candidate_examples(
    model,
    optimizer,
    sae,
    bundle,
    plan,
    config,
    feature_id,
    start_step,
    end_step,
    alignment,
    initial_model_state,
    initial_optimizer_state,
    selected,
):
    model.load_state_dict(initial_model_state)
    optimizer.load_state_dict(initial_optimizer_state)
    device = next(model.parameters()).device
    names_and_parameters = parameter_scope(
        model,
        config.model.hook_layer,
        config.attribution.parameter_scope,
    )
    parameters = [parameter for _, parameter in names_and_parameters]
    probe_loader = eval_loader(
        bundle.validation,
        config,
        config.attribution.probe_images,
        config.data.eval_batch_size,
    )
    loader = exact_train_loader(bundle.train, plan, config, start_step, end_step)
    selected_by_step = {row.step: row for row in selected}
    feature_gradients = None
    feature_norm = 0.0
    refresh = max(1, config.attribution.gradient_refresh_steps)
    step_range = range(start_step, end_step)
    for step, batch in progress(
        zip(step_range, loader, strict=True),
        total=end_step - start_step,
        desc="enrich candidate examples",
    ):
        if feature_gradients is None or (step - start_step) % refresh == 0:
            model.zero_grad(set_to_none=True)
            objective = feature_objective(
                model,
                sae,
                probe_loader,
                config.model.hook_layer,
                feature_id,
                device,
                config.attribution.probe_images,
                alignment,
                True,
            )
            values = torch.autograd.grad(
                objective,
                parameters,
                retain_graph=False,
                allow_unused=True,
            )
            feature_gradients = tuple(
                None if value is None else value.detach().clone()
                for value in values
            )
            feature_norm = gradient_norm(feature_gradients)
        learning_rate = cosine_learning_rate(
            step,
            len(plan),
            config.train.learning_rate,
            config.train.min_learning_rate,
            config.train.warmup_steps,
        )
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        model.train()
        optimizer.zero_grad(set_to_none=True)
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        if config.train.channels_last and images.ndim == 4:
            images = images.contiguous(memory_format=torch.channels_last)
        with autocast_context(config, device):
            losses = F.cross_entropy(
                model(images),
                labels,
                reduction="none",
                label_smoothing=config.train.label_smoothing,
            )
        loss = losses.sum() / losses.numel()
        loss.backward()
        gradient_scale = 1.0
        if config.train.gradient_clip_norm is not None:
            total_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                config.train.gradient_clip_norm,
            )
            gradient_scale = min(
                1.0,
                config.train.gradient_clip_norm / max(float(total_norm), 1e-12),
            )
        selected_row = selected_by_step.get(step)
        if selected_row is not None:
            example_values = directional_example_influences(
                model,
                names_and_parameters,
                feature_gradients,
                feature_norm,
                images,
                labels,
                batch["dataset_index"],
                batch["epoch"],
                learning_rate,
                config.train.label_smoothing,
                config.attribution.directional_epsilon,
                gradient_scale,
            )
            example_count = config.attribution.per_example_candidates
            selected_row.example_builders = sorted(
                example_values,
                key=lambda item: item["influence"],
                reverse=True,
            )[:example_count]
            selected_row.example_breakers = sorted(
                example_values,
                key=lambda item: item["influence"],
            )[:example_count]
        optimizer.step()


@torch.no_grad()
def directional_example_influences(
    model: nn.Module,
    names_and_parameters: list[tuple[str, nn.Parameter]],
    feature_gradients: tuple[torch.Tensor | None, ...],
    feature_norm: float,
    images: torch.Tensor,
    labels: torch.Tensor,
    dataset_indices: torch.Tensor,
    epochs: torch.Tensor,
    learning_rate: float,
    label_smoothing: float,
    epsilon: float,
    gradient_scale: float,
) -> list[dict[str, Any]]:
    if feature_norm <= 1e-12:
        return []
    base_parameters = dict(model.named_parameters())
    buffers = dict(model.named_buffers())
    directions = {
        name: gradient / feature_norm
        for (name, _), gradient in zip(
            names_and_parameters,
            feature_gradients,
            strict=True,
        )
        if gradient is not None
    }
    plus = {
        name: parameter + epsilon * directions[name]
        if name in directions
        else parameter
        for name, parameter in base_parameters.items()
    }
    minus = {
        name: parameter - epsilon * directions[name]
        if name in directions
        else parameter
        for name, parameter in base_parameters.items()
    }
    plus_losses = F.cross_entropy(
        functional_call(model, (plus, buffers), (images,)),
        labels,
        reduction="none",
        label_smoothing=label_smoothing,
    )
    minus_losses = F.cross_entropy(
        functional_call(model, (minus, buffers), (images,)),
        labels,
        reduction="none",
        label_smoothing=label_smoothing,
    )
    directional_derivative = (plus_losses - minus_losses) / (2 * epsilon)
    raw_dot = directional_derivative * feature_norm
    influence = -learning_rate * raw_dot * gradient_scale / max(1, len(images))
    return [
        {
            "dataset_index": int(dataset_index),
            "label": int(label),
            "epoch": int(epoch),
            "influence": float(value),
        }
        for dataset_index, label, epoch, value in zip(
            dataset_indices.tolist(),
            labels.tolist(),
            epochs.tolist(),
            influence.detach().cpu().tolist(),
            strict=True,
        )
    ]


def gradient_dot(left: tuple[torch.Tensor | None, ...], right: tuple[torch.Tensor | None, ...]) -> float:
    reference = next((value for value in left if value is not None), None)
    if reference is None:
        return 0.0
    value = torch.zeros((), device=reference.device, dtype=torch.float32)
    for left_value, right_value in zip(left, right, strict=True):
        if left_value is not None and right_value is not None:
            value += (left_value.detach().float() * right_value.detach().float()).sum()
    return float(value.cpu())


def gradient_norm(values: tuple[torch.Tensor | None, ...]) -> float:
    reference = next((value for value in values if value is not None), None)
    if reference is None:
        return 0.0
    squared = torch.zeros((), device=reference.device, dtype=torch.float32)
    for value in values:
        if value is not None:
            squared += value.detach().float().pow(2).sum()
    return float(squared.sqrt().cpu())


def influences_to_json(values: list[BatchInfluence]) -> list[dict[str, Any]]:
    return [value.__dict__ for value in values]
