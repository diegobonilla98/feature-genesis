from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from feature_genesis.attribution.counterfactual import LossMask
from feature_genesis.attribution.gradient import (
    gradient_dot,
    gradient_norm,
    loss_weighted_example_templates,
    scaled_example_proxies,
)
from feature_genesis.circuits.discovery import CircuitProbeSet
from feature_genesis.circuits.sparse import centered_logit, modify_sparse_features
from feature_genesis.data.factory import exact_train_loader
from feature_genesis.hashing import state_dict_hash
from feature_genesis.models.hooks import MultiActivationModifier
from feature_genesis.progress import progress
from feature_genesis.training.replay import reconstruct_state
from feature_genesis.training.schedule import cosine_learning_rate
from feature_genesis.training.trainer import autocast_context


def circuit_objective(model, saes, graph, probes, batch_size, device):
    selected = _selected_features(graph)
    functions = _keep_functions(saes, selected)
    class_id = int(graph["target"]["selected_logit"]["class_id"])
    values = []
    model.eval()
    with MultiActivationModifier(model, functions):
        for batch in probes.batches(batch_size, device):
            logits = model(batch["image"])
            values.append(centered_logit(logits, class_id))
    if not values:
        raise ValueError("Circuit objective requires probe images")
    return torch.cat(values).mean()


@torch.no_grad()
def circuit_score(model, saes, graph, probes, batch_size, device):
    return float(circuit_objective(model, saes, graph, probes, batch_size, device).cpu())


def rank_circuit_training_batches(
    model,
    optimizer,
    saes,
    graph,
    probes,
    bundle,
    plan,
    config,
    start_step,
    end_step,
):
    device = next(model.parameters()).device
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("Circuit attribution parameter scope is empty")
    loader = exact_train_loader(bundle.train, plan, config, start_step, end_step)
    circuit_gradient = None
    circuit_norm = 0.0
    rows = []
    refresh = max(1, config.circuit.attribution_gradient_refresh_steps)
    for step, batch in progress(
        zip(range(start_step, end_step), loader, strict=True),
        total=end_step - start_step,
        desc="rank circuit-building batches",
    ):
        if circuit_gradient is None or (step - start_step) % refresh == 0:
            model.zero_grad(set_to_none=True)
            objective = circuit_objective(
                model,
                saes,
                graph,
                probes,
                config.circuit.discovery_batch_size,
                device,
            )
            gradients = torch.autograd.grad(
                objective,
                parameters,
                retain_graph=False,
                allow_unused=True,
            )
            circuit_gradient = tuple(
                None if value is None else value.detach().clone()
                for value in gradients
            )
            circuit_norm = gradient_norm(circuit_gradient)
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
        if config.train.channels_last:
            images = images.contiguous(memory_format=torch.channels_last)
        with autocast_context(config, device):
            losses = F.cross_entropy(
                model(images),
                labels,
                reduction="none",
                label_smoothing=config.train.label_smoothing,
            )
        losses.mean().backward()
        if config.train.gradient_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.gradient_clip_norm)
        batch_gradient = tuple(parameter.grad for parameter in parameters)
        batch_norm = gradient_norm(batch_gradient)
        dot = gradient_dot(circuit_gradient, batch_gradient)
        influence = float(-learning_rate * dot)
        cosine = float(-dot / max(circuit_norm * batch_norm, 1e-12))
        high, low = loss_weighted_example_templates(
            losses,
            batch["dataset_index"],
            batch["label"],
            batch["epoch"],
            config.attribution.per_example_candidates,
        )
        builders, breakers = scaled_example_proxies((high, low), influence)
        rows.append(
            {
                "step": step,
                "influence": influence,
                "cosine": cosine,
                "learning_rate": float(learning_rate),
                "circuit_gradient_norm": float(circuit_norm),
                "batch_gradient_norm": float(batch_norm),
                "example_builders": builders,
                "example_breakers": breakers,
            }
        )
        optimizer.step()
    ranked = sorted(rows, key=lambda row: row["influence"], reverse=True)
    count = config.circuit.attribution_candidate_batches
    return {
        "method": "future_circuit_gradient_alignment_proxy",
        "start_step": start_step,
        "end_step": end_step,
        "builders": ranked[:count],
        "breakers": sorted(ranked, key=lambda row: row["influence"])[:count],
    }


def counterfactual_circuit_replay(
    config,
    saes,
    graph,
    probes,
    start_step,
    end_step,
    dataset_indices,
    intervention_steps,
    bundle=None,
):
    mask = LossMask(frozenset(dataset_indices), frozenset(intervention_steps))
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
        loss_weight_function=mask.weights,
    )
    device = next(baseline_model.parameters()).device
    baseline_score = circuit_score(
        baseline_model,
        saes,
        graph,
        probes,
        config.circuit.discovery_batch_size,
        device,
    )
    counterfactual_score = circuit_score(
        counterfactual_model,
        saes,
        graph,
        probes,
        config.circuit.discovery_batch_size,
        device,
    )
    parameter_l2 = 0.0
    parameter_max = 0.0
    for baseline, counterfactual in zip(
        baseline_model.parameters(),
        counterfactual_model.parameters(),
        strict=True,
    ):
        difference = baseline.detach() - counterfactual.detach()
        parameter_l2 += float(difference.float().pow(2).sum().cpu())
        parameter_max = max(parameter_max, float(difference.float().abs().max().cpu()))
    return {
        "start_step": start_step,
        "end_step": end_step,
        "masked_dataset_indices": sorted(set(int(value) for value in dataset_indices)),
        "masked_steps": sorted(set(int(value) for value in intervention_steps)),
        "baseline_circuit_score": baseline_score,
        "counterfactual_circuit_score": counterfactual_score,
        "circuit_score_change": counterfactual_score - baseline_score,
        "relative_circuit_score_change": (counterfactual_score - baseline_score)
        / max(abs(baseline_score), 1e-12),
        "baseline_model_sha256": state_dict_hash(baseline_model.state_dict()),
        "counterfactual_model_sha256": state_dict_hash(counterfactual_model.state_dict()),
        "parameter_l2": parameter_l2**0.5,
        "parameter_max_abs": parameter_max,
    }


def _selected_features(graph):
    output = {}
    for node in graph["nodes"]:
        if node["kind"] == "sae_feature":
            output.setdefault(node["layer"], []).append(int(node["feature_id"]))
    return {layer: sorted(set(values)) for layer, values in output.items()}


def _keep_functions(saes, selected):
    return {
        layer: (
            lambda activation, layer=layer: modify_sparse_features(
                activation,
                saes[layer],
                selected[layer],
                "keep_selected",
            )
        )
        for layer in selected
    }
