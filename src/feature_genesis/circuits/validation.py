from __future__ import annotations

from typing import Any

import torch

from feature_genesis.circuits.discovery import CircuitProbeSet
from feature_genesis.circuits.sparse import (
    centered_logit,
    encode_spatial,
    max_feature_values,
    modify_sparse_features,
)
from feature_genesis.models.hooks import ActivationCapture, MultiActivationModifier


@torch.no_grad()
def validate_sparse_circuit(
    model,
    saes: dict[str, Any],
    probes: CircuitProbeSet,
    graph: dict[str, Any],
    config,
) -> dict[str, Any]:
    device = next(model.parameters()).device
    model.eval()
    for sae in saes.values():
        sae.eval()
    all_selected = _selected_features(graph)
    class_id = int(graph["target"]["selected_logit"]["class_id"])
    batch_size = config.circuit.discovery_batch_size
    baseline = _mean_logit_score(model, probes, class_id, batch_size, device)
    error_only = _mean_logit_score(
        model,
        probes,
        class_id,
        batch_size,
        device,
        _modifier_functions(saes, all_selected, "remove_all"),
    )
    denominator = baseline - error_only
    pruning_curve = []
    selected = all_selected
    circuit_only = error_only
    for node_count in config.circuit.pruning_node_counts:
        candidate = _selected_features(graph, node_count)
        candidate_score = _mean_logit_score(
            model,
            probes,
            class_id,
            batch_size,
            device,
            _modifier_functions(saes, candidate, "keep_selected"),
        )
        candidate_faithfulness = _normalized_effect(
            candidate_score - error_only,
            denominator,
        )
        pruning_curve.append(
            {
                "maximum_nodes_per_layer": node_count,
                "selected_nodes": sum(len(values) for values in candidate.values()),
                "circuit_only_centered_logit": candidate_score,
                "sufficiency_faithfulness": candidate_faithfulness,
            }
        )
        selected = candidate
        circuit_only = candidate_score
        if candidate_faithfulness >= config.circuit.minimum_faithfulness:
            break
    selected_ablated = _mean_logit_score(
        model,
        probes,
        class_id,
        batch_size,
        device,
        _modifier_functions(saes, selected, "ablate_selected"),
    )
    sufficiency = _normalized_effect(circuit_only - error_only, denominator)
    necessity = _normalized_effect(baseline - selected_ablated, denominator)
    selected_node_ids = {
        f"{layer}:feature:{feature_id}"
        for layer, feature_ids in selected.items()
        for feature_id in feature_ids
    }
    ranked_edges = sorted(
        [
            edge
            for edge in graph["edges"]
            if edge["source"] in selected_node_ids
            and (
                edge["target"] in selected_node_ids
                or edge["target"].startswith("logit:")
            )
        ],
        key=lambda row: abs(float(row["attribution_mean"])),
        reverse=True,
    )[: config.circuit.validation_edges]
    edge_results = [
        _validate_edge(model, saes, probes, edge, batch_size, device)
        for edge in ranked_edges
    ]
    testable = [row for row in edge_results if row["testable"]]
    confirmation_fraction = (
        sum(row["predicted_sign_confirmed"] for row in testable) / len(testable)
        if testable
        else 0.0
    )
    target_layer = graph["target"]["layer"]
    target_feature_id = int(graph["target"]["feature_id"])
    target_effect = _target_feature_effect(
        model,
        saes[target_layer],
        probes,
        target_layer,
        target_feature_id,
        class_id,
        batch_size,
        device,
    )
    passed = (
        sufficiency >= config.circuit.minimum_faithfulness
        and confirmation_fraction >= config.circuit.minimum_edge_sign_agreement
        and target_effect["absolute_centered_logit_effect"] > 1e-8
    )
    return {
        "schema_version": "sparse-visual-circuit-validation-v1",
        "model_step": int(graph["model_step"]),
        "target": graph["target"],
        "probe_images": len(probes.images),
        "scores": {
            "baseline_centered_logit": baseline,
            "circuit_only_centered_logit": circuit_only,
            "error_only_centered_logit": error_only,
            "selected_ablated_centered_logit": selected_ablated,
            "sufficiency_faithfulness": sufficiency,
            "necessity_fraction": necessity,
        },
        "pruning_curve": pruning_curve,
        "selected_node_ids": sorted(selected_node_ids),
        "target_feature_intervention": target_effect,
        "edge_validation": edge_results,
        "testable_edges": len(testable),
        "edge_sign_confirmation_fraction": confirmation_fraction,
        "thresholds": {
            "minimum_faithfulness": config.circuit.minimum_faithfulness,
            "minimum_edge_sign_agreement": config.circuit.minimum_edge_sign_agreement,
        },
        "passed": passed,
    }


def _selected_features(graph, maximum_per_layer=None):
    output = {}
    for node in graph["nodes"]:
        if node["kind"] == "sae_feature":
            output.setdefault(node["layer"], []).append(node)
    selected = {}
    for layer, nodes in output.items():
        ranked = sorted(
            nodes,
            key=lambda row: float(row.get("maximum_absolute_attribution", 0.0)),
            reverse=True,
        )
        if maximum_per_layer is not None:
            forced = [row for row in ranked if row.get("forced_target")]
            others = [row for row in ranked if not row.get("forced_target")]
            ranked = forced + others[: max(0, maximum_per_layer - len(forced))]
        selected[layer] = sorted(set(int(row["feature_id"]) for row in ranked))
    return selected


def _modifier_functions(saes, selected, mode):
    return {
        layer: (
            lambda activation, layer=layer: modify_sparse_features(
                activation,
                saes[layer],
                selected[layer],
                mode,
            )
        )
        for layer in selected
    }


def _normalized_effect(numerator, denominator):
    if abs(denominator) <= 1e-12:
        return 0.0
    return numerator / denominator


@torch.no_grad()
def _mean_logit_score(model, probes, class_id, batch_size, device, functions=None):
    values = []
    for batch in probes.batches(batch_size, device):
        if functions:
            with MultiActivationModifier(model, functions):
                logits = model(batch["image"])
        else:
            logits = model(batch["image"])
        values.append(centered_logit(logits, class_id).cpu())
    return float(torch.cat(values).mean())


@torch.no_grad()
def _mean_feature_score(model, sae, probes, layer, feature_id, batch_size, device, functions=None):
    values = []
    with ActivationCapture(model, layer, detach=True) as capture:
        for batch in probes.batches(batch_size, device):
            if functions:
                with MultiActivationModifier(model, functions):
                    model(batch["image"])
            else:
                model(batch["image"])
            codes, _ = encode_spatial(sae, capture.value)
            values.append(max_feature_values(codes, feature_id).cpu())
    return float(torch.cat(values).mean())


@torch.no_grad()
def _validate_edge(model, saes, probes, edge, batch_size, device):
    source = _parse_feature_node(edge["source"])
    target = _parse_feature_node(edge["target"])
    if source is None:
        return {"source": edge["source"], "target": edge["target"], "testable": False}
    source_layer, source_feature = source
    function = {
        source_layer: lambda activation: modify_sparse_features(
            activation,
            saes[source_layer],
            [source_feature],
            "ablate_selected",
        )
    }
    if target is None and edge["target"].startswith("logit:"):
        class_id = int(edge["target"].split(":", 1)[1])
        baseline = _mean_logit_score(model, probes, class_id, batch_size, device)
        ablated = _mean_logit_score(model, probes, class_id, batch_size, device, function)
    elif target is not None:
        target_layer, target_feature = target
        baseline = _mean_feature_score(
            model,
            saes[target_layer],
            probes,
            target_layer,
            target_feature,
            batch_size,
            device,
        )
        ablated = _mean_feature_score(
            model,
            saes[target_layer],
            probes,
            target_layer,
            target_feature,
            batch_size,
            device,
            function,
        )
    else:
        return {"source": edge["source"], "target": edge["target"], "testable": False}
    presence_effect = baseline - ablated
    predicted = float(edge["attribution_mean"])
    return {
        "source": edge["source"],
        "target": edge["target"],
        "testable": True,
        "predicted_attribution": predicted,
        "baseline_target_response": baseline,
        "source_ablated_target_response": ablated,
        "causal_presence_effect": presence_effect,
        "predicted_sign_confirmed": bool(predicted * presence_effect > 0),
    }


@torch.no_grad()
def _target_feature_effect(
    model,
    sae,
    probes,
    layer,
    feature_id,
    class_id,
    batch_size,
    device,
):
    function = {
        layer: lambda activation: modify_sparse_features(
            activation,
            sae,
            [feature_id],
            "ablate_selected",
        )
    }
    baseline = _mean_logit_score(model, probes, class_id, batch_size, device)
    ablated = _mean_logit_score(model, probes, class_id, batch_size, device, function)
    return {
        "baseline_centered_logit": baseline,
        "ablated_centered_logit": ablated,
        "centered_logit_presence_effect": baseline - ablated,
        "absolute_centered_logit_effect": abs(baseline - ablated),
    }


def _parse_feature_node(node_id):
    marker = ":feature:"
    if marker not in node_id:
        return None
    layer, value = node_id.rsplit(marker, 1)
    return layer, int(value)
