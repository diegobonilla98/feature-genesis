from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from feature_genesis.circuits.sparse import (
    centered_logit,
    contribution_scores,
    encode_spatial,
    feature_node_id,
    logit_node_id,
    max_feature_values,
)
from feature_genesis.hashing import stable_seed
from feature_genesis.models.hooks import ActivationCapture, ActivationModifier, MultiActivationCapture
from feature_genesis.models.interventions import ablate_sae_feature


@dataclass
class CircuitProbeSet:
    images: torch.Tensor
    labels: torch.Tensor
    dataset_indices: torch.Tensor
    target_values: torch.Tensor
    scanned_images: int

    def batches(self, batch_size: int, device: torch.device):
        for start in range(0, len(self.images), batch_size):
            end = min(len(self.images), start + batch_size)
            yield {
                "image": self.images[start:end].to(device),
                "label": self.labels[start:end].to(device),
                "dataset_index": self.dataset_indices[start:end],
                "target_value": self.target_values[start:end],
            }


@torch.no_grad()
def collect_feature_probes(
    model,
    target_sae,
    loader,
    target_layer: str,
    feature_id: int,
    device: torch.device,
    scan_images: int,
    positive_images: int,
) -> CircuitProbeSet:
    model.eval()
    target_sae.eval()
    images = []
    labels = []
    dataset_indices = []
    values = []
    scanned = 0
    with ActivationCapture(model, target_layer, detach=True) as capture:
        for batch in loader:
            remaining = scan_images - scanned
            if remaining <= 0 or sum(len(value) for value in images) >= positive_images:
                break
            batch_images = batch["image"][:remaining].to(device)
            model(batch_images)
            if capture.value is None:
                raise RuntimeError("Target activation hook did not fire")
            codes, _ = encode_spatial(target_sae, capture.value)
            feature_values = max_feature_values(codes, feature_id)
            active = feature_values > 0
            if active.any():
                positions = active.nonzero().flatten().cpu()
                wanted = positive_images - sum(len(value) for value in images)
                positions = positions[:wanted]
                images.append(batch_images[positions.to(device)].cpu())
                labels.append(batch["label"][:remaining][positions].cpu())
                dataset_indices.append(batch["dataset_index"][:remaining][positions].cpu())
                values.append(feature_values[positions.to(device)].cpu())
            scanned += len(batch_images)
    if not images:
        raise RuntimeError(f"Feature {feature_id} did not activate in {scanned} scanned images")
    return CircuitProbeSet(
        images=torch.cat(images),
        labels=torch.cat(labels),
        dataset_indices=torch.cat(dataset_indices),
        target_values=torch.cat(values),
        scanned_images=scanned,
    )


@torch.no_grad()
def select_downstream_logit(
    model,
    target_sae,
    probes: CircuitProbeSet,
    target_layer: str,
    feature_id: int,
    batch_size: int,
    device: torch.device,
    class_id: int | None = None,
) -> dict[str, Any]:
    baseline_rows = []
    ablated_rows = []
    model.eval()
    for batch in probes.batches(batch_size, device):
        baseline = model(batch["image"])
        with ActivationModifier(
            model,
            target_layer,
            lambda activation: ablate_sae_feature(activation, target_sae, feature_id),
        ):
            ablated = model(batch["image"])
        baseline_rows.append(baseline.cpu())
        ablated_rows.append(ablated.cpu())
    baseline_logits = torch.cat(baseline_rows)
    ablated_logits = torch.cat(ablated_rows)
    deltas = (ablated_logits - baseline_logits).mean(dim=0)
    class_id = int(deltas.abs().argmax()) if class_id is None else int(class_id)
    baseline_centered = centered_logit(baseline_logits, class_id)
    ablated_centered = centered_logit(ablated_logits, class_id)
    return {
        "class_id": class_id,
        "mean_logit_change_when_target_ablated": float(deltas[class_id]),
        "mean_centered_logit_change_when_target_ablated": float(
            (ablated_centered - baseline_centered).mean()
        ),
        "baseline_centered_logit": float(baseline_centered.mean()),
        "ablated_centered_logit": float(ablated_centered.mean()),
    }


@torch.no_grad()
def refresh_probe_target_values(
    model,
    target_sae,
    probes: CircuitProbeSet,
    target_layer: str,
    feature_id: int,
    batch_size: int,
    device: torch.device,
) -> CircuitProbeSet:
    model.eval()
    target_sae.eval()
    values = []
    with ActivationCapture(model, target_layer, detach=True) as capture:
        for batch in probes.batches(batch_size, device):
            model(batch["image"])
            if capture.value is None:
                raise RuntimeError("Target activation hook did not fire")
            codes, _ = encode_spatial(target_sae, capture.value)
            values.append(max_feature_values(codes, feature_id).cpu())
    return CircuitProbeSet(
        images=probes.images,
        labels=probes.labels,
        dataset_indices=probes.dataset_indices,
        target_values=torch.cat(values),
        scanned_images=probes.scanned_images,
    )


def discover_sparse_circuit(
    model,
    saes: dict[str, Any],
    probes: CircuitProbeSet,
    layers: list[str],
    target_layer: str,
    target_feature_id: int,
    model_step: int,
    config,
    class_names: list[str] | None = None,
    selected_class_id: int | None = None,
) -> dict[str, Any]:
    device = next(model.parameters()).device
    model.eval()
    for sae in saes.values():
        sae.eval()
    downstream = select_downstream_logit(
        model,
        saes[target_layer],
        probes,
        target_layer,
        target_feature_id,
        config.circuit.discovery_batch_size,
        device,
        selected_class_id,
    )
    class_id = int(downstream["class_id"])
    forced = {layer: set() for layer in layers}
    forced[target_layer].add(target_feature_id)
    nodes: dict[str, dict[str, Any]] = {
        logit_node_id(class_id): {
            "id": logit_node_id(class_id),
            "kind": "logit",
            "class_id": class_id,
            "label": class_names[class_id] if class_names else f"class {class_id}",
        }
    }
    edges = []
    last_layer = layers[-1]
    logit_stats = _score_features_to_logit(
        model,
        saes[last_layer],
        probes,
        last_layer,
        class_id,
        config.circuit.discovery_batch_size,
        device,
    )
    selected_last = _top_features(
        logit_stats,
        config.circuit.max_nodes_per_layer,
        forced[last_layer],
        config.circuit.minimum_edge_attribution,
    )
    for feature_id in selected_last:
        node_id = feature_node_id(last_layer, feature_id)
        nodes[node_id] = _feature_node(last_layer, feature_id, logit_stats, forced)
        edges.append(
            _edge_payload(
                node_id,
                logit_node_id(class_id),
                logit_stats[feature_id],
                model_step,
                config,
            )
        )
    frontier = selected_last
    for target_index in range(len(layers) - 1, 0, -1):
        target_name = layers[target_index]
        source_name = layers[target_index - 1]
        transition = _score_feature_transition(
            model,
            saes[source_name],
            saes[target_name],
            probes,
            source_name,
            target_name,
            frontier,
            config.circuit.discovery_batch_size,
            device,
        )
        selected_sources, selected_edges = _select_transition(
            transition,
            config.circuit.max_nodes_per_layer,
            config.circuit.max_edges_per_target,
            forced[source_name],
            config.circuit.minimum_edge_attribution,
        )
        for feature_id in selected_sources:
            source_id = feature_node_id(source_name, feature_id)
            source_stats = _aggregate_source_stats(transition, feature_id)
            nodes[source_id] = _feature_node(source_name, feature_id, source_stats, forced)
        for source_feature, target_feature, stats in selected_edges:
            target_id = feature_node_id(target_name, target_feature)
            if target_id not in nodes:
                nodes[target_id] = _feature_node(target_name, target_feature, {}, forced)
            edges.append(
                _edge_payload(
                    feature_node_id(source_name, source_feature),
                    target_id,
                    stats,
                    model_step,
                    config,
                )
            )
        frontier = selected_sources
    target_node = feature_node_id(target_layer, target_feature_id)
    if target_node not in nodes:
        nodes[target_node] = _feature_node(target_layer, target_feature_id, {}, forced)
    connected = _nodes_connected_to_sink(nodes, edges, logit_node_id(class_id), target_node)
    nodes = {key: value for key, value in nodes.items() if key in connected}
    edges = [edge for edge in edges if edge["source"] in nodes and edge["target"] in nodes]
    return {
        "schema_version": "sparse-visual-circuit-v1",
        "method": "sparse_cross_layer_attribution_activation_times_gradient",
        "model_step": model_step,
        "layers": layers,
        "target": {
            "layer": target_layer,
            "feature_id": target_feature_id,
            "node_id": target_node,
            "selected_logit": downstream,
        },
        "probe": {
            "scanned_images": probes.scanned_images,
            "active_images": len(probes.images),
            "dataset_indices": [int(value) for value in probes.dataset_indices.tolist()],
            "target_activation_mean": float(probes.target_values.mean()),
            "target_activation_median": float(probes.target_values.median()),
        },
        "nodes": sorted(nodes.values(), key=lambda row: row["id"]),
        "edges": sorted(edges, key=lambda row: abs(row["attribution_mean"]), reverse=True),
        "graph_summary": {
            "nodes": len(nodes),
            "edges": len(edges),
            "positive_edges": sum(edge["attribution_mean"] > 0 for edge in edges),
            "negative_edges": sum(edge["attribution_mean"] < 0 for edge in edges),
            "total_absolute_attribution": float(sum(abs(edge["attribution_mean"]) for edge in edges)),
        },
    }


def _score_features_to_logit(model, sae, probes, layer, class_id, batch_size, device):
    batch_values = []
    for batch in probes.batches(batch_size, device):
        with MultiActivationCapture(model, [layer], detach=False) as capture:
            logits = model(batch["image"])
            activation = capture.values[layer]
            scalar = centered_logit(logits, class_id).sum()
            gradient = torch.autograd.grad(scalar, activation, retain_graph=False)[0]
            values = contribution_scores(activation, gradient, sae).mean(dim=0)
            batch_values.append(values.detach().cpu().numpy())
    return _summarize_matrix(np.stack(batch_values), range(sae.dictionary_size))


def _score_feature_transition(
    model,
    source_sae,
    target_sae,
    probes,
    source_layer,
    target_layer,
    target_features,
    batch_size,
    device,
):
    values = {int(feature_id): [] for feature_id in target_features}
    for batch in probes.batches(batch_size, device):
        with MultiActivationCapture(model, [source_layer, target_layer], detach=False) as capture:
            model(batch["image"])
            source_activation = capture.values[source_layer]
            target_codes, _ = encode_spatial(target_sae, capture.values[target_layer])
            for feature_id in target_features:
                target_values = max_feature_values(target_codes, feature_id)
                active = target_values > 0
                if not active.any():
                    continue
                scalar = target_values[active].sum()
                gradient = torch.autograd.grad(scalar, source_activation, retain_graph=True)[0]
                effects = contribution_scores(source_activation, gradient, source_sae)
                values[int(feature_id)].append(effects[active].mean(dim=0).detach().cpu().numpy())
    output = {}
    for target_feature, rows in values.items():
        if rows:
            output[target_feature] = _summarize_matrix(
                np.stack(rows),
                range(source_sae.dictionary_size),
            )
        else:
            output[target_feature] = {}
    return output


def _summarize_matrix(matrix: np.ndarray, feature_ids) -> dict[int, dict[str, Any]]:
    output = {}
    for column, feature_id in enumerate(feature_ids):
        values = matrix[:, column].astype(np.float64)
        mean = float(values.mean())
        output[int(feature_id)] = {
            "attribution_mean": mean,
            "attribution_std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "sign_agreement": float((np.sign(values) == np.sign(mean)).mean()) if mean != 0 else 0.0,
            "batch_values": values,
        }
    return output


def _top_features(stats, count, forced, minimum):
    ranked = [
        feature_id
        for feature_id, row in sorted(
            stats.items(),
            key=lambda item: abs(item[1]["attribution_mean"]),
            reverse=True,
        )
        if abs(row["attribution_mean"]) >= minimum
    ]
    selected = ranked[:count]
    for feature_id in sorted(forced):
        if feature_id not in selected and feature_id in stats:
            selected.append(feature_id)
    return selected


def _select_transition(transition, max_nodes, max_edges, forced, minimum):
    candidates = []
    for target_feature, stats in transition.items():
        ranked = [
            (source_feature, target_feature, row)
            for source_feature, row in sorted(
                stats.items(),
                key=lambda item: abs(item[1]["attribution_mean"]),
                reverse=True,
            )
            if abs(row["attribution_mean"]) >= minimum
        ]
        candidates.extend(ranked[:max_edges])
    source_strength = {}
    for source, target, row in candidates:
        source_strength[source] = max(source_strength.get(source, 0.0), abs(row["attribution_mean"]))
    selected = [
        source
        for source, _ in sorted(source_strength.items(), key=lambda item: item[1], reverse=True)[:max_nodes]
    ]
    for source in sorted(forced):
        if source not in selected:
            selected.append(source)
        forced_candidates = [
            (source, target, stats[source])
            for target, stats in transition.items()
            if source in stats
        ]
        if forced_candidates:
            candidates.append(
                max(forced_candidates, key=lambda item: abs(item[2]["attribution_mean"]))
            )
    unique = {}
    for source, target, row in candidates:
        if source in selected:
            unique[(source, target)] = (source, target, row)
    return selected, list(unique.values())


def _aggregate_source_stats(transition, feature_id):
    rows = [stats[feature_id] for stats in transition.values() if feature_id in stats]
    if not rows:
        return {}
    best = max(rows, key=lambda row: abs(row["attribution_mean"]))
    return best


def _feature_node(layer, feature_id, stats, forced):
    return {
        "id": feature_node_id(layer, feature_id),
        "kind": "sae_feature",
        "layer": layer,
        "feature_id": int(feature_id),
        "forced_target": int(feature_id) in forced.get(layer, set()),
        "maximum_absolute_attribution": abs(float(stats.get("attribution_mean", 0.0))),
    }


def _edge_payload(source, target, stats, model_step, config):
    values = np.asarray(stats["batch_values"], dtype=np.float64)
    lower, upper = _bootstrap_interval(
        values,
        config.circuit.bootstrap_samples,
        config.circuit.bootstrap_confidence,
        stable_seed(config.project.seed, "circuit-edge", model_step, source, target),
    )
    return {
        "source": source,
        "target": target,
        "attribution_mean": float(stats["attribution_mean"]),
        "attribution_std": float(stats["attribution_std"]),
        "bootstrap_ci": [lower, upper],
        "sign_agreement": float(stats["sign_agreement"]),
        "batches": int(len(values)),
        "causal_status": "not_tested",
    }


def _bootstrap_interval(values, samples, confidence, seed):
    if len(values) == 0:
        return 0.0, 0.0
    if len(values) == 1 or samples <= 0:
        value = float(values.mean())
        return value, value
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, len(values), size=(samples, len(values)))
    means = values[indices].mean(axis=1)
    tail = (1 - confidence) / 2
    return float(np.quantile(means, tail)), float(np.quantile(means, 1 - tail))


def _nodes_connected_to_sink(nodes, edges, sink, required):
    reverse = {}
    forward = {}
    for edge in edges:
        reverse.setdefault(edge["target"], set()).add(edge["source"])
        forward.setdefault(edge["source"], set()).add(edge["target"])
    ancestors = {sink}
    frontier = [sink]
    while frontier:
        current = frontier.pop()
        for source in reverse.get(current, set()):
            if source not in ancestors:
                ancestors.add(source)
                frontier.append(source)
    if required not in ancestors:
        ancestors.add(required)
        required_ancestors = [required]
        while required_ancestors:
            current = required_ancestors.pop()
            for source in reverse.get(current, set()):
                if source not in ancestors:
                    ancestors.add(source)
                    required_ancestors.append(source)
        descendants = [required]
        while descendants:
            current = descendants.pop()
            for target in forward.get(current, set()):
                if target not in ancestors:
                    ancestors.add(target)
                    descendants.append(target)
    return ancestors & set(nodes)
