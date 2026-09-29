from __future__ import annotations

import math
from typing import Any


def build_temporal_circuit_lineage(
    graphs: dict[int, dict[str, Any]],
    validations: dict[int, dict[str, Any]],
    config,
) -> dict[str, Any]:
    if not graphs:
        raise ValueError("At least one circuit graph is required")
    final_step = max(graphs)
    final_graph = graphs[final_step]
    final_edges = _edge_weights(final_graph)
    rows = []
    for step in sorted(graphs):
        graph = graphs[step]
        validation = validations.get(step)
        inactive = graph.get("status") == "target_inactive"
        similarity = weighted_edge_jaccard(_edge_weights(graph), final_edges)
        stable_edges = [
            edge
            for edge in graph.get("edges", [])
            if _interval_excludes_zero(edge.get("bootstrap_ci", [0.0, 0.0]))
            and float(edge.get("sign_agreement", 0.0))
            >= config.circuit.minimum_edge_sign_agreement
        ]
        stable_fraction = len(stable_edges) / max(1, len(graph.get("edges", [])))
        faithfulness = (
            float(validation["scores"]["sufficiency_faithfulness"])
            if validation is not None
            else 0.0
        )
        sign_confirmation = (
            float(validation["edge_sign_confirmation_fraction"])
            if validation is not None
            else 0.0
        )
        target_effect = (
            float(
                validation["target_feature_intervention"][
                    "absolute_centered_logit_effect"
                ]
            )
            if validation is not None
            else 0.0
        )
        confidence = _geometric_mean(
            [
                similarity,
                min(1.0, max(0.0, faithfulness)),
                sign_confirmation,
                stable_fraction,
            ]
        )
        qualifies = bool(
            not inactive
            and validation is not None
            and validation.get("passed")
            and similarity >= config.circuit.minimum_final_edge_similarity
            and target_effect > 1e-8
        )
        rows.append(
            {
                "step": step,
                "nodes": len(graph.get("nodes", [])),
                "edges": len(graph.get("edges", [])),
                "total_absolute_attribution": float(
                    graph.get("graph_summary", {}).get("total_absolute_attribution", 0.0)
                ),
                "target_activation_mean": float(
                    graph.get("probe", {}).get("target_activation_mean", 0.0)
                ),
                "final_edge_similarity": similarity,
                "stable_edge_fraction": stable_fraction,
                "sufficiency_faithfulness": faithfulness,
                "edge_sign_confirmation_fraction": sign_confirmation,
                "target_absolute_logit_effect": target_effect,
                "emergence_confidence": confidence,
                "qualifies_as_emerged": qualifies,
                "target_inactive": inactive,
            }
        )
    emergence_index = _first_persistent(rows, config.circuit.persistence_checkpoints)
    emergence_step = rows[emergence_index]["step"] if emergence_index is not None else None
    emergence_status = (
        "unresolved"
        if emergence_index is None
        else "left_censored"
        if emergence_index == 0
        else "interval_censored"
    )
    for index, row in enumerate(rows):
        row["phase"] = _phase(index, emergence_index, row, final_step)
    return {
        "schema_version": "sparse-visual-circuit-lineage-v1",
        "target": final_graph["target"],
        "final_step": final_step,
        "emergence_step": emergence_step,
        "emergence_status": emergence_status,
        "emergence_interval": _emergence_interval(rows, emergence_index),
        "persistence_checkpoints": config.circuit.persistence_checkpoints,
        "thresholds": {
            "minimum_faithfulness": config.circuit.minimum_faithfulness,
            "minimum_edge_sign_agreement": config.circuit.minimum_edge_sign_agreement,
            "minimum_final_edge_similarity": config.circuit.minimum_final_edge_similarity,
        },
        "trajectory": rows,
    }


def weighted_edge_jaccard(left: dict[str, float], right: dict[str, float]) -> float:
    keys = set(left) | set(right)
    if not keys:
        return 1.0
    numerator = sum(min(abs(left.get(key, 0.0)), abs(right.get(key, 0.0))) for key in keys)
    denominator = sum(max(abs(left.get(key, 0.0)), abs(right.get(key, 0.0))) for key in keys)
    return numerator / max(denominator, 1e-12)


def _edge_weights(graph):
    return {
        f"{edge['source']}->{edge['target']}": float(edge["attribution_mean"])
        for edge in graph.get("edges", [])
    }


def _interval_excludes_zero(interval):
    lower, upper = float(interval[0]), float(interval[1])
    return lower > 0 or upper < 0


def _geometric_mean(values):
    clipped = [min(1.0, max(0.0, float(value))) for value in values]
    if any(value == 0 for value in clipped):
        return 0.0
    return math.exp(sum(math.log(value) for value in clipped) / len(clipped))


def _first_persistent(rows, persistence):
    for start in range(len(rows)):
        window = rows[start : start + persistence]
        if len(window) == persistence and all(row["qualifies_as_emerged"] for row in window):
            return start
    return None


def _emergence_interval(rows, index):
    if index is None:
        return None
    lower = rows[index - 1]["step"] if index > 0 else rows[index]["step"]
    return [lower, rows[index]["step"]]


def _phase(index, emergence_index, row, final_step):
    if emergence_index is None:
        return "unresolved"
    if index < emergence_index:
        return "assembly" if row["final_edge_similarity"] > 0 else "pre-circuit"
    if index == emergence_index:
        return "emergence"
    if row["step"] == final_step:
        return "mature"
    return "consolidation"
