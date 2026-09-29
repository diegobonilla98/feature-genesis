from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from feature_genesis.sae.io import sae_analysis_root


def strongest_ancestry_path(
    payload: dict[str, Any],
    final_step: int,
    final_feature_id: int,
) -> dict[str, list[dict[str, Any]]]:
    nodes = {str(node["id"]): node for node in payload.get("nodes", [])}
    incoming: dict[str, list[dict[str, Any]]] = {}
    for edge in payload.get("edges", []):
        incoming.setdefault(str(edge["target"]), []).append(edge)
    current_id = f"s{final_step}:f{final_feature_id}"
    if current_id not in nodes:
        return {"nodes": [], "edges": []}
    reversed_nodes = [nodes[current_id]]
    reversed_edges = []
    visited = {current_id}
    while current_id in incoming:
        current_step = int(nodes[current_id]["step"])
        candidates = [
            edge
            for edge in incoming[current_id]
            if str(edge["source"]) in nodes
            and int(nodes[str(edge["source"])]["step"]) < current_step
        ]
        if not candidates:
            break
        edge = max(candidates, key=lineage_edge_priority)
        source_id = str(edge["source"])
        if source_id in visited:
            raise RuntimeError(f"Lineage ancestry contains a cycle at {source_id}")
        visited.add(source_id)
        reversed_edges.append(edge)
        reversed_nodes.append(nodes[source_id])
        current_id = source_id
    return {
        "nodes": list(reversed(reversed_nodes)),
        "edges": list(reversed(reversed_edges)),
    }


def lineage_edge_priority(edge: dict[str, Any]) -> tuple[float, float, float, float, str]:
    return (
        float(edge.get("score", float("-inf"))),
        float(edge.get("concept_similarity", float("-inf"))),
        float(edge.get("activation_similarity", float("-inf"))),
        float(edge.get("decoder_similarity", float("-inf"))),
        str(edge.get("source", "")),
    )


def resolve_birth_target(
    config,
    kind: str,
    final_feature_id: int,
) -> tuple[int, int, dict[str, list[dict[str, Any]]]]:
    steps = sorted(config.activations.checkpoint_steps)
    final_step = steps[-1]
    path = sae_analysis_root(config, "lineage", kind=kind) / "lineage.json"
    if not path.exists():
        return final_step, final_feature_id, {
            "nodes": [{"step": final_step, "feature": final_feature_id}],
            "edges": [],
        }
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    ancestry = strongest_ancestry_path(payload, final_step, final_feature_id)
    if not ancestry["nodes"]:
        return final_step, final_feature_id, {
            "nodes": [{"step": final_step, "feature": final_feature_id}],
            "edges": [],
        }
    eligible = [
        node for node in ancestry["nodes"] if int(node["step"]) > steps[0]
    ]
    target = eligible[0] if eligible else ancestry["nodes"][-1]
    target_step = int(target["step"])
    if not any(step < target_step for step in steps):
        raise RuntimeError(
            f"Feature {final_feature_id} has no checkpoint before birth step {target_step}"
        )
    return target_step, int(target["feature"]), ancestry
