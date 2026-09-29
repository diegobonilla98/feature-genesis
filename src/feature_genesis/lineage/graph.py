from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import networkx as nx

from feature_genesis.lineage.matching import PairwiseLineage
from feature_genesis.lineage.signatures import FeatureSignatures


def build_lineage_graph(
    steps: list[int],
    signatures: dict[int, FeatureSignatures],
    pairwise: dict[tuple[int, int], PairwiseLineage],
) -> nx.DiGraph:
    graph = nx.DiGraph()
    for step in steps:
        signature = signatures[step]
        for feature in range(len(signature.frequency)):
            graph.add_node(
                node_id(step, feature),
                step=step,
                feature=feature,
                frequency=float(signature.frequency[feature]),
                mean_when_active=float(signature.mean_when_active[feature]),
            )
    for source_step, target_step in zip(steps[:-1], steps[1:], strict=True):
        result = pairwise[(source_step, target_step)]
        for match in result.matches:
            graph.add_edge(
                node_id(source_step, match.source),
                node_id(target_step, match.target),
                relation=match.relation,
                score=match.score,
                decoder_similarity=match.decoder_similarity,
                activation_similarity=match.activation_similarity,
                concept_similarity=match.concept_similarity,
            )
        for feature in result.births:
            graph.nodes[node_id(target_step, feature)]["birth"] = True
        for feature in result.deaths:
            graph.nodes[node_id(source_step, feature)]["death"] = True
    for component in nx.weakly_connected_components(graph):
        identity = "|".join(sorted(component))
        trajectory_uid = "trajectory_" + hashlib.sha256(
            identity.encode("utf-8")
        ).hexdigest()[:16]
        for node in component:
            graph.nodes[node]["trajectory_uid"] = trajectory_uid
    return graph


def save_lineage_graph(graph: nx.DiGraph, root: str | Path) -> Path:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    data = nx.node_link_data(graph, edges="edges")
    (root / "lineage.json").write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    nx.write_graphml(graph, root / "lineage.graphml")
    return root


def trace_ancestors(graph: nx.DiGraph, step: int, feature: int) -> list[dict[str, Any]]:
    target = node_id(step, feature)
    ancestors = nx.ancestors(graph, target)
    nodes = [target, *ancestors]
    return [dict(node=node, **graph.nodes[node]) for node in sorted(nodes, key=lambda value: graph.nodes[value]["step"])]


def node_id(step: int, feature: int) -> str:
    return f"s{step}:f{feature}"
