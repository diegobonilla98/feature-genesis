import hashlib
import json
from pathlib import Path

import numpy as np

from feature_genesis.activations import ActivationCache
from feature_genesis.circuits.paths import (
    accepted_target_feature_ids,
    circuit_cache_root,
    circuit_graph_path,
    circuit_layer_config,
    circuit_root,
    circuit_sae_path,
    circuit_steps,
    layer_slug,
)
from feature_genesis.config import load_config
from feature_genesis.data.factory import build_datasets
from feature_genesis.determinism import resolve_device
from feature_genesis.evaluation.concepts import image_max_codes
from feature_genesis.labeling.cards import (
    add_blind_validation_tiles,
    build_feature_card_from_scores,
    render_individual_discovery_evidence,
    render_individual_validation_evidence,
    write_feature_card,
)
from feature_genesis.labeling.gemini import evaluate_feature_cards
from feature_genesis.sae.io import load_sae


CONFIG_PATH = Path("configs/quickdraw_dgx_v4.yaml")
MAX_NEW_NODES = None
RUN_GEMINI = True


def main():
    config = load_config(CONFIG_PATH)
    final_step = max(circuit_steps(config))
    root = circuit_root(config) / "node_labels"
    root.mkdir(parents=True, exist_ok=True)
    labels = existing_labels(config, root)
    strengths = circuit_node_strengths(config, final_step)
    missing = [
        node
        for node, _ in sorted(strengths.items(), key=lambda item: item[1], reverse=True)
        if node not in labels
    ]
    limit = config.circuit.gemini_max_new_nodes if MAX_NEW_NODES is None else MAX_NEW_NODES
    selected = missing[:limit]
    if selected and RUN_GEMINI:
        bundle = build_datasets(config, include_annotations=True, canonical_train=True)
        device = resolve_device(config.project.device)
        feature_inputs = []
        encoded_to_node = {}
        for layer_index, layer in enumerate(config.circuit.layers):
            layer_nodes = [node for node in selected if node.startswith(f"{layer}:feature:")]
            if not layer_nodes:
                continue
            feature_ids = [int(node.rsplit(":", 1)[1]) for node in layer_nodes]
            layer_config = circuit_layer_config(config, layer)
            layer_config.activations.evaluation_split = config.activations.evaluation_split
            layer_config.evaluation.gemini_max_uncached_requests = limit
            layer_config.evaluation.gemini_max_run_cost_usd = config.circuit.gemini_max_run_cost_usd
            cache = ActivationCache.load(
                circuit_cache_root(
                    config,
                    layer,
                    final_step,
                    config.activations.evaluation_split,
                )
            )
            sae, _ = load_sae(config, circuit_sae_path(config, layer, final_step), device)
            scores = image_max_codes(
                sae,
                cache,
                device,
                layer_config.activations.batch_size,
                np.asarray(feature_ids, dtype=np.int64),
            )
            layer_root = root / "evidence" / layer_slug(layer)
            for local_index, (node, feature_id) in enumerate(zip(layer_nodes, feature_ids, strict=True)):
                card = build_feature_card_from_scores(
                    layer_config,
                    sae,
                    cache,
                    feature_id,
                    scores[:, local_index],
                )
                card = add_blind_validation_tiles(layer_config, card)
                discovery = render_individual_discovery_evidence(
                    layer_config,
                    bundle,
                    cache,
                    sae,
                    card,
                    layer_root,
                    device,
                )
                validation = render_individual_validation_evidence(
                    layer_config,
                    bundle,
                    card,
                    layer_root,
                )
                encoded_id = (layer_index + 1) * 10000 + feature_id
                encoded_to_node[encoded_id] = node
                card["original_feature_id"] = feature_id
                card["circuit_layer"] = layer
                card["circuit_node_id"] = node
                card["feature_id"] = encoded_id
                card_path = write_feature_card(card, root / "cards")
                feature_inputs.append(
                    {
                        "card": card,
                        "card_path": card_path,
                        "discovery_images": discovery,
                        "validation_images": validation,
                    }
                )
        batch_key = hashlib.sha256("|".join(sorted(selected)).encode()).hexdigest()[:12]
        gemini_root = root / "gemini" / f"batch_{batch_key}"
        gemini_config = config.model_copy(deep=True)
        gemini_config.evaluation.gemini_max_uncached_requests = limit
        gemini_config.evaluation.gemini_max_run_cost_usd = (
            config.circuit.gemini_max_run_cost_usd
        )
        manifest = evaluate_feature_cards(gemini_config, feature_inputs, gemini_root)
        if manifest.get("status") != "complete":
            raise RuntimeError(f"Circuit-node Gemini batch did not complete: {manifest['path']}")
        for path in gemini_root.glob("feature_*_evaluation.json"):
            row = json.loads(path.read_text(encoding="utf-8"))
            node = encoded_to_node[int(row["feature_id"])]
            labels[node] = {
                "canonical_label": row["label"]["canonical_label"],
                "plain_english_summary": row["label"]["plain_english_summary"],
                "validated_status": row["validated_status"],
                "confidence": row["label"].get("confidence"),
                "source": str(path),
            }
    path = root / "labels.json"
    path.write_text(json.dumps(labels, indent=2, sort_keys=True), encoding="utf-8")
    print(path)
    print(f"labeled={len(labels)} remaining={max(0, len(strengths) - len(labels))}")


def circuit_node_strengths(config, step):
    strengths = {}
    for feature_id in accepted_target_feature_ids(config):
        path = circuit_graph_path(config, feature_id, step)
        if not path.exists():
            continue
        graph = json.loads(path.read_text(encoding="utf-8"))
        for edge in graph.get("edges", []):
            strength = abs(float(edge["attribution_mean"]))
            for node in [edge["source"], edge["target"]]:
                if ":feature:" in node:
                    strengths[node] = strengths.get(node, 0.0) + strength
    return strengths


def existing_labels(config, root):
    labels_path = root / "labels.json"
    labels = json.loads(labels_path.read_text(encoding="utf-8")) if labels_path.exists() else {}
    for feature_id in accepted_target_feature_ids(config):
        matches = list(
            Path(config.project.run_dir).glob(
                f"feature_evaluation/**/gemini/feature_{feature_id:05d}_evaluation.json"
            )
        )
        if not matches:
            continue
        row = json.loads(matches[-1].read_text(encoding="utf-8"))
        node = f"{config.circuit.target_layer}:feature:{feature_id}"
        labels[node] = {
            "canonical_label": row["label"]["canonical_label"],
            "plain_english_summary": row["label"]["plain_english_summary"],
            "validated_status": row["validated_status"],
            "confidence": row["label"].get("confidence"),
            "source": str(matches[-1]),
        }
    for path in root.glob("gemini/batch_*/feature_*_evaluation.json"):
        row = json.loads(path.read_text(encoding="utf-8"))
        card_path = root / "cards" / f"feature_{int(row['feature_id']):05d}.json"
        if not card_path.exists():
            continue
        card = json.loads(card_path.read_text(encoding="utf-8"))
        labels[card["circuit_node_id"]] = {
            "canonical_label": row["label"]["canonical_label"],
            "plain_english_summary": row["label"]["plain_english_summary"],
            "validated_status": row["validated_status"],
            "confidence": row["label"].get("confidence"),
            "source": str(path),
        }
    return labels


if __name__ == "__main__":
    main()
