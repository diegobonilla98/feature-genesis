import json
from pathlib import Path

import torch

from feature_genesis.circuits.discovery import CircuitProbeSet
from feature_genesis.circuits.paths import (
    accepted_target_feature_ids,
    circuit_graph_path,
    circuit_sae_path,
    circuit_steps,
    circuit_validation_path,
)
from feature_genesis.circuits.validation import validate_sparse_circuit
from feature_genesis.config import load_config
from feature_genesis.data.factory import build_datasets
from feature_genesis.hashing import sha256_file, state_dict_hash
from feature_genesis.sae.io import load_sae
from feature_genesis.training.replay import reconstruct_state


CONFIG_PATH = Path("configs/quickdraw_dgx_v4.yaml")
FEATURE_IDS = None
CHECKPOINT_STEPS = None
REUSE_EXISTING = True


def main():
    config = load_config(CONFIG_PATH)
    bundle = build_datasets(config, include_annotations=False, canonical_train=True)
    feature_ids = FEATURE_IDS or accepted_target_feature_ids(config)
    steps = CHECKPOINT_STEPS or circuit_steps(config)
    for step in steps:
        model, _, _, _ = reconstruct_state(config, step, bundle=bundle)
        device = next(model.parameters()).device
        saes = {
            layer: load_sae(config, circuit_sae_path(config, layer, step), device)[0]
            for layer in config.circuit.layers
        }
        for feature_id in feature_ids:
            graph_path = circuit_graph_path(config, feature_id, step)
            path = circuit_validation_path(config, feature_id, step)
            if REUSE_EXISTING and path.exists():
                print(path)
                continue
            graph = json.loads(graph_path.read_text(encoding="utf-8"))
            if graph.get("status") == "target_inactive":
                continue
            probe_path = Path(graph["provenance"]["probe_artifact"])
            if sha256_file(probe_path) != graph["provenance"]["probe_artifact_sha256"]:
                raise RuntimeError(f"Circuit probe artifact hash mismatch: {probe_path}")
            payload = torch.load(probe_path, map_location="cpu", weights_only=False)
            probes = CircuitProbeSet(**payload)
            result = validate_sparse_circuit(model, saes, probes, graph, config)
            result["provenance"] = {
                "graph": str(graph_path),
                "graph_sha256": sha256_file(graph_path),
                "model_sha256": state_dict_hash(model.state_dict()),
            }
            path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
            print(path)


if __name__ == "__main__":
    main()
