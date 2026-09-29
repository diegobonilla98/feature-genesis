import json
from pathlib import Path

import torch

from feature_genesis.circuits.attribution import rank_circuit_training_batches
from feature_genesis.circuits.discovery import CircuitProbeSet
from feature_genesis.circuits.paths import (
    accepted_target_feature_ids,
    circuit_graph_path,
    circuit_lineage_path,
    circuit_sae_path,
    feature_circuit_root,
)
from feature_genesis.config import load_config
from feature_genesis.data.factory import build_datasets
from feature_genesis.hashing import state_dict_hash
from feature_genesis.sae.io import load_sae
from feature_genesis.training.replay import reconstruct_state


CONFIG_PATH = Path("configs/quickdraw_dgx_v4.yaml")
FEATURE_IDS = None
REUSE_EXISTING = True


def main():
    config = load_config(CONFIG_PATH)
    bundle = build_datasets(config, include_annotations=False, canonical_train=True)
    feature_ids = FEATURE_IDS or accepted_target_feature_ids(config)
    for feature_id in feature_ids:
        path = feature_circuit_root(config, feature_id) / "training_attribution.json"
        if REUSE_EXISTING and path.exists():
            print(path)
            continue
        lineage = json.loads(circuit_lineage_path(config, feature_id).read_text(encoding="utf-8"))
        end_step = lineage.get("emergence_step")
        if end_step is None:
            print(f"skip unresolved circuit feature {feature_id}")
            continue
        prior_steps = [row["step"] for row in lineage["trajectory"] if row["step"] < end_step]
        start_step = max(prior_steps) if prior_steps else 0
        graph = json.loads(circuit_graph_path(config, feature_id, end_step).read_text(encoding="utf-8"))
        probe_payload = torch.load(
            Path(graph["provenance"]["probe_artifact"]),
            map_location="cpu",
            weights_only=False,
        )
        probes = CircuitProbeSet(**probe_payload)
        model, optimizer, _, plan = reconstruct_state(config, start_step, bundle=bundle)
        device = next(model.parameters()).device
        saes = {
            layer: load_sae(config, circuit_sae_path(config, layer, end_step), device)[0]
            for layer in config.circuit.layers
        }
        result = rank_circuit_training_batches(
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
        )
        result["feature_id"] = feature_id
        result["graph_step"] = end_step
        result["replayed_model_sha256"] = state_dict_hash(model.state_dict())
        expected_model, _, _, _ = reconstruct_state(config, end_step, bundle=bundle)
        result["expected_model_sha256"] = state_dict_hash(expected_model.state_dict())
        result["replay_verified"] = (
            result["replayed_model_sha256"] == result["expected_model_sha256"]
        )
        if not result["replay_verified"]:
            raise RuntimeError(f"Circuit attribution replay diverged for feature {feature_id}")
        path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
        print(path)


if __name__ == "__main__":
    main()
