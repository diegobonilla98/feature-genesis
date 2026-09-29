import json
from pathlib import Path

import torch

from feature_genesis.circuits.attribution import counterfactual_circuit_replay
from feature_genesis.circuits.discovery import CircuitProbeSet
from feature_genesis.circuits.paths import (
    accepted_target_feature_ids,
    circuit_graph_path,
    circuit_sae_path,
    feature_circuit_root,
)
from feature_genesis.config import load_config
from feature_genesis.data.factory import build_datasets
from feature_genesis.sae.io import load_sae


CONFIG_PATH = Path("configs/quickdraw_dgx_v4.yaml")
FEATURE_IDS = None
CANDIDATE_DIRECTIONS = ["builders", "breakers"]
REUSE_EXISTING = True


def main():
    config = load_config(CONFIG_PATH)
    bundle = build_datasets(config, include_annotations=False, canonical_train=True)
    feature_ids = FEATURE_IDS or accepted_target_feature_ids(config)
    for feature_id in feature_ids:
        root = feature_circuit_root(config, feature_id)
        attribution_path = root / "training_attribution.json"
        if not attribution_path.exists():
            continue
        attribution = json.loads(attribution_path.read_text(encoding="utf-8"))
        start_step = int(attribution["start_step"])
        end_step = int(attribution["end_step"])
        graph = json.loads(circuit_graph_path(config, feature_id, end_step).read_text(encoding="utf-8"))
        probe_payload = torch.load(
            Path(graph["provenance"]["probe_artifact"]),
            map_location="cpu",
            weights_only=False,
        )
        probes = CircuitProbeSet(**probe_payload)
        device = torch.device(config.project.device)
        saes = {
            layer: load_sae(config, circuit_sae_path(config, layer, end_step), device)[0]
            for layer in config.circuit.layers
        }
        for direction in CANDIDATE_DIRECTIONS:
            for rank in range(config.circuit.exact_replay_ranks):
                path = root / f"counterfactual_{direction}_rank_{rank:02d}.json"
                if REUSE_EXISTING and path.exists():
                    print(path)
                    continue
                candidate = attribution[direction][rank]
                example_key = "example_builders" if direction == "builders" else "example_breakers"
                indices = [int(row["dataset_index"]) for row in candidate[example_key]]
                result = counterfactual_circuit_replay(
                    config,
                    saes,
                    graph,
                    probes,
                    start_step,
                    end_step,
                    indices,
                    [int(candidate["step"])],
                    bundle,
                )
                presence_effect = -float(result["circuit_score_change"])
                result.update(
                    {
                        "feature_id": feature_id,
                        "candidate_direction": direction,
                        "candidate_rank": rank,
                        "candidate_step": int(candidate["step"]),
                        "predicted_influence": float(candidate["influence"]),
                        "causal_presence_effect": presence_effect,
                        "causal_role": "builder"
                        if presence_effect > 0
                        else "breaker"
                        if presence_effect < 0
                        else "neutral",
                        "predicted_direction_confirmed": (
                            direction == "builders" and presence_effect > 0
                        )
                        or (direction == "breakers" and presence_effect < 0),
                    }
                )
                path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
                print(path)


if __name__ == "__main__":
    main()
