import json
from pathlib import Path

import torch

from feature_genesis.circuits.discovery import (
    CircuitProbeSet,
    collect_feature_probes,
    discover_sparse_circuit,
    refresh_probe_target_values,
    select_downstream_logit,
)
from feature_genesis.circuits.paths import (
    accepted_target_feature_ids,
    circuit_graph_path,
    circuit_root,
    circuit_sae_path,
    circuit_steps,
    feature_circuit_root,
)
from feature_genesis.config import load_config
from feature_genesis.data.factory import build_datasets, eval_loader
from feature_genesis.hashing import sha256_file, state_dict_hash
from feature_genesis.sae.io import load_sae
from feature_genesis.training.replay import reconstruct_state


CONFIG_PATH = Path("configs/quickdraw_dgx_v4.yaml")
FEATURE_IDS = None
CHECKPOINT_STEPS = None
REUSE_EXISTING = True


def main():
    config = load_config(CONFIG_PATH)
    sae_gate_path = circuit_root(config) / "sae_gate.json"
    if not sae_gate_path.exists():
        raise RuntimeError(
            "Circuit SAE gate is missing; run scripts/23_train_circuit_saes.py successfully first"
        )
    sae_gate = json.loads(sae_gate_path.read_text(encoding="utf-8"))
    if not all(row.get("passed", False) for row in sae_gate.values()):
        raise RuntimeError(f"Circuit SAE gate failed: {sae_gate_path}")
    bundle = build_datasets(config, include_annotations=False, canonical_train=True)
    feature_ids = FEATURE_IDS or accepted_target_feature_ids(config)
    steps = CHECKPOINT_STEPS or circuit_steps(config)
    if not feature_ids:
        raise RuntimeError("No validated circuit target features were found")
    final_step = max(steps)
    final_model, _, _, _ = reconstruct_state(config, final_step, bundle=bundle)
    final_device = next(final_model.parameters()).device
    final_target_sae = load_sae(
        config,
        circuit_sae_path(config, config.circuit.target_layer, final_step),
        final_device,
    )[0]
    for feature_id in feature_ids:
        fixed_probe_path = feature_circuit_root(config, feature_id) / "fixed_probes.pt"
        fixed_target_path = feature_circuit_root(config, feature_id) / "fixed_target.json"
        if REUSE_EXISTING and fixed_probe_path.exists() and fixed_target_path.exists():
            continue
        loader = eval_loader(
            bundle.validation,
            config,
            config.circuit.discovery_scan_images,
            config.circuit.discovery_batch_size,
        )
        probes = collect_feature_probes(
            final_model,
            final_target_sae,
            loader,
            config.circuit.target_layer,
            feature_id,
            final_device,
            config.circuit.discovery_scan_images,
            config.circuit.discovery_positive_images,
        )
        downstream = select_downstream_logit(
            final_model,
            final_target_sae,
            probes,
            config.circuit.target_layer,
            feature_id,
            config.circuit.discovery_batch_size,
            final_device,
        )
        fixed_probe_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "images": probes.images,
                "labels": probes.labels,
                "dataset_indices": probes.dataset_indices,
                "target_values": probes.target_values,
                "scanned_images": probes.scanned_images,
            },
            fixed_probe_path,
        )
        fixed_target_path.write_text(
            json.dumps(downstream, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    del final_model
    del final_target_sae
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    for step in steps:
        model, _, _, _ = reconstruct_state(config, step, bundle=bundle)
        device = next(model.parameters()).device
        saes = {
            layer: load_sae(config, circuit_sae_path(config, layer, step), device)[0]
            for layer in config.circuit.layers
        }
        for feature_id in feature_ids:
            path = circuit_graph_path(config, feature_id, step)
            if REUSE_EXISTING and path.exists():
                print(path)
                continue
            fixed_probe_path = feature_circuit_root(config, feature_id) / "fixed_probes.pt"
            fixed_target_path = feature_circuit_root(config, feature_id) / "fixed_target.json"
            probes = CircuitProbeSet(
                **torch.load(fixed_probe_path, map_location="cpu", weights_only=False)
            )
            probes = refresh_probe_target_values(
                model,
                saes[config.circuit.target_layer],
                probes,
                config.circuit.target_layer,
                feature_id,
                config.circuit.discovery_batch_size,
                device,
            )
            if not bool((probes.target_values > 0).any()):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(
                    json.dumps(
                        {
                            "schema_version": "sparse-visual-circuit-v1",
                            "status": "target_inactive",
                            "model_step": step,
                            "target": {
                                "layer": config.circuit.target_layer,
                                "feature_id": feature_id,
                            },
                            "probe": {
                                "cohort": "fixed_final_positive_images",
                                "images": len(probes.images),
                                "dataset_indices": [
                                    int(value) for value in probes.dataset_indices.tolist()
                                ],
                            },
                            "nodes": [],
                            "edges": [],
                            "graph_summary": {"nodes": 0, "edges": 0},
                        },
                        indent=2,
                        sort_keys=True,
                    ),
                    encoding="utf-8",
                )
                print(path)
                continue
            fixed_target = json.loads(fixed_target_path.read_text(encoding="utf-8"))
            graph = discover_sparse_circuit(
                model,
                saes,
                probes,
                config.circuit.layers,
                config.circuit.target_layer,
                feature_id,
                step,
                config,
                bundle.class_names,
                int(fixed_target["class_id"]),
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            graph["provenance"] = {
                "model_sha256": state_dict_hash(model.state_dict()),
                "sae_sha256": {
                    layer: sha256_file(circuit_sae_path(config, layer, step))
                    for layer in config.circuit.layers
                },
                "probe_artifact": str(fixed_probe_path),
                "probe_artifact_sha256": sha256_file(fixed_probe_path),
                "fixed_target": str(fixed_target_path),
                "fixed_target_sha256": sha256_file(fixed_target_path),
            }
            graph["probe"]["cohort"] = "fixed_final_positive_images"
            path.write_text(json.dumps(graph, indent=2, sort_keys=True), encoding="utf-8")
            print(path)


if __name__ == "__main__":
    main()
