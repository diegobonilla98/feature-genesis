import importlib.util
import json
from pathlib import Path

from feature_genesis.circuits.paths import (
    accepted_target_feature_ids,
    circuit_graph_path,
    circuit_lineage_path,
    circuit_root,
    circuit_steps,
    circuit_validation_path,
)
from feature_genesis.circuits.lineage import build_temporal_circuit_lineage
from feature_genesis.circuits.report import render_circuit_report
from feature_genesis.config import load_config


CONFIG_PATH = Path("configs/quickdraw_dgx_v4.yaml")
PILOT_FEATURE_IDS = [89, 99, 164]


def main():
    config = load_config(CONFIG_PATH)
    accepted = set(accepted_target_feature_ids(config))
    feature_ids = [feature_id for feature_id in PILOT_FEATURE_IDS if feature_id in accepted]
    if not feature_ids:
        feature_ids = sorted(accepted)[:3]
    final_step = max(circuit_steps(config))
    discovery = _load_script("scripts/24_discover_sparse_circuits.py", "circuit_discovery_pilot")
    discovery.FEATURE_IDS = feature_ids
    discovery.CHECKPOINT_STEPS = [final_step]
    discovery.main()
    validation = _load_script("scripts/25_validate_sparse_circuits.py", "circuit_validation_pilot")
    validation.FEATURE_IDS = feature_ids
    validation.CHECKPOINT_STEPS = [final_step]
    validation.main()
    rows = []
    for feature_id in feature_ids:
        graph = json.loads(
            circuit_graph_path(config, feature_id, final_step).read_text(encoding="utf-8")
        )
        gate = json.loads(
            circuit_validation_path(config, feature_id, final_step).read_text(encoding="utf-8")
        )
        lineage = build_temporal_circuit_lineage(
            {final_step: graph},
            {final_step: gate},
            config,
        )
        circuit_lineage_path(config, feature_id).write_text(
            json.dumps(lineage, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        target_id = graph["target"]["node_id"]
        rows.append(
            {
                "feature_id": feature_id,
                "nodes": len(graph["nodes"]),
                "edges": len(graph["edges"]),
                "target_present": any(node["id"] == target_id for node in graph["nodes"]),
                "sufficiency_faithfulness": gate["scores"]["sufficiency_faithfulness"],
                "edge_sign_confirmation_fraction": gate["edge_sign_confirmation_fraction"],
                "passed": gate["passed"],
            }
        )
    result = {"passed": all(row["passed"] for row in rows), "rows": rows}
    path = circuit_root(config) / "pilot_gate.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(path)
    print(render_circuit_report(config))
    if not result["passed"]:
        raise RuntimeError("Sparse circuit pilot failed; do not launch the full temporal sweep")


def _load_script(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


if __name__ == "__main__":
    main()
