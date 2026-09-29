import json
from pathlib import Path

from feature_genesis.circuits.paths import (
    accepted_target_feature_ids,
    circuit_graph_path,
    circuit_lineage_path,
    circuit_root,
    circuit_steps,
    circuit_validation_path,
    feature_circuit_root,
)
from feature_genesis.config import load_config


CONFIG_PATH = Path("configs/quickdraw_dgx_v4.yaml")
REQUIRE_TRAINING_CAUSALITY = True


def main():
    config = load_config(CONFIG_PATH)
    checks = []
    for feature_id in accepted_target_feature_ids(config):
        graphs = [circuit_graph_path(config, feature_id, step) for step in circuit_steps(config)]
        checks.append(_check(f"feature_{feature_id:05d}_all_graphs", all(path.exists() for path in graphs), len([path for path in graphs if path.exists()])))
        final_step = max(circuit_steps(config))
        final_graph_path = circuit_graph_path(config, feature_id, final_step)
        final_graph = json.loads(final_graph_path.read_text(encoding="utf-8")) if final_graph_path.exists() else {}
        checks.append(_check(f"feature_{feature_id:05d}_final_graph_nonempty", bool(final_graph.get("nodes")) and bool(final_graph.get("edges")), final_graph.get("graph_summary")))
        validation_path = circuit_validation_path(config, feature_id, final_step)
        validation = json.loads(validation_path.read_text(encoding="utf-8")) if validation_path.exists() else {}
        checks.append(_check(f"feature_{feature_id:05d}_final_causal_gate", bool(validation.get("passed")), validation.get("scores")))
        lineage_path = circuit_lineage_path(config, feature_id)
        lineage = json.loads(lineage_path.read_text(encoding="utf-8")) if lineage_path.exists() else {}
        checks.append(_check(f"feature_{feature_id:05d}_emergence_resolved", lineage.get("emergence_step") is not None, lineage.get("emergence_interval")))
        if REQUIRE_TRAINING_CAUSALITY:
            root = feature_circuit_root(config, feature_id)
            causal = list(root.glob("counterfactual_*_rank_*.json"))
            confirmed = sum(json.loads(path.read_text(encoding="utf-8")).get("predicted_direction_confirmed", False) for path in causal)
            checks.append(_check(f"feature_{feature_id:05d}_training_replays", len(causal) >= 2, len(causal)))
            checks.append(_check(f"feature_{feature_id:05d}_confirmed_training_event", confirmed >= 1, confirmed))
    result = {
        "schema_version": "sparse-visual-circuit-artifact-gate-v1",
        "passed": all(row["passed"] for row in checks),
        "checks": checks,
    }
    path = circuit_root(config) / "artifact_validation.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(path)
    if not result["passed"]:
        failed = [row["name"] for row in checks if not row["passed"]]
        raise RuntimeError(f"Circuit scientific artifact gate failed: {failed}")


def _check(name, passed, value):
    return {"name": name, "passed": bool(passed), "value": value}


if __name__ == "__main__":
    main()
