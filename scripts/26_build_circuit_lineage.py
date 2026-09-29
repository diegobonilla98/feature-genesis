import json
from pathlib import Path

from feature_genesis.circuits.lineage import build_temporal_circuit_lineage
from feature_genesis.circuits.paths import (
    accepted_target_feature_ids,
    circuit_graph_path,
    circuit_lineage_path,
    circuit_steps,
    circuit_validation_path,
)
from feature_genesis.config import load_config


CONFIG_PATH = Path("configs/quickdraw_dgx_v4.yaml")
FEATURE_IDS = None


def main():
    config = load_config(CONFIG_PATH)
    feature_ids = FEATURE_IDS or accepted_target_feature_ids(config)
    for feature_id in feature_ids:
        graphs = {}
        validations = {}
        for step in circuit_steps(config):
            graph_path = circuit_graph_path(config, feature_id, step)
            if not graph_path.exists():
                continue
            graph = json.loads(graph_path.read_text(encoding="utf-8"))
            graphs[step] = graph
            validation_path = circuit_validation_path(config, feature_id, step)
            if validation_path.exists():
                validations[step] = json.loads(validation_path.read_text(encoding="utf-8"))
        lineage = build_temporal_circuit_lineage(graphs, validations, config)
        path = circuit_lineage_path(config, feature_id)
        path.write_text(json.dumps(lineage, indent=2, sort_keys=True), encoding="utf-8")
        print(path)


if __name__ == "__main__":
    main()
