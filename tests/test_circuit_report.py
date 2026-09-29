import json
from pathlib import Path

from feature_genesis.circuits.paths import (
    circuit_graph_path,
    circuit_lineage_path,
    circuit_validation_path,
)
from feature_genesis.circuits.report import render_circuit_report
from feature_genesis.config import load_config


CONFIG_PATH = Path("configs/quickdraw_dgx_v4.yaml")


def test_circuit_report_contains_interactive_graph_and_causal_metrics(tmp_path):
    config = load_config(CONFIG_PATH)
    config.project.run_dir = str(tmp_path / "run")
    config.activations.checkpoint_steps = [0]
    config.circuit.checkpoint_steps = [0]
    config.circuit.target_feature_ids = [89]
    graph = {
        "model_step": 0,
        "layers": ["layer1", "layer2"],
        "target": {
            "layer": "layer2",
            "feature_id": 89,
            "node_id": "layer2:feature:89",
            "selected_logit": {"class_id": 3},
        },
        "probe": {"target_activation_mean": 1.0},
        "nodes": [
            {"id": "layer2:feature:89", "kind": "sae_feature", "layer": "layer2", "feature_id": 89, "forced_target": True},
            {"id": "logit:3", "kind": "logit", "class_id": 3, "label": "cat"},
        ],
        "edges": [
            {"source": "layer2:feature:89", "target": "logit:3", "attribution_mean": 1.0, "bootstrap_ci": [0.8, 1.2], "sign_agreement": 1.0}
        ],
        "graph_summary": {"nodes": 2, "edges": 1, "total_absolute_attribution": 1.0},
    }
    validation = {
        "passed": True,
        "scores": {"sufficiency_faithfulness": 0.9},
        "edge_sign_confirmation_fraction": 1.0,
    }
    lineage = {
        "final_step": 0,
        "emergence_step": 0,
        "trajectory": [
            {
                "step": 0,
                "phase": "mature",
                "nodes": 2,
                "edges": 1,
                "final_edge_similarity": 1.0,
                "sufficiency_faithfulness": 0.9,
                "emergence_confidence": 0.9,
            }
        ],
    }
    for path, payload in [
        (circuit_graph_path(config, 89, 0), graph),
        (circuit_validation_path(config, 89, 0), validation),
        (circuit_lineage_path(config, 89), lineage),
    ]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")
    report = render_circuit_report(config)
    feature_page = report.parent / "feature_00089" / "index.html"
    assert report.exists()
    text = feature_page.read_text(encoding="utf-8")
    assert "Circuit explorer" in text
    assert "Final faithfulness" in text
    assert "const graphs=" in text
