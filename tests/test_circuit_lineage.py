from types import SimpleNamespace

from feature_genesis.circuits.lineage import (
    build_temporal_circuit_lineage,
    weighted_edge_jaccard,
)


def graph(step, weight):
    return {
        "model_step": step,
        "target": {"feature_id": 1},
        "nodes": [{"id": "a"}, {"id": "b"}],
        "edges": [
            {
                "source": "a",
                "target": "b",
                "attribution_mean": weight,
                "bootstrap_ci": [weight * 0.8, weight * 1.2],
                "sign_agreement": 1.0,
            }
        ],
        "probe": {"target_activation_mean": weight},
        "graph_summary": {"total_absolute_attribution": abs(weight)},
    }


def validation(step, passed):
    return {
        "model_step": step,
        "passed": passed,
        "scores": {"sufficiency_faithfulness": 0.9 if passed else 0.2},
        "edge_sign_confirmation_fraction": 1.0 if passed else 0.0,
        "target_feature_intervention": {"absolute_centered_logit_effect": 0.5},
    }


def test_weighted_edge_jaccard_uses_edge_mass():
    assert weighted_edge_jaccard({"a": 1.0}, {"a": 2.0}) == 0.5
    assert weighted_edge_jaccard({}, {}) == 1.0


def test_emergence_requires_persistence():
    config = SimpleNamespace(
        circuit=SimpleNamespace(
            minimum_edge_sign_agreement=0.75,
            minimum_final_edge_similarity=0.5,
            minimum_faithfulness=0.8,
            persistence_checkpoints=2,
        )
    )
    graphs = {0: graph(0, 0.1), 10: graph(10, 0.8), 20: graph(20, 1.0)}
    validations = {0: validation(0, False), 10: validation(10, True), 20: validation(20, True)}
    result = build_temporal_circuit_lineage(graphs, validations, config)
    assert result["emergence_step"] == 10
    assert result["emergence_interval"] == [0, 10]
    assert result["trajectory"][-1]["phase"] == "mature"
