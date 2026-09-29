from types import SimpleNamespace

import torch
from torch import nn

from feature_genesis.circuits.discovery import (
    CircuitProbeSet,
    discover_sparse_circuit,
    refresh_probe_target_values,
)
from feature_genesis.circuits.validation import validate_sparse_circuit
from feature_genesis.sae.topk import TopKSAE


class ToyCircuitModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Conv2d(1, 2, 1, bias=False)
        self.layer2 = nn.Conv2d(2, 2, 1, bias=False)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            self.layer1.weight.copy_(torch.tensor([[[[1.0]]], [[[0.2]]]]))
            self.layer2.weight.copy_(torch.tensor([[[[1.0]], [[0.0]]], [[[0.0]], [[1.0]]]]))
            self.fc.weight.copy_(torch.tensor([[2.0, 0.0], [0.0, 1.0]]))

    def forward(self, images):
        values = torch.relu(self.layer1(images))
        values = torch.relu(self.layer2(values))
        return self.fc(self.pool(values).flatten(1))


def identity_sae():
    sae = TopKSAE(2, 2, 2, 0.0, 2, 10)
    with torch.no_grad():
        sae.encoder.copy_(torch.eye(2))
        sae.decoder.copy_(torch.eye(2))
        sae.encoder_bias.zero_()
        sae.decoder_bias.zero_()
        sae.input_scale.fill_(1)
    return sae


def config():
    return SimpleNamespace(
        project=SimpleNamespace(seed=7),
        circuit=SimpleNamespace(
            discovery_batch_size=2,
            max_nodes_per_layer=2,
            max_edges_per_target=2,
            pruning_node_counts=[1, 2],
            minimum_edge_attribution=0.0,
            bootstrap_samples=20,
            bootstrap_confidence=0.9,
            validation_edges=4,
            minimum_faithfulness=0.5,
            minimum_edge_sign_agreement=0.5,
        ),
    )


def probes():
    images = torch.tensor([[[[1.0]]], [[[2.0]]], [[[3.0]]], [[[4.0]]]])
    return CircuitProbeSet(
        images=images,
        labels=torch.zeros(4, dtype=torch.long),
        dataset_indices=torch.arange(4),
        target_values=torch.tensor([1.0, 2.0, 3.0, 4.0]),
        scanned_images=4,
    )


def test_discovery_builds_connected_sparse_feature_graph():
    model = ToyCircuitModel()
    saes = {"layer1": identity_sae(), "layer2": identity_sae()}
    graph = discover_sparse_circuit(
        model,
        saes,
        probes(),
        ["layer1", "layer2"],
        "layer1",
        0,
        10,
        config(),
        ["zero", "one"],
    )
    assert graph["target"]["feature_id"] == 0
    assert graph["target"]["selected_logit"]["class_id"] == 0
    assert any(node.get("forced_target") for node in graph["nodes"])
    assert any(edge["target"] == "logit:0" for edge in graph["edges"])
    assert all("bootstrap_ci" in edge for edge in graph["edges"])


def test_discovered_graph_can_be_causally_validated():
    model = ToyCircuitModel()
    saes = {"layer1": identity_sae(), "layer2": identity_sae()}
    graph = discover_sparse_circuit(
        model,
        saes,
        probes(),
        ["layer1", "layer2"],
        "layer1",
        0,
        10,
        config(),
        ["zero", "one"],
    )
    result = validate_sparse_circuit(model, saes, probes(), graph, config())
    assert result["scores"]["sufficiency_faithfulness"] > 0.5
    assert result["target_feature_intervention"]["absolute_centered_logit_effect"] > 0
    assert result["testable_edges"] > 0


def test_fixed_probe_cohort_can_be_rescored_without_reselection():
    model = ToyCircuitModel()
    sae = identity_sae()
    rescored = refresh_probe_target_values(
        model,
        sae,
        probes(),
        "layer1",
        0,
        2,
        torch.device("cpu"),
    )
    assert torch.allclose(rescored.target_values, torch.tensor([1.0, 2.0, 3.0, 4.0]))
    assert torch.equal(rescored.dataset_indices, probes().dataset_indices)
