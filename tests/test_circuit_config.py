from pathlib import Path

from feature_genesis.circuits.paths import (
    circuit_cache_root,
    circuit_layer_config,
    circuit_steps,
)
from feature_genesis.config import load_config


CONFIG_PATH = Path("configs/quickdraw_dgx_v4.yaml")


def test_circuit_config_is_namespaced_and_reuses_target_layer():
    config = load_config(CONFIG_PATH)
    assert config.circuit.layers == ["layer1", "layer2", "layer3", "layer4"]
    assert config.circuit.target_layer == "layer2"
    assert len(circuit_steps(config)) == len(config.activations.checkpoint_steps)
    layer3 = circuit_layer_config(config, "layer3")
    assert layer3.model.hook_layer == "layer3"
    assert layer3.sae.dictionary_size == 1024
    assert layer3.sae.k == 12
    assert layer3.sae.artifact_version == "v1_sparse_visual_layer3_d1024_k12"
    assert "circuits/v1_sparse_visual" in layer3.activations.cache_dir
    layer1 = circuit_layer_config(config, "layer1")
    assert layer1.sae.k == 6
    assert layer1.sae.artifact_version == "v1_sparse_visual_layer1_d256_k6"
    layer4 = circuit_layer_config(config, "layer4")
    assert layer4.sae.k == 48
    assert layer4.sae.artifact_version == "v1_sparse_visual_layer4_d1024_k48"
    assert layer4.sae.maximum_positive_decoder_cosine == 0.98
    target = circuit_cache_root(config, "layer2", max(circuit_steps(config)))
    assert target.parent.name == "activations"
