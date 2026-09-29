from __future__ import annotations

import json
import re
from pathlib import Path

from feature_genesis.config import Config
from feature_genesis.sae.io import latest_sae_path


def layer_slug(layer: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", layer).strip("_").lower()


def circuit_root(config: Config) -> Path:
    return Path(config.project.run_dir) / "circuits" / config.circuit.artifact_version


def circuit_steps(config: Config) -> list[int]:
    values = config.circuit.checkpoint_steps or config.activations.checkpoint_steps
    return sorted(set(int(value) for value in values))


def circuit_cache_root(config: Config, layer: str, step: int, split: str | None = None) -> Path:
    resolved_split = config.circuit.cache_split if split is None else split
    if layer == config.circuit.target_layer and config.circuit.reuse_target_layer_sae:
        return (
            Path(config.project.run_dir)
            / config.activations.cache_dir
            / f"step_{step:08d}_{resolved_split}"
        )
    return (
        circuit_root(config)
        / "activations"
        / layer_slug(layer)
        / f"step_{step:08d}_{resolved_split}"
    )


def circuit_layer_config(config: Config, layer: str) -> Config:
    resolved = config.model_copy(deep=True)
    slug = layer_slug(layer)
    resolved.model.hook_layer = layer
    resolved.activations.split = config.circuit.cache_split
    resolved.activations.max_images = config.circuit.cache_images
    resolved.activations.batch_size = config.circuit.cache_batch_size
    resolved.activations.random_spatial_tokens_per_image = config.circuit.spatial_tokens_per_image
    resolved.activations.cache_dir = str(
        Path("circuits")
        / config.circuit.artifact_version
        / "activations"
        / slug
    )
    resolved.activations.checkpoint_steps = circuit_steps(config)
    resolved.sae.kind = config.circuit.sae_kind
    resolved.sae.dictionary_size = config.circuit.sae_dictionary_size_by_layer.get(layer)
    resolved.sae.expansion_factor = config.circuit.sae_expansion_factor
    resolved.sae.k = config.circuit.sae_k_by_layer.get(layer, config.circuit.sae_k)
    dictionary_slug = (
        f"d{resolved.sae.dictionary_size}"
        if resolved.sae.dictionary_size is not None
        else f"x{resolved.sae.expansion_factor}"
    )
    resolved.sae.artifact_version = (
        f"{config.circuit.artifact_version}_{slug}_{dictionary_slug}_k{resolved.sae.k}"
    )
    resolved.sae.steps = config.circuit.sae_steps_by_layer.get(layer, config.circuit.sae_steps)
    resolved.sae.batch_size = config.circuit.sae_batch_size_by_layer.get(
        layer,
        config.circuit.sae_batch_size,
    )
    resolved.sae.initialization_samples = config.circuit.sae_initialization_samples
    return resolved


def circuit_sae_path(config: Config, layer: str, step: int) -> Path:
    if layer == config.circuit.target_layer and config.circuit.reuse_target_layer_sae:
        return latest_sae_path(config, step, config.circuit.sae_kind)
    layer_config = circuit_layer_config(config, layer)
    return latest_sae_path(layer_config, step, config.circuit.sae_kind)


def feature_circuit_root(config: Config, feature_id: int) -> Path:
    return circuit_root(config) / "features" / f"feature_{feature_id:05d}"


def circuit_graph_path(config: Config, feature_id: int, step: int) -> Path:
    return feature_circuit_root(config, feature_id) / f"step_{step:08d}" / "graph.json"


def circuit_validation_path(config: Config, feature_id: int, step: int) -> Path:
    return feature_circuit_root(config, feature_id) / f"step_{step:08d}" / "validation.json"


def circuit_lineage_path(config: Config, feature_id: int) -> Path:
    return feature_circuit_root(config, feature_id) / "lineage.json"


def accepted_target_feature_ids(config: Config) -> list[int]:
    if config.circuit.target_feature_ids:
        return sorted(set(config.circuit.target_feature_ids))
    step = max(config.activations.checkpoint_steps)
    root = (
        Path(config.project.run_dir)
        / "feature_evaluation"
        / f"step_{step:08d}"
        / config.sae.kind
        / config.sae.artifact_version
        / "continual"
        / f"seed_{config.sae.primary_seed:04d}"
        / "gemini"
    )
    feature_ids = []
    for path in sorted(root.glob("feature_*_evaluation.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        if row.get("validated_status") == "accepted":
            feature_ids.append(int(row["feature_id"]))
    return feature_ids
