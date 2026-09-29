import json
from pathlib import Path

from feature_genesis.activations import cache_checkpoint_activations
from feature_genesis.circuits.paths import (
    circuit_cache_root,
    circuit_layer_config,
    circuit_steps,
)
from feature_genesis.config import load_config
from feature_genesis.training.quality import require_classifier_gate


CONFIG_PATH = Path("configs/quickdraw_dgx_v4.yaml")
LAYERS = None
CHECKPOINT_STEPS = None
REUSE_EXISTING = True


def main():
    config = load_config(CONFIG_PATH)
    require_classifier_gate(config.project.run_dir)
    layers = LAYERS or config.circuit.layers
    steps = CHECKPOINT_STEPS or circuit_steps(config)
    for layer in layers:
        if layer == config.circuit.target_layer and config.circuit.reuse_target_layer_sae:
            print(f"reuse existing target-layer caches: {layer}")
            continue
        layer_config = circuit_layer_config(config, layer)
        for step in steps:
            root = circuit_cache_root(config, layer, step)
            if REUSE_EXISTING and _valid_cache(root, layer, step, config.circuit.cache_split):
                print(root)
                continue
            print(
                cache_checkpoint_activations(
                    layer_config,
                    step,
                    config.circuit.cache_split,
                    config.circuit.cache_images,
                    config.circuit.spatial_tokens_per_image,
                    replay_config=config,
                )
            )
        final_step = max(steps)
        evaluation_root = circuit_cache_root(
            config,
            layer,
            final_step,
            config.activations.evaluation_split,
        )
        if not (
            REUSE_EXISTING
            and _valid_cache(
                evaluation_root,
                layer,
                final_step,
                config.activations.evaluation_split,
            )
        ):
            print(
                cache_checkpoint_activations(
                    layer_config,
                    final_step,
                    config.activations.evaluation_split,
                    config.circuit.evaluation_cache_images,
                    0,
                    replay_config=config,
                )
            )


def _valid_cache(root, layer, step, split):
    metadata_path = root / "metadata.json"
    activation_path = root / "activations.npy"
    if not metadata_path.exists() or not activation_path.exists():
        return False
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return (
        metadata.get("layer") == layer
        and int(metadata.get("checkpoint_step", -1)) == step
        and metadata.get("split") == split
    )


if __name__ == "__main__":
    main()
