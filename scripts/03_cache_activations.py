from pathlib import Path

from feature_genesis.activations import cache_checkpoint_activations
from feature_genesis.config import load_config
from feature_genesis.training.quality import require_classifier_gate
from feature_genesis.progress import progress

CONFIG_PATH = Path("configs/quickdraw_dgx_v4.yaml")
CHECKPOINT_STEPS = None


def main():
    config = load_config(CONFIG_PATH)
    require_classifier_gate(config.project.run_dir)
    steps = CHECKPOINT_STEPS or config.activations.checkpoint_steps
    for step in progress(steps, desc="cache activations"):
        print(
            cache_checkpoint_activations(
                config,
                step,
                config.activations.split,
                config.activations.max_images,
                config.activations.random_spatial_tokens_per_image,
            )
        )
        if (
            config.activations.evaluation_split != config.activations.split
            and step == max(steps)
        ):
            print(
                cache_checkpoint_activations(
                    config,
                    step,
                    config.activations.evaluation_split,
                    config.activations.evaluation_max_images,
                    0,
                )
            )


if __name__ == "__main__":
    main()
