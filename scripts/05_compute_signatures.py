from pathlib import Path

from feature_genesis.activations import ActivationCache
from feature_genesis.config import load_config
from feature_genesis.determinism import resolve_device
from feature_genesis.lineage.signatures import compute_feature_signatures
from feature_genesis.progress import progress
from feature_genesis.sae.io import latest_sae_path, load_sae, sae_analysis_root
from feature_genesis.sae.validation import require_stage_four_gate

CONFIG_PATH = Path("configs/quickdraw_dgx_v4.yaml")
SAE_KIND = "topk"
PROBE_ROWS = 4096
PROJECTION_DIM = 256


def main():
    config = load_config(CONFIG_PATH)
    require_stage_four_gate(config)
    device = resolve_device(config.project.device)
    for step in progress(config.activations.checkpoint_steps, desc="compute signatures"):
        cache_root = Path(config.project.run_dir) / config.activations.cache_dir / f"step_{step:08d}_{config.activations.split}"
        cache = ActivationCache.load(cache_root)
        sae, _ = load_sae(config, latest_sae_path(config, step, SAE_KIND), device)
        signatures = compute_feature_signatures(
            sae,
            cache,
            device,
            config.activations.batch_size,
            config.project.seed,
            PROBE_ROWS,
            PROJECTION_DIM,
            config.evaluation.top_examples,
        )
        root = sae_analysis_root(config, "signatures", step, SAE_KIND)
        print(signatures.save(root))


if __name__ == "__main__":
    main()
