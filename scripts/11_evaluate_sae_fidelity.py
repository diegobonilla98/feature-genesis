import json
from pathlib import Path

from feature_genesis.config import load_config
from feature_genesis.data.factory import build_datasets, eval_loader
from feature_genesis.evaluation.fidelity import evaluate_sae_substitution
from feature_genesis.sae.io import latest_sae_path, load_sae, sae_analysis_root
from feature_genesis.sae.validation import require_stage_four_gate
from feature_genesis.training.replay import reconstruct_state

CONFIG_PATH = Path("configs/quickdraw_dgx_v4.yaml")
SAE_KIND = "topk"
MODEL_STEP = None
MAX_IMAGES = None


def main():
    config = load_config(CONFIG_PATH)
    require_stage_four_gate(config)
    step = max(config.activations.checkpoint_steps) if MODEL_STEP is None else MODEL_STEP
    bundle = build_datasets(config, include_annotations=False, canonical_train=True)
    model, _, _, _ = reconstruct_state(config, step, bundle=bundle)
    device = next(model.parameters()).device
    sae, _ = load_sae(config, latest_sae_path(config, step, SAE_KIND), device)
    loader = eval_loader(bundle.validation, config, MAX_IMAGES)
    metrics = evaluate_sae_substitution(model, sae, loader, config.model.hook_layer, device, MAX_IMAGES)
    root = sae_analysis_root(config, "evaluation", step, SAE_KIND)
    root.mkdir(parents=True, exist_ok=True)
    path = root / "model_fidelity.json"
    path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
