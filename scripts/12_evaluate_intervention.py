import json
from pathlib import Path

from feature_genesis.config import load_config
from feature_genesis.data.factory import build_datasets, eval_loader
from feature_genesis.evaluation.interventions import evaluate_feature_intervention
from feature_genesis.sae.io import latest_sae_path, load_sae, sae_analysis_root
from feature_genesis.sae.validation import require_stage_four_gate
from feature_genesis.training.replay import reconstruct_state

CONFIG_PATH = Path("configs/quickdraw_dgx_v4.yaml")
SAE_KIND = "topk"
MODEL_STEP = None
MODES = ["ablate", "steer"]
AMOUNTS = {"ablate": 1.0, "steer": 1.0}
MAX_IMAGES = 512
REUSE_EXISTING = True


def main():
    config = load_config(CONFIG_PATH)
    require_stage_four_gate(config)
    step = max(config.activations.checkpoint_steps) if MODEL_STEP is None else MODEL_STEP
    bundle = build_datasets(config, include_annotations=False, canonical_train=True)
    model, _, _, _ = reconstruct_state(config, step, bundle=bundle)
    device = next(model.parameters()).device
    sae, _ = load_sae(config, latest_sae_path(config, step, SAE_KIND), device)
    root = sae_analysis_root(config, "evaluation", step, SAE_KIND) / "interventions"
    root.mkdir(parents=True, exist_ok=True)
    for feature_id in accepted_feature_ids(config, step):
        for mode in MODES:
            path = root / f"feature_{feature_id:05d}_{mode}.json"
            if REUSE_EXISTING and path.exists():
                print(path)
                continue
            loader = eval_loader(bundle.validation, config, MAX_IMAGES)
            metrics = evaluate_feature_intervention(
                model,
                sae,
                loader,
                config.model.hook_layer,
                feature_id,
                device,
                mode,
                AMOUNTS[mode],
                MAX_IMAGES,
            )
            path.write_text(
                json.dumps(metrics, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            print(path)


def accepted_feature_ids(config, step):
    root = sae_analysis_root(config, "feature_evaluation", step, SAE_KIND) / "gemini"
    feature_ids = []
    for path in sorted(root.glob("feature_*_evaluation.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        if row["validated_status"] == "accepted":
            feature_ids.append(int(row["feature_id"]))
    return feature_ids


if __name__ == "__main__":
    main()
