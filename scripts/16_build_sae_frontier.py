import json
from pathlib import Path

from feature_genesis.activations import ActivationCache
from feature_genesis.config import load_config
from feature_genesis.data.factory import build_datasets, eval_loader
from feature_genesis.evaluation.fidelity import evaluate_sae_substitution
from feature_genesis.progress import progress
from feature_genesis.sae.io import load_sae
from feature_genesis.sae.trainer import train_sae
from feature_genesis.training.replay import reconstruct_state

CONFIG_PATH = Path("configs/quickdraw_dgx_v4.yaml")
MODEL_STEP = None
MAX_FIDELITY_IMAGES = 600
RUNS = [
    {"kind": "batch_topk", "k": 8},
    {"kind": "batch_topk", "k": 16},
    {"kind": "batch_topk", "k": 32},
    {"kind": "cosine_batch_topk", "k": 8},
    {"kind": "cosine_batch_topk", "k": 16},
    {"kind": "cosine_batch_topk", "k": 32},
    {"kind": "matryoshka", "k": 8},
    {"kind": "matryoshka", "k": 16},
    {"kind": "matryoshka", "k": 32},
    {"kind": "jumprelu", "jump_sparsity_coefficient": 0.0003},
    {"kind": "jumprelu", "jump_sparsity_coefficient": 0.001},
    {"kind": "jumprelu", "jump_sparsity_coefficient": 0.003},
    {"kind": "jumprelu", "jump_sparsity_coefficient": 0.01},
    {"kind": "matching_pursuit", "mp_steps": 8},
    {"kind": "matching_pursuit", "mp_steps": 16},
    {"kind": "matching_pursuit", "mp_steps": 32},
]


def main():
    base = load_config(CONFIG_PATH)
    base_run_dir = Path(base.project.run_dir)
    step = max(base.activations.checkpoint_steps) if MODEL_STEP is None else MODEL_STEP
    activation_root = base_run_dir / base.activations.cache_dir / f"step_{step:08d}_{base.activations.split}"
    ActivationCache.load(activation_root)
    bundle = build_datasets(base, include_annotations=False, canonical_train=True)
    model, _, _, _ = reconstruct_state(base, step, bundle=bundle)
    device = next(model.parameters()).device
    rows = []
    for specification in progress(RUNS, desc="sae frontier"):
        config = base.model_copy(deep=True)
        kind = specification["kind"]
        for key, value in specification.items():
            if key != "kind":
                setattr(config.sae, key, value)
        suffix = "_".join(f"{key}-{value}" for key, value in specification.items())
        config.project.run_dir = str(base_run_dir / "sae_frontier" / suffix)
        summary = train_sae(config, activation_root, step, kind)
        sae, _ = load_sae(config, summary["path"], device)
        fidelity = evaluate_sae_substitution(
            model,
            sae,
            eval_loader(bundle.validation, base, MAX_FIDELITY_IMAGES),
            base.model.hook_layer,
            device,
            MAX_FIDELITY_IMAGES,
        )
        rows.append({"specification": specification, "sae": summary, "fidelity": fidelity})
    output = {
        "model_step": step,
        "comparison_rule": "Compare FVE and loss_recovered at matched realized mean_l0",
        "runs": rows,
    }
    path = base_run_dir / "sae_frontier.json"
    path.write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
