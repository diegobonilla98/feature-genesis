import json
from pathlib import Path

from feature_genesis.config import load_config
from feature_genesis.data.factory import build_datasets, eval_loader
from feature_genesis.evaluation.fidelity import evaluate_sae_substitution
from feature_genesis.sae.io import load_sae, sae_root
from feature_genesis.sae.trainer import train_continual_sae_sequence, train_sae
from feature_genesis.sae.validation import validate_final_candidate, validate_sae_artifact
from feature_genesis.training.replay import reconstruct_state

CONFIG_PATH = Path("configs/quickdraw_dgx_v4.yaml")
CONTINUAL_KIND = "topk"
FINAL_BENCHMARK_KINDS = ["topk"]
PILOT_STEPS = [0, 3200]
FIDELITY_IMAGES = 2048
MINIMUM_LOSS_RECOVERED = 0.99
MINIMUM_ACCURACY_CHANGE = -0.01


def main():
    config = load_config(CONFIG_PATH)
    roots = {
        step: Path(config.project.run_dir) / config.activations.cache_dir / f"step_{step:08d}_{config.activations.split}"
        for step in config.activations.checkpoint_steps
    }
    final_step = max(roots)
    for sae_seed in config.sae.seeds:
        print(
            train_sae(
                config,
                roots[final_step],
                final_step,
                CONTINUAL_KIND,
                sae_seed=sae_seed,
                role="independent",
            )
        )
    candidate = validate_final_candidate(config, CONTINUAL_KIND)
    print(candidate["path"])
    if not candidate["passed"]:
        raise RuntimeError(
            "Final-checkpoint SAE candidate failed its scientific gate; "
            "the expensive continual sequence was not started"
        )
    bundle = build_datasets(config, include_annotations=False, canonical_train=True)
    model, _, _, _ = reconstruct_state(config, final_step, bundle=bundle)
    device = next(model.parameters()).device
    sae, _ = load_sae(
        config,
        sae_root(
            config,
            final_step,
            CONTINUAL_KIND,
            config.sae.primary_seed,
            "independent",
        )
        / "final.pt",
        device,
    )
    fidelity = evaluate_sae_substitution(
        model,
        sae,
        eval_loader(bundle.validation, config, FIDELITY_IMAGES),
        config.model.hook_layer,
        device,
        FIDELITY_IMAGES,
    )
    fidelity_path = (
        Path(config.project.run_dir)
        / "sae_validation"
        / config.sae.artifact_version
        / "final_candidate_fidelity.json"
    )
    fidelity_path.write_text(json.dumps(fidelity, indent=2, sort_keys=True), encoding="utf-8")
    print(fidelity_path)
    if (
        fidelity["loss_recovered"] < MINIMUM_LOSS_RECOVERED
        or fidelity["accuracy_change"] < MINIMUM_ACCURACY_CHANGE
    ):
        raise RuntimeError("Final-checkpoint SAE candidate failed task-fidelity requirements")
    for step in PILOT_STEPS:
        if step not in roots:
            continue
        print(
            train_sae(
                config,
                roots[step],
                step,
                CONTINUAL_KIND,
                sae_seed=config.sae.primary_seed,
                role="continual",
            )
        )
        pilot = validate_sae_artifact(
            config,
            sae_root(
                config,
                step,
                CONTINUAL_KIND,
                config.sae.primary_seed,
                "continual",
            ),
            step,
            config.sae.primary_seed,
            "continual",
        )
        if not pilot["passed"]:
            failed = [check["name"] for check in pilot["checks"] if not check["passed"]]
            raise RuntimeError(f"Checkpointwise SAE pilot failed at model step {step}: {failed}")
    for sae_seed in config.sae.seeds:
        print(
            train_continual_sae_sequence(
                config,
                roots,
                CONTINUAL_KIND,
                sae_seed=sae_seed,
            )
        )
    for kind in FINAL_BENCHMARK_KINDS:
        seeds = config.sae.seeds if kind == CONTINUAL_KIND else [config.sae.primary_seed]
        for sae_seed in seeds:
            if kind == CONTINUAL_KIND:
                continue
            print(
                train_sae(
                    config,
                    roots[final_step],
                    final_step,
                    kind,
                    sae_seed=sae_seed,
                    role="independent",
                )
            )


if __name__ == "__main__":
    main()
