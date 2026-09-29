from pathlib import Path

from feature_genesis.config import load_config
from feature_genesis.evaluation.feature_dossier import write_feature_dossiers
from feature_genesis.sae.io import sae_analysis_root, sae_root
from feature_genesis.sae.validation import require_stage_four_gate

CONFIG_PATH = Path("configs/quickdraw_dgx_v4.yaml")
SAE_KIND = "topk"


def main():
    config = load_config(CONFIG_PATH)
    require_stage_four_gate(config)
    run_dir = Path(config.project.run_dir)
    final_step = max(config.activations.checkpoint_steps)
    evaluation_root = sae_analysis_root(
        config,
        "feature_evaluation",
        final_step,
        SAE_KIND,
    )
    lineage_path = (
        sae_analysis_root(config, "lineage", kind=SAE_KIND)
        / "backward_feature_trajectories.json"
    )
    attribution_root = sae_analysis_root(config, "attribution", kind=SAE_KIND)
    feature_quality_path = (
        sae_analysis_root(config, "evaluation", final_step, SAE_KIND)
        / "feature_quality.json"
    )
    sae_summary_path = (
        sae_root(
            config,
            final_step,
            SAE_KIND,
            config.sae.primary_seed,
            "continual",
        )
        / "summary.json"
    )
    path = write_feature_dossiers(
        run_dir,
        evaluation_root,
        lineage_path,
        attribution_root,
        feature_quality_path,
        sae_summary_path,
        run_dir / config.evaluation.report_dir / "features",
    )
    print(path)


if __name__ == "__main__":
    main()
