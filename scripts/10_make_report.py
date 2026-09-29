import json
from pathlib import Path

from feature_genesis.config import load_config
from feature_genesis.evaluation.report import write_research_report
from feature_genesis.sae.io import sae_analysis_root
from feature_genesis.sae.validation import require_stage_four_gate

CONFIG_PATH = Path("configs/quickdraw_dgx_v4.yaml")


def main():
    config = load_config(CONFIG_PATH)
    require_stage_four_gate(config)
    run_dir = Path(config.project.run_dir)
    final_step = max(config.activations.checkpoint_steps)
    gemini_root = (
        sae_analysis_root(
            config,
            "feature_evaluation",
            final_step,
            "topk",
        )
        / "gemini"
    )
    sections = []
    for title, path in [
        ("Artifact validation", run_dir / "artifact_validation.json"),
        ("Experiment summary", run_dir / "experiment_summary.json"),
        (
            "Per-feature genesis dossiers",
            run_dir / config.evaluation.report_dir / "features" / "manifest.json",
        ),
        ("Training result", run_dir / "result.json"),
        ("Replay verification", run_dir / "replay_verification.json"),
        (
            "Stage 4 scientific gate",
            run_dir
            / "sae_validation"
            / config.sae.artifact_version
            / "stage_04_gate.json",
        ),
        (
            "Lineage summary",
            sae_analysis_root(config, "lineage", kind="topk")
            / "summary.json",
        ),
        (
            "Concept metrics",
            sae_analysis_root(config, "evaluation", final_step, "topk")
            / "concept_metrics.json",
        ),
        ("Gemini multimodal feature evaluation", gemini_root / "run_manifest.json"),
        (
            "Causal training-example validation",
            sae_analysis_root(config, "attribution", kind="topk")
            / "causal_validation_summary.json",
        ),
    ]:
        if path.exists():
            sections.append((title, json.loads(path.read_text(encoding="utf-8"))))
    path = write_research_report(
        run_dir / config.evaluation.report_dir,
        "Feature Genesis Research Report",
        sections,
    )
    print(path)


if __name__ == "__main__":
    main()
