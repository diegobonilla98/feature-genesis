import json
from pathlib import Path

from feature_genesis.config import load_config
from feature_genesis.sae.io import sae_analysis_root
from feature_genesis.sae.validation import require_stage_four_gate

CONFIG_PATH = Path("configs/quickdraw_dgx_v4.yaml")
SAE_KIND = "topk"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    config = load_config(CONFIG_PATH)
    gate = require_stage_four_gate(config)
    run_dir = Path(config.project.run_dir)
    final_step = max(config.activations.checkpoint_steps)
    gemini_root = sae_analysis_root(
        config,
        "feature_evaluation",
        final_step,
        SAE_KIND,
    ) / "gemini"
    lineage_path = (
        sae_analysis_root(config, "lineage", kind=SAE_KIND)
        / "backward_feature_trajectories.json"
    )
    attribution_root = sae_analysis_root(config, "attribution", kind=SAE_KIND)
    lineage = read_json(lineage_path)
    adjudication = read_json(run_dir / "manual_adjudication.json")
    causal_path = attribution_root / "causal_validation_summary.json"
    causal = read_json(causal_path) if causal_path.exists() else None
    decisions = {
        int(row["feature_id"]): row for row in adjudication["features"]
    }
    causal_by_feature = {}
    if causal is not None:
        for row in causal["records"]:
            causal_by_feature.setdefault(int(row["feature_id"]), []).append(row)
    features = []
    evaluation_paths = sorted(gemini_root.glob("feature_*_evaluation.json"))
    for evaluation_path in evaluation_paths:
        evaluation = read_json(evaluation_path)
        feature_id = int(evaluation["feature_id"])
        label = evaluation["label"]
        heldout = evaluation["heldout_validation"]["metrics"]
        ancestry = lineage["strongest_ancestry"].get(str(feature_id))
        phase = lineage["phases"].get(str(feature_id))
        influence_path = (
            attribution_root
            / f"feature_{feature_id:05d}"
            / "training_influence_manifest.json"
        )
        intervention_path = (
            sae_analysis_root(config, "evaluation", final_step, SAE_KIND)
            / "interventions"
            / f"feature_{feature_id:05d}_ablate.json"
        )
        steering_path = intervention_path.with_name(
            f"feature_{feature_id:05d}_steer.json"
        )
        decision = decisions.get(feature_id, {})
        features.append(
            {
                "feature_id": feature_id,
                "status": evaluation["validated_status"],
                "acceptance_mode": evaluation["acceptance_mode"],
                "manual_decision": decision.get("decision"),
                "canonical_label": label["canonical_label"],
                "plain_english_summary": label["plain_english_summary"],
                "description": label["description"],
                "trigger_conditions": label["trigger_conditions"],
                "non_trigger_conditions": label["non_trigger_conditions"],
                "do_not_conflate_with": label["do_not_conflate_with"],
                "falsification_tests": label["falsification_tests"],
                "visual_signature": label["visual_signature"],
                "heldout_validation": heldout,
                "strongest_edge_ancestry": ancestry,
                "phase_estimate": phase,
                "causal_training_examples": causal_by_feature.get(feature_id, []),
                "final_model_ablation": (
                    read_json(intervention_path) if intervention_path.exists() else None
                ),
                "final_model_steering": (
                    read_json(steering_path) if steering_path.exists() else None
                ),
                "training_influence_manifest": (
                    str(influence_path) if influence_path.exists() else None
                ),
                "source_evaluation": str(evaluation_path),
            }
        )
    accepted = [row for row in features if row["status"] == "accepted"]
    rejected = [row for row in features if row["status"] != "accepted"]
    summary = {
        "schema_version": 1,
        "run_id": run_dir.name,
        "final_model_step": final_step,
        "sae_kind": SAE_KIND,
        "sae_artifact_version": config.sae.artifact_version,
        "scientific_gate_passed": bool(gate.get("passed", True)),
        "accepted_feature_count": len(accepted),
        "rejected_or_control_count": len(rejected),
        "accepted_features": accepted,
        "rejected_or_control_features": rejected,
        "causal_interpretation": (
            causal["interpretation"] if causal is not None else None
        ),
        "claims_policy": {
            "structural_birth": "Earliest node connected to the final feature by the deterministic strongest incoming-edge path.",
            "phase_estimate": "Left-censored when decoder alignment is too unreliable before the measurement window.",
            "training_attribution": "Gradient influence ranks candidates but does not establish builder or breaker status.",
            "causal_role": "Builder or breaker status is assigned only from exact single-step masking and replay.",
        },
        "source_artifacts": {
            "manual_adjudication": str(run_dir / "manual_adjudication.json"),
            "lineage": str(lineage_path),
            "causal_validation": str(causal_path) if causal_path.exists() else None,
            "gemini_manifest": str(gemini_root / "run_manifest.json"),
        },
    }
    output_path = run_dir / "experiment_summary.json"
    output_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(output_path)


if __name__ == "__main__":
    main()
