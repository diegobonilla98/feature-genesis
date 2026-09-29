import json
from pathlib import Path

from feature_genesis.config import load_config
from feature_genesis.sae.io import sae_analysis_root
from feature_genesis.sae.validation import require_stage_four_gate


CONFIG_PATH = Path("configs/quickdraw_dgx_v4.yaml")
SAE_KIND = "topk"


def main():
    config = load_config(CONFIG_PATH)
    require_stage_four_gate(config)
    records = []
    for feature_id in accepted_feature_ids(config):
        feature_root = (
            sae_analysis_root(config, "attribution", kind=SAE_KIND)
            / f"feature_{feature_id:05d}"
        )
        ranking = json.loads(
            (feature_root / "gradient_ranked_batches.json").read_text(
                encoding="utf-8"
            )
        )
        for path in sorted(feature_root.glob("counterfactual_*_rank_*.json")):
            row = json.loads(path.read_text(encoding="utf-8"))
            presence_effect = float(
                row["baseline_feature_score"] - row["counterfactual_feature_score"]
            )
            causal_role = (
                "builder"
                if presence_effect > 0
                else "breaker"
                if presence_effect < 0
                else "neutral"
            )
            predicted_direction = str(row["candidate_direction"])
            direction_confirmed = (
                predicted_direction == "builders" and causal_role == "builder"
            ) or (
                predicted_direction == "breakers" and causal_role == "breaker"
            )
            records.append(
                {
                    "feature_id": feature_id,
                    "target_feature_id": int(row["target_feature_id"]),
                    "birth_step": int(row["end_step"]),
                    "candidate_step": int(row["candidate_step"]),
                    "candidate_rank": int(row["candidate_rank"]),
                    "predicted_direction": predicted_direction,
                    "predicted_influence": float(row["predicted_influence"]),
                    "causal_role": causal_role,
                    "predicted_direction_confirmed": direction_confirmed,
                    "causal_presence_effect": presence_effect,
                    "relative_removal_change": float(
                        row["relative_feature_score_change"]
                    ),
                    "masked_dataset_indices": row["masked_dataset_indices"],
                    "masked_steps": row["masked_steps"],
                    "baseline_hash_verified": row["baseline_model_sha256"]
                    == ranking["expected_model_sha256"],
                    "counterfactual_path": str(path),
                }
            )
    output = {
        "schema_version": 1,
        "method": "single-step loss masking followed by exact replay to feature birth",
        "interpretation": "Gradient direction is a candidate ranking only. Builder or breaker terminology is confirmed exclusively from the sign of the causal presence effect.",
        "records": records,
    }
    root = sae_analysis_root(config, "attribution", kind=SAE_KIND)
    path = root / "causal_validation_summary.json"
    path.write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")
    print(path)


def accepted_feature_ids(config):
    step = max(config.activations.checkpoint_steps)
    root = sae_analysis_root(config, "feature_evaluation", step, SAE_KIND) / "gemini"
    feature_ids = []
    for path in sorted(root.glob("feature_*_evaluation.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        if row["validated_status"] == "accepted":
            feature_ids.append(int(row["feature_id"]))
    return feature_ids


if __name__ == "__main__":
    main()
