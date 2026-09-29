import json
from pathlib import Path

from feature_genesis.config import load_config
from feature_genesis.sae.io import sae_analysis_root
from feature_genesis.sae.validation import require_stage_four_gate


CONFIG_PATH = Path("configs/quickdraw_dgx_v4.yaml")
SAE_KIND = "topk"
MODEL_STEP = None
IMPORT_ONLY_ACCEPTED = True


def main():
    config = load_config(CONFIG_PATH)
    require_stage_four_gate(config)
    step = max(config.activations.checkpoint_steps) if MODEL_STEP is None else MODEL_STEP
    root = sae_analysis_root(config, "feature_evaluation", step, SAE_KIND)
    feature_card_root = root / "cards"
    gemini_root = root / "gemini"
    audit_jsonl = gemini_root / "feature_evaluations.jsonl"
    labels_jsonl = (
        gemini_root / "accepted_feature_evaluations.jsonl"
        if IMPORT_ONLY_ACCEPTED
        else audit_jsonl
    )
    evaluated_feature_ids = {
        int(json.loads(line)["feature_id"])
        for line in audit_jsonl.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    labels = {}
    for line in labels_jsonl.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            required = {
                "feature_id",
                "feature_uid",
                "model",
                "prompt_version",
                "label",
                "heldout_validation",
                "validated_status",
            }
            missing = sorted(required - set(row))
            if missing:
                raise ValueError(f"Feature evaluation is missing fields: {missing}")
            if IMPORT_ONLY_ACCEPTED and row["validated_status"] != "accepted":
                continue
            labels[int(row["feature_id"])] = row
    imported = 0
    removed_stale = 0
    for path in sorted(feature_card_root.glob("feature_*.json")):
        card = json.loads(path.read_text(encoding="utf-8"))
        feature_id = int(card["feature_id"])
        if feature_id in labels:
            card["llm_evaluation"] = labels[feature_id]
            path.write_text(json.dumps(card, indent=2, sort_keys=True), encoding="utf-8")
            imported += 1
        elif feature_id in evaluated_feature_ids and "llm_evaluation" in card:
            del card["llm_evaluation"]
            path.write_text(json.dumps(card, indent=2, sort_keys=True), encoding="utf-8")
            removed_stale += 1
    print(
        {
            "imported": imported,
            "excluded": len(evaluated_feature_ids - set(labels)),
            "removed_stale": removed_stale,
            "accepted_only": IMPORT_ONLY_ACCEPTED,
            "root": str(feature_card_root),
        }
    )


if __name__ == "__main__":
    main()
