import json
import math
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote

from feature_genesis.config import load_config
from feature_genesis.sae.io import sae_analysis_root

CONFIG_PATH = Path("configs/quickdraw_dgx_v4.yaml")
SAE_KIND = "topk"
EXPECTED_ACCEPTED_FEATURES = {89, 99, 164}
EXPECTED_REJECTED_OR_CONTROL_FEATURES = {63, 101, 114}


class LocalReferenceParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.references = []

    def handle_starttag(self, tag, attrs):
        values = dict(attrs)
        for key in ["src", "href"]:
            value = values.get(key)
            if value and not value.startswith(("#", "http://", "https://", "mailto:")):
                self.references.append(value)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def main():
    config = load_config(CONFIG_PATH)
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
    result = read_json(run_dir / "result.json")
    replay = read_json(run_dir / "replay_verification.json")
    stage_gate = read_json(
        run_dir
        / "sae_validation"
        / config.sae.artifact_version
        / "stage_04_gate.json"
    )
    candidate_gate = read_json(
        run_dir
        / "sae_validation"
        / config.sae.artifact_version
        / "final_candidate_gate.json"
    )
    fidelity = read_json(
        sae_analysis_root(config, "evaluation", final_step, SAE_KIND)
        / "model_fidelity.json"
    )
    require(result["best_step"] == final_step, "Final model is not the best model")
    require(replay["bitwise_equal"], "Supervised replay is not bitwise exact")
    require(
        replay["expected_sha256"] == replay["replay_sha256"],
        "Supervised replay hashes differ",
    )
    require(stage_gate["passed"], "Stage 4 SAE gate failed")
    require(candidate_gate["passed"], "Final SAE candidate gate failed")
    require(fidelity["loss_recovered"] >= 0.99, "SAE loss recovery is below 99%")
    require(abs(fidelity["accuracy_change"]) <= 0.01, "SAE accuracy change exceeds 1%")
    evaluations = {}
    for path in gemini_root.glob("feature_*_evaluation.json"):
        row = read_json(path)
        evaluations[int(row["feature_id"])] = row
    accepted = {
        feature_id
        for feature_id, row in evaluations.items()
        if row["validated_status"] == "accepted"
    }
    rejected = set(evaluations) - accepted
    require(
        EXPECTED_ACCEPTED_FEATURES.issubset(accepted),
        "Previously validated feature set regressed",
    )
    require(
        EXPECTED_REJECTED_OR_CONTROL_FEATURES.issubset(rejected),
        "Previously rejected or control feature set changed",
    )
    for feature_id in accepted:
        row = evaluations[feature_id]
        metrics = row["heldout_validation"]["metrics"]
        require(metrics["coverage"] == 1.0, f"Feature {feature_id} has incomplete validation")
        require(metrics["auc"] >= 0.75, f"Feature {feature_id} AUC is below 0.75")
        require(
            metrics["one_sided_mannwhitney_p"] <= 0.05,
            f"Feature {feature_id} validation is not significant",
        )
        require(
            row["model_metadata"]["version"].startswith("3.6-flash"),
            f"Feature {feature_id} was not evaluated with Gemini 3.6 Flash",
        )
    lineage = read_json(lineage_path)
    for feature_id in EXPECTED_ACCEPTED_FEATURES | {63}:
        ancestry = lineage["strongest_ancestry"][str(feature_id)]
        edges = ancestry["edges"]
        if feature_id in EXPECTED_ACCEPTED_FEATURES:
            require(
                ancestry["attribution_target_step"]
                == ancestry["structural_birth_step"],
                f"Feature {feature_id} attribution step differs from structural birth",
            )
            require(
                ancestry["attribution_target_feature_id"]
                == ancestry["structural_birth_feature_id"],
                f"Feature {feature_id} attribution feature differs from structural birth",
            )
        else:
            require(
                ancestry["structural_birth_step"] == 0
                and ancestry["attribution_target_step"] > 0,
                "Initialization control does not advance to a trainable attribution target",
            )
        for previous, current in zip(edges, edges[1:]):
            require(
                previous["target"] == current["source"],
                f"Feature {feature_id} ancestry is discontinuous",
            )
        if edges:
            require(
                edges[-1]["target"] == f"s{final_step}:f{feature_id}",
                f"Feature {feature_id} ancestry does not end at the final feature",
            )
        phase = lineage["phases"][str(feature_id)]
        require(
            phase["status"] == "left_censored",
            f"Feature {feature_id} should be marked left-censored",
        )
    causal = read_json(attribution_root / "causal_validation_summary.json")
    causal_by_feature = {
        feature_id: [
            row
            for row in causal["records"]
            if int(row["feature_id"]) == feature_id
        ]
        for feature_id in accepted
    }
    for feature_id, rows in causal_by_feature.items():
        require(
            len(rows) == 2,
            f"Feature {feature_id} does not have builder and breaker replay results",
        )
    for record in causal["records"]:
        raw = read_json(Path(record["counterfactual_path"]))
        effect = raw["baseline_feature_score"] - raw["counterfactual_feature_score"]
        role = "builder" if effect > 0 else "breaker" if effect < 0 else "neutral"
        predicted_role = record["predicted_direction"].removesuffix("s")
        require(record["baseline_hash_verified"], "Causal baseline hash was not verified")
        require(
            math.isclose(effect, record["causal_presence_effect"], abs_tol=1e-9),
            "Causal presence effect differs from raw replay",
        )
        require(role == record["causal_role"], "Causal role has the wrong sign")
        require(
            (role == predicted_role) == record["predicted_direction_confirmed"],
            "Causal confirmation flag is inconsistent",
        )
        require(
            raw["masked_steps"] == [raw["candidate_step"]],
            "Counterfactual masks more than the selected training step",
        )
    for feature_id in accepted:
        manifest = read_json(
            attribution_root
            / f"feature_{feature_id:05d}"
            / "training_influence_manifest.json"
        )
        validated = [
            row for row in manifest if row["causal_status"] == "validated"
        ]
        require(
            validated,
            f"Feature {feature_id} training-image manifest has no causal annotations",
        )
        for row in validated:
            matches = [
                record
                for record in causal_by_feature[feature_id]
                if record["candidate_step"] == row["step"]
                and record["predicted_direction"] == row["predicted_direction"]
            ]
            require(len(matches) == 1, "Manifest causal annotation is ambiguous")
            require(
                row["causal_role"] == matches[0]["causal_role"],
                "Manifest causal role differs from replay summary",
            )
    experiment = read_json(run_dir / "experiment_summary.json")
    require(
        {row["feature_id"] for row in experiment["accepted_features"]}
        == accepted,
        "Experiment summary accepted features are inconsistent",
    )
    for row in experiment["accepted_features"]:
        feature_id = int(row["feature_id"])
        for key in ["final_model_ablation", "final_model_steering"]:
            intervention = row.get(key)
            require(
                intervention is not None,
                f"Feature {feature_id} {key} is missing",
            )
            require(
                int(intervention["images"]) == 512,
                f"Feature {feature_id} {key} has incomplete coverage",
            )
    dossier_root = run_dir / config.evaluation.report_dir / "features"
    dossier_manifest = read_json(dossier_root / "manifest.json")
    feature_quality = read_json(
        sae_analysis_root(config, "evaluation", final_step, SAE_KIND)
        / "feature_quality.json"
    )
    candidate_ids = {
        int(row["feature_id"]) for row in feature_quality["features"]
    }
    require(
        {row["feature_id"] for row in dossier_manifest["features"]}
        == candidate_ids,
        "Per-feature dossier set is inconsistent",
    )
    inventory = dossier_manifest["inventory"]
    require(inventory["dictionary_slots"] == 512, "Dictionary slot count changed")
    require(inventory["live_features"] == 508, "Live feature count changed")
    require(
        inventory["explorable_candidates"] == len(candidate_ids) == 155,
        "Explorable candidate count changed",
    )
    require(
        inventory["validated"] == len(accepted),
        "Validated concept count is inconsistent with Gemini evaluations",
    )
    for row in dossier_manifest["features"]:
        feature_id = int(row["feature_id"])
        feature_root = dossier_root / f"feature_{feature_id:05d}"
        dossier = read_json(feature_root / "dossier.json")
        dossier_report = feature_root / "index.html"
        dossier_html = dossier_report.read_text(encoding="utf-8")
        require(
            dossier["feature_id"] == feature_id,
            f"Feature {feature_id} dossier identity is inconsistent",
        )
        require(
            dossier_html.count("<svg") == 2,
            f"Feature {feature_id} dossier does not contain two evolution charts",
        )
        if dossier["evaluation"] is None:
            require(
                "No semantic claim yet." in dossier_html
                and "no Gemini tokens spent" in dossier_html,
                f"Feature {feature_id} unlabeled dossier overclaims semantics",
            )
        else:
            require(
                "Claim boundary." in dossier_html
                and "Remaining uncertainty." in dossier_html,
                f"Feature {feature_id} evaluated dossier omits scientific caveats",
            )
        if feature_id in accepted:
            require(
                len(dossier["causal_records"]) == 2,
                f"Feature {feature_id} dossier lacks two causal replay results",
            )
            require(
                set(dossier["interventions"]) == {"ablate", "steer"},
                f"Feature {feature_id} dossier lacks removal and steering",
            )
            require(
                dossier["emergence_assessment"]["semantic_onset_status"]
                == "left_censored",
                f"Feature {feature_id} emergence uncertainty is inconsistent",
            )
            require(
                "When did this concept emerge" in dossier_html
                and "helped, confused, or failed to matter" in dossier_html
                and "removed or amplified" in dossier_html,
                f"Feature {feature_id} dossier omits complete causal sections",
            )
        parser = LocalReferenceParser()
        parser.feed(dossier_html)
        for reference in parser.references:
            target = (dossier_report.parent / unquote(reference)).resolve()
            require(
                target.exists(),
                f"Feature {feature_id} dossier has a broken reference: {reference}",
            )
    report_path = run_dir / config.evaluation.report_dir / "report.html"
    report = report_path.read_text(encoding="utf-8")
    require("Experiment summary" in report, "Report omits the experiment summary")
    require(
        "Causal training-example validation" in report,
        "Report omits causal validation",
    )
    output = {
        "passed": True,
        "accepted_features": sorted(accepted),
        "rejected_or_control_features": sorted(rejected),
        "checks": {
            "supervised_replay_bitwise_exact": True,
            "sae_scientific_gate": True,
            "sae_loss_recovered_at_least_99_percent": True,
            "gemini_heldout_validation": True,
            "strongest_edge_lineage": True,
            "causal_replay_consistency": True,
            "all_validated_feature_interventions": True,
            "per_feature_dossiers": True,
            "report": True,
        },
    }
    output_path = run_dir / "artifact_validation.json"
    output_path.write_text(
        json.dumps(output, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(output_path)


if __name__ == "__main__":
    main()
