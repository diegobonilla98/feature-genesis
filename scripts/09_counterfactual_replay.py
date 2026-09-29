import json
from pathlib import Path

from feature_genesis.attribution.counterfactual import (
    LossMask,
    PartOcclusion,
    SameClassReplacement,
    counterfactual_replay,
)
from feature_genesis.config import load_config
from feature_genesis.data.batch_plan import ExactBatchPlan
from feature_genesis.data.factory import build_datasets
from feature_genesis.determinism import resolve_device
from feature_genesis.sae.io import latest_sae_path, load_sae, sae_analysis_root
from feature_genesis.sae.validation import require_stage_four_gate

CONFIG_PATH = Path("configs/quickdraw_dgx_v4.yaml")
SAE_KIND = "topk"
DATASET_INDICES = []
CANDIDATE_DIRECTIONS = ["builders", "breakers"]
CANDIDATE_RANKS = [0]
INTERVENTION_STEPS = None
PART_INDEX = 0
PART_RADIUS = 12
SAME_CLASS_REPLACEMENTS = {}
REUSE_EXISTING = True


def main():
    config = load_config(CONFIG_PATH)
    require_stage_four_gate(config)
    ExactBatchPlan.load(Path(config.project.run_dir) / "batch_plan.npz")
    include_annotations = config.attribution.intervention == "part_occlusion"
    bundle = build_datasets(config, include_annotations=include_annotations)
    device = resolve_device(config.project.device)
    for feature_id in accepted_feature_ids(config):
        replay_feature(config, bundle, device, feature_id)


def replay_feature(config, bundle, device, feature_id):
    attribution_payload = ranked_candidate_payload(config, feature_id)
    start_step = int(attribution_payload["start_step"])
    end_step = int(attribution_payload["target_step"])
    target_feature_id = int(
        attribution_payload.get("target_feature_id", feature_id)
    )
    sae, _ = load_sae(
        config,
        latest_sae_path(config, end_step, SAE_KIND),
        device,
    )
    root = (
        sae_analysis_root(config, "attribution", kind=SAE_KIND)
        / f"feature_{feature_id:05d}"
    )
    root.mkdir(parents=True, exist_ok=True)
    for direction in CANDIDATE_DIRECTIONS:
        for rank in CANDIDATE_RANKS:
            path = root / (
                f"counterfactual_{config.attribution.intervention}_{direction}"
                f"_rank_{rank:02d}.json"
            )
            if REUSE_EXISTING and path.exists():
                print(path)
                continue
            candidate = ranked_candidate(config, feature_id, direction, rank)
            indices = DATASET_INDICES or ranked_candidate_indices(
                config,
                feature_id,
                direction,
                rank,
            )
            steps = frozenset(
                INTERVENTION_STEPS
                if INTERVENTION_STEPS is not None
                else [int(candidate["step"])]
            )
            mask, image_intervention = build_intervention(
                config,
                bundle,
                indices,
                steps,
            )
            result = counterfactual_replay(
                config,
                sae,
                target_feature_id,
                start_step,
                end_step,
                mask=mask,
                bundle=bundle,
                image_intervention=image_intervention,
            )
            expected_hash = attribution_payload.get("expected_model_sha256")
            if expected_hash and result["baseline_model_sha256"] != expected_hash:
                raise RuntimeError(
                    "Counterfactual baseline differs from the verified attribution checkpoint"
                )
            result["final_feature_id"] = feature_id
            result["target_feature_id"] = target_feature_id
            result["candidate_direction"] = direction
            result["candidate_rank"] = rank
            result["candidate_step"] = int(candidate["step"])
            result["predicted_influence"] = float(candidate["influence"])
            result["causal_presence_effect"] = -float(result["feature_score_change"])
            result["causal_role"] = (
                "builder"
                if result["causal_presence_effect"] > 0
                else "breaker"
                if result["causal_presence_effect"] < 0
                else "neutral"
            )
            result["predicted_direction_confirmed"] = (
                direction == "builders" and result["causal_role"] == "builder"
            ) or (
                direction == "breakers" and result["causal_role"] == "breaker"
            )
            path.write_text(
                json.dumps(result, indent=2, sort_keys=True),
                encoding="utf-8",
            )
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


def build_intervention(
    config,
    bundle,
    indices: list[int],
    steps: frozenset[int],
) -> tuple[LossMask | None, PartOcclusion | SameClassReplacement | None]:
    if config.attribution.intervention == "mask_loss":
        return LossMask(frozenset(indices), steps), None
    if config.attribution.intervention == "part_occlusion":
        return None, PartOcclusion(frozenset(indices), PART_INDEX, PART_RADIUS, steps)
    if config.attribution.intervention == "replace_same_class":
        if not SAME_CLASS_REPLACEMENTS:
            raise ValueError("Set SAME_CLASS_REPLACEMENTS to source_index: replacement_index pairs")
        return None, SameClassReplacement(bundle.train, SAME_CLASS_REPLACEMENTS, steps)
    raise ValueError(f"Unknown intervention: {config.attribution.intervention}")


def ranked_candidate_indices(
    config,
    feature_id: int,
    direction: str,
    rank: int,
) -> list[int]:
    if direction not in {"builders", "breakers"}:
        raise ValueError("CANDIDATE_DIRECTION must be builders or breakers")
    payload = ranked_candidate_payload(config, feature_id)
    candidates = payload.get(direction, [])
    if rank < 0 or rank >= len(candidates):
        raise IndexError(f"Candidate rank {rank} is unavailable for {direction}")
    example_key = (
        "example_builders"
        if direction == "builders"
        else "example_breakers"
    )
    examples = candidates[rank].get(example_key, [])
    if examples:
        return [int(row["dataset_index"]) for row in examples]
    return [int(value) for value in candidates[rank]["dataset_indices"]]


def ranked_candidate(
    config,
    feature_id: int,
    direction: str,
    rank: int,
) -> dict:
    if direction not in {"builders", "breakers"}:
        raise ValueError("Candidate direction must be builders or breakers")
    candidates = ranked_candidate_payload(config, feature_id).get(direction, [])
    if rank < 0 or rank >= len(candidates):
        raise IndexError(f"Candidate rank {rank} is unavailable for {direction}")
    return candidates[rank]


def ranked_candidate_payload(config, feature_id: int) -> dict:
    path = (
        sae_analysis_root(config, "attribution", kind=SAE_KIND)
        / f"feature_{feature_id:05d}"
        / "gradient_ranked_batches.json"
    )
    if not path.exists():
        raise ValueError(
            "Set DATASET_INDICES explicitly or run scripts/08_rank_training_batches.py first"
        )
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
