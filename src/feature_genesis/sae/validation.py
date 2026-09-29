from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from feature_genesis.config import Config
from feature_genesis.hashing import canonical_json_hash, state_dict_hash
from feature_genesis.sae.io import load_sae, sae_root
from feature_genesis.sae.metrics import maximum_positive_decoder_cosine
from feature_genesis.training.checkpoints import read_json
from feature_genesis.training.trainer import project_source_manifest


def validate_stage_four(config: Config, kind: str = "batch_topk") -> dict[str, Any]:
    rows = []
    for role in ["continual", "independent"]:
        steps = config.activations.checkpoint_steps if role == "continual" else [max(config.activations.checkpoint_steps)]
        for sae_seed in config.sae.seeds:
            for model_step in steps:
                root = sae_root(config, model_step, kind, sae_seed, role)
                rows.append(validate_sae_artifact(config, root, model_step, sae_seed, role))
    final_step = max(config.activations.checkpoint_steps)
    stability = cross_seed_direction_stability(config, final_step, kind, "continual")
    passed = all(row["passed"] for row in rows) and stability["passed"]
    result = {
        "passed": passed,
        "artifact_version": config.sae.artifact_version,
        "kind": kind,
        "rows": rows,
        "cross_seed_stability": stability,
    }
    root = Path(config.project.run_dir) / "sae_validation" / config.sae.artifact_version
    root.mkdir(parents=True, exist_ok=True)
    path = root / "stage_04_gate.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    result["path"] = str(path)
    return result


def validate_final_candidate(config: Config, kind: str = "batch_topk") -> dict[str, Any]:
    final_step = max(config.activations.checkpoint_steps)
    rows = [
        validate_sae_artifact(
            config,
            sae_root(config, final_step, kind, sae_seed, "independent"),
            final_step,
            sae_seed,
            "independent",
        )
        for sae_seed in config.sae.seeds
    ]
    stability = cross_seed_direction_stability(config, final_step, kind, "independent")
    result = {
        "passed": all(row["passed"] for row in rows) and stability["passed"],
        "artifact_version": config.sae.artifact_version,
        "kind": kind,
        "model_step": final_step,
        "rows": rows,
        "cross_seed_stability": stability,
    }
    root = Path(config.project.run_dir) / "sae_validation" / config.sae.artifact_version
    root.mkdir(parents=True, exist_ok=True)
    path = root / "final_candidate_gate.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    result["path"] = str(path)
    return result


def require_stage_four_gate(config: Config) -> dict[str, Any]:
    path = (
        Path(config.project.run_dir)
        / "sae_validation"
        / config.sae.artifact_version
        / "stage_04_gate.json"
    )
    if not path.exists():
        raise RuntimeError(f"Stage 4 gate has not been run: {path}")
    result = read_json(path)
    if not result.get("passed", False):
        raise RuntimeError(f"Stage 4 gate failed: {path}")
    return result


def validate_sae_artifact(
    config: Config,
    root: Path,
    model_step: int,
    sae_seed: int,
    role: str,
) -> dict[str, Any]:
    summary_path = root / "summary.json"
    final_path = root / "final.pt"
    checks = []
    if not summary_path.exists() or not final_path.exists():
        return {
            "model_step": model_step,
            "sae_seed": sae_seed,
            "role": role,
            "passed": False,
            "checks": [
                {
                    "name": "artifact_exists",
                    "passed": False,
                    "value": str(root),
                }
            ],
        }
    summary = read_json(summary_path)
    sae, payload = load_sae(config, final_path)
    calibration = summary.get("inference_calibration")
    saved_threshold = float(sae.inference_threshold.detach().cpu())
    positive_decoder_cosine = maximum_positive_decoder_cosine(
        sae.decoder_directions().detach()
    )
    threshold_calibrated = summary.get("kind") in {
        "batch_topk",
        "cosine_batch_topk",
        "matryoshka",
    }
    feature_frequency = np.asarray(summary["metrics"].get("feature_frequency", []), dtype=np.float64)
    feature_image_frequency = np.asarray(
        summary["metrics"].get("feature_image_frequency", []),
        dtype=np.float64,
    )
    usable = feature_frequency >= config.sae.minimum_feature_frequency
    contrastive = (
        usable
        & (feature_image_frequency >= config.sae.minimum_image_frequency)
        & (feature_image_frequency <= config.sae.maximum_image_frequency)
    ) if len(feature_image_frequency) == len(feature_frequency) else np.zeros_like(usable)
    stored_source_manifest = payload["metadata"].get("source_manifest")
    stored_source_hash = payload["metadata"].get("source_manifest_sha256")
    source_manifest_integrity = (
        canonical_json_hash(stored_source_manifest) == stored_source_hash
        if stored_source_manifest is not None
        else isinstance(stored_source_hash, str) and len(stored_source_hash) == 64
    )
    current_source_hash = canonical_json_hash(project_source_manifest())
    checks.extend(
        [
            check("artifact_exists", True, str(final_path)),
            check(
                "state_hash_matches",
                payload["model_sha256"] == state_dict_hash(payload["model"]),
                payload["model_sha256"],
            ),
            check(
                "normalization_is_checkpoint_local",
                payload["metadata"].get("normalization_source_model_step") == model_step
                and payload["metadata"].get("warm_start_copied_buffers") is False,
                payload["metadata"].get("normalization_source_model_step"),
            ),
            check(
                "source_manifest_integrity",
                source_manifest_integrity,
                stored_source_hash,
            ),
            check(
                "calibration_saved",
                (
                    bool(summary.get("calibration_saved_in_checkpoint"))
                    and calibration is not None
                    and abs(saved_threshold - float(calibration["threshold"])) <= 1e-7
                )
                if threshold_calibrated
                else calibration is None and saved_threshold == 0,
                saved_threshold,
            ),
            check(
                "fraction_variance_explained",
                summary["metrics"]["fraction_variance_explained"]
                >= config.sae.minimum_fraction_variance_explained,
                summary["metrics"]["fraction_variance_explained"],
            ),
            check(
                "dead_fraction",
                summary["metrics"]["dead_fraction"] <= config.sae.maximum_dead_fraction,
                summary["metrics"]["dead_fraction"],
            ),
            check(
                "usable_features",
                int(usable.sum()) >= config.sae.minimum_usable_features,
                int(usable.sum()),
            ),
            check(
                "contrastive_features",
                int(contrastive.sum()) >= config.sae.minimum_contrastive_features,
                int(contrastive.sum()),
            ),
            check(
                "maximum_positive_decoder_cosine",
                positive_decoder_cosine <= config.sae.maximum_positive_decoder_cosine,
                positive_decoder_cosine,
            ),
            check(
                "mean_l0_lower",
                summary["metrics"]["mean_l0"] >= config.sae.k * config.sae.minimum_mean_l0_fraction,
                summary["metrics"]["mean_l0"],
            ),
            check(
                "mean_l0_upper",
                summary["metrics"]["mean_l0"] <= config.sae.k * config.sae.maximum_mean_l0_fraction,
                summary["metrics"]["mean_l0"],
            ),
        ]
    )
    return {
        "model_step": model_step,
        "sae_seed": sae_seed,
        "role": role,
        "passed": all(item["passed"] for item in checks),
        "input_scale": float(sae.input_scale.detach().cpu()),
        "checks": checks,
        "provenance": {
            "stored_source_manifest_sha256": stored_source_hash,
            "current_source_manifest_sha256": current_source_hash,
            "current_source_matches": stored_source_hash == current_source_hash,
            "legacy_hash_only": stored_source_manifest is None,
        },
    }


def cross_seed_direction_stability(
    config: Config,
    model_step: int,
    kind: str,
    role: str,
) -> dict[str, Any]:
    if len(config.sae.seeds) < 2:
        return {
            "passed": True,
            "pairs": [],
            "minimum_required_median_cosine": config.sae.minimum_cross_seed_cosine,
            "minimum_required_match_fraction": config.sae.minimum_cross_seed_match_fraction,
            "minimum_required_match_count": config.sae.minimum_cross_seed_match_count,
            "replicated_primary_feature_ids": [],
        }
    primary_seed = config.sae.primary_seed
    primary_root = sae_root(config, model_step, kind, primary_seed, role)
    missing = [
        str(sae_root(config, model_step, kind, sae_seed, role) / "final.pt")
        for sae_seed in config.sae.seeds
        if not (
            sae_root(config, model_step, kind, sae_seed, role) / "final.pt"
        ).exists()
    ]
    if missing:
        return {
            "passed": False,
            "pairs": [],
            "missing_artifacts": missing,
            "minimum_required_median_cosine": config.sae.minimum_cross_seed_cosine,
            "minimum_required_match_fraction": config.sae.minimum_cross_seed_match_fraction,
            "minimum_required_match_count": config.sae.minimum_cross_seed_match_count,
            "replicated_primary_feature_ids": [],
        }
    primary_summary = read_json(primary_root / "summary.json")
    primary_sae, _ = load_sae(config, primary_root / "final.pt")
    primary_frequency = np.asarray(primary_summary["metrics"]["feature_frequency"], dtype=np.float32)
    primary_live = primary_frequency > 0
    primary_live_ids = np.flatnonzero(primary_live)
    primary_directions = primary_sae.decoder_directions().detach().cpu()
    pairs = []
    replicated_sets = []
    for sae_seed in config.sae.seeds:
        if sae_seed == primary_seed:
            continue
        target_root = sae_root(config, model_step, kind, sae_seed, role)
        target_summary = read_json(target_root / "summary.json")
        target_sae, _ = load_sae(config, target_root / "final.pt")
        target_frequency = np.asarray(target_summary["metrics"]["feature_frequency"], dtype=np.float32)
        target_live = target_frequency > 0
        target_live_ids = np.flatnonzero(target_live)
        scores, nearest_targets, reciprocal = reciprocal_nearest_direction_matches(
            primary_directions[torch.from_numpy(primary_live)],
            target_sae.decoder_directions().detach().cpu()[torch.from_numpy(target_live)],
        )
        strong = scores >= config.sae.minimum_cross_seed_cosine
        reciprocal_strong = strong & reciprocal
        replicated_primary_ids = primary_live_ids[reciprocal_strong]
        replicated_sets.append(set(replicated_primary_ids.tolist()))
        median = float(np.median(scores)) if len(scores) else 0.0
        replicated_fraction = float(np.mean(reciprocal_strong)) if len(scores) else 0.0
        pairs.append(
            {
                "primary_seed": primary_seed,
                "target_seed": sae_seed,
                "primary_live_features": int(primary_live.sum()),
                "target_live_features": int(target_live.sum()),
                "median_nearest_absolute_cosine": median,
                "fraction_at_threshold": float(np.mean(strong)) if len(scores) else 0.0,
                "reciprocal_fraction_at_threshold": replicated_fraction,
                "reciprocal_matches_at_threshold": int(reciprocal_strong.sum()),
                "median_reciprocal_match_cosine": float(
                    np.median(scores[reciprocal_strong])
                )
                if reciprocal_strong.any()
                else None,
                "replicated_matches": [
                    {
                        "primary_feature_id": int(primary_live_ids[source_index]),
                        "target_feature_id": int(target_live_ids[nearest_targets[source_index]]),
                        "absolute_cosine": float(scores[source_index]),
                    }
                    for source_index in np.flatnonzero(reciprocal_strong)
                ],
                "passed": replicated_fraction
                >= config.sae.minimum_cross_seed_match_fraction,
            }
        )
    replicated_across_all = sorted(set.intersection(*replicated_sets)) if replicated_sets else []
    count_passed = len(replicated_across_all) >= config.sae.minimum_cross_seed_match_count
    return {
        "passed": all(row["passed"] for row in pairs) and count_passed,
        "pairs": pairs,
        "minimum_required_median_cosine": config.sae.minimum_cross_seed_cosine,
        "minimum_required_match_fraction": config.sae.minimum_cross_seed_match_fraction,
        "minimum_required_match_count": config.sae.minimum_cross_seed_match_count,
        "match_count_passed": count_passed,
        "replicated_primary_feature_ids": replicated_across_all,
        "replicated_primary_features": len(replicated_across_all),
        "replicated_primary_fraction": len(replicated_across_all) / max(1, int(primary_live.sum())),
    }


@torch.no_grad()
def reciprocal_nearest_direction_matches(
    source: torch.Tensor,
    target: torch.Tensor,
    block_size: int = 512,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(source) == 0 or len(target) == 0:
        return (
            np.empty(0, dtype=np.float32),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.bool_),
        )
    source = F.normalize(source.float(), dim=-1)
    target = F.normalize(target.float(), dim=-1)
    scores = []
    target_indices = []
    for start in range(0, len(source), block_size):
        similarities = source[start : start + block_size] @ target.T
        values, indices = similarities.max(dim=1)
        scores.append(values.cpu())
        target_indices.append(indices.cpu())
    reverse_indices = []
    for start in range(0, len(target), block_size):
        similarities = target[start : start + block_size] @ source.T
        reverse_indices.append(similarities.argmax(dim=1).cpu())
    scores_array = torch.cat(scores).numpy()
    target_indices_array = torch.cat(target_indices).numpy()
    reverse_indices_array = torch.cat(reverse_indices).numpy()
    source_indices = np.arange(len(source), dtype=np.int64)
    reciprocal = reverse_indices_array[target_indices_array] == source_indices
    return scores_array, target_indices_array, reciprocal


@torch.no_grad()
def nearest_direction_cosines(
    source: torch.Tensor,
    target: torch.Tensor,
    block_size: int = 512,
) -> np.ndarray:
    scores, _, _ = reciprocal_nearest_direction_matches(source, target, block_size)
    return scores


def check(name: str, passed: bool, value: Any) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "value": value}
