from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from feature_genesis.config import Config
from feature_genesis.data.batch_plan import ExactBatchPlan
from feature_genesis.data.factory import DatasetBundle, build_datasets
from feature_genesis.determinism import (
    configure_determinism,
    critical_environment_fingerprint,
    resolve_device,
    restore_rng_state,
)
from feature_genesis.hashing import canonical_json_hash, state_dict_hash
from feature_genesis.models.factory import build_model
from feature_genesis.training.checkpoints import anchor_path, list_anchors, load_anchor, nearest_anchor
from feature_genesis.training.trainer import (
    ImageFunction,
    LossWeightFunction,
    build_optimizer,
    train_range,
    training_config_payload,
)


CRITICAL_ENVIRONMENT_KEYS = (
    "python",
    "platform",
    "machine",
    "torch",
    "torchvision",
    "numpy",
    "pillow",
    "cuda_runtime",
    "cudnn",
    "cuda_device",
)


def reconstruct_state(
    config: Config,
    target_step: int,
    bundle: DatasetBundle | None = None,
    loss_weight_function: LossWeightFunction | None = None,
    image_function: ImageFunction | None = None,
    force_anchor_step: int | None = None,
) -> tuple[torch.nn.Module, torch.optim.Optimizer, DatasetBundle, ExactBatchPlan]:
    configure_determinism(config.project.seed, config.reproducibility)
    run_dir = Path(config.project.run_dir)
    bundle = bundle or build_datasets(config, include_annotations=False)
    plan = ExactBatchPlan.load(run_dir / "batch_plan.npz")
    metadata = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
    _verify_replay_contract(config, bundle, plan, metadata)
    if not 0 <= target_step <= len(plan):
        raise ValueError(f"target_step must be in [0, {len(plan)}], received {target_step}")
    device = resolve_device(config.project.device)
    model = build_model(config, bundle.num_classes).to(device)
    optimizer = build_optimizer(model, config)
    if force_anchor_step is None:
        start_step, path = nearest_anchor(run_dir, target_step)
    else:
        start_step = force_anchor_step
        path = anchor_path(run_dir, start_step)
    payload = load_anchor(path, model, optimizer, map_location=device)
    anchor_training_hash = payload.get("metadata", {}).get("training_config_sha256")
    if anchor_training_hash != metadata["training_config_sha256"]:
        raise ValueError("Anchor metadata does not match run_metadata.json")
    restore_rng_state(payload["rng"])
    if target_step > start_step:
        train_range(
            model,
            optimizer,
            bundle,
            plan,
            config,
            start_step,
            target_step,
            loss_weight_function=loss_weight_function,
            image_function=image_function,
        )
    return model, optimizer, bundle, plan


def _verify_replay_contract(
    config: Config,
    bundle: DatasetBundle,
    plan: ExactBatchPlan,
    metadata: dict[str, Any],
) -> None:
    current_training_hash = canonical_json_hash(training_config_payload(config))
    if current_training_hash != metadata.get("training_config_sha256"):
        raise ValueError("Training configuration differs from the original run")
    current_plan_hash = plan.metadata()["plan_sha256"]
    if current_plan_hash != metadata.get("batch_plan", {}).get("plan_sha256"):
        raise ValueError("Batch plan differs from the original run")
    current_dataset_hash = canonical_json_hash(bundle.signature)
    if current_dataset_hash != metadata.get("dataset_sha256"):
        raise ValueError("Dataset signature differs from the original run")
    if config.reproducibility.verify_environment:
        expected = metadata.get("environment", {})
        current = critical_environment_fingerprint()
        mismatches = {
            key: {"expected": expected.get(key), "current": current.get(key)}
            for key in CRITICAL_ENVIRONMENT_KEYS
            if expected.get(key) != current.get(key)
        }
        if mismatches:
            raise ValueError(f"Critical replay environment differs: {mismatches}")


def verify_anchor_replay(config: Config, target_step: int) -> dict[str, Any]:
    run_dir = Path(config.project.run_dir)
    target_path = anchor_path(run_dir, target_step)
    if not target_path.exists():
        raise FileNotFoundError(target_path)
    previous = [(step, path) for step, path in list_anchors(run_dir) if step < target_step]
    if not previous:
        raise ValueError("Target anchor must have a previous anchor")
    start_step, _ = max(previous, key=lambda item: item[0])
    model, _, _, _ = reconstruct_state(config, target_step, force_anchor_step=start_step)
    replay_hash = state_dict_hash(model.state_dict())
    payload = torch.load(target_path, map_location="cpu", weights_only=False)
    expected_hash = payload["model_sha256"]
    max_abs_difference = 0.0
    replay_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    for key, expected in payload["model"].items():
        difference = (replay_state[key] - expected.cpu()).abs().max().item()
        max_abs_difference = max(max_abs_difference, float(difference))
    return {
        "start_step": start_step,
        "target_step": target_step,
        "expected_sha256": expected_hash,
        "replay_sha256": replay_hash,
        "bitwise_equal": replay_hash == expected_hash,
        "max_abs_difference": max_abs_difference,
    }
