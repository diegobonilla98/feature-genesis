from __future__ import annotations

import json
import shutil
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F
from torch import nn

from feature_genesis.config import Config, save_resolved_config
from feature_genesis.data.batch_plan import ExactBatchPlan, create_batch_plan
from feature_genesis.data.factory import DatasetBundle, build_datasets, eval_loader, exact_train_loader
from feature_genesis.determinism import configure_determinism, environment_fingerprint, resolve_device
from feature_genesis.hashing import canonical_json_hash, directory_manifest, state_dict_hash
from feature_genesis.models.factory import build_model
from feature_genesis.progress import progress
from feature_genesis.training.checkpoints import (
    append_jsonl,
    best_checkpoint_path,
    list_anchors,
    load_anchor,
    load_training_resume,
    read_json,
    resume_checkpoint_path,
    save_anchor,
    save_training_best,
    save_training_resume,
)
from feature_genesis.training.schedule import cosine_learning_rate


LossWeightFunction = Callable[[int, dict[str, torch.Tensor]], torch.Tensor]
ImageFunction = Callable[[int, dict[str, torch.Tensor]], torch.Tensor]


def autocast_context(config: Config, device: torch.device):
    if device.type == "cuda" and config.train.amp_dtype == "bfloat16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()
PREFLIGHT_ARTIFACT_NAMES = frozenset(
    {
        "resolved_config.yaml",
        "stage_00_quickdraw_gate.json",
    }
)


def build_optimizer(model: nn.Module, config: Config) -> torch.optim.SGD:
    return torch.optim.SGD(
        model.parameters(),
        lr=config.train.learning_rate,
        momentum=config.train.momentum,
        weight_decay=config.train.weight_decay,
        nesterov=True,
    )


def train_project(config: Config, overwrite: bool = False, resume: bool = True) -> dict[str, Any]:
    configure_determinism(config.project.seed, config.reproducibility)
    run_dir = Path(config.project.run_dir)
    result_path = run_dir / "result.json"
    if result_path.exists() and not overwrite:
        return read_json(result_path)
    if resume and resume_checkpoint_path(run_dir).exists() and not overwrite:
        return _resume_train_project(config)
    if resume and not overwrite:
        recovered = _recover_resume_from_anchors(config)
        if recovered is not None:
            return recovered
    if run_dir.exists() and any(run_dir.iterdir()) and not _contains_only_preflight_artifacts(run_dir):
        if not overwrite:
            raise FileExistsError(
                f"Run directory is not empty and no resume checkpoint was found: {run_dir}"
            )
        shutil.rmtree(run_dir)
    return _start_train_project(config)


def _start_train_project(config: Config) -> dict[str, Any]:
    run_dir = Path(config.project.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    save_resolved_config(config, run_dir / "resolved_config.yaml")
    bundle = build_datasets(config, include_annotations=False)
    plan = create_batch_plan(
        dataset_size=len(bundle.train),
        batch_size=config.data.batch_size,
        epochs=config.train.epochs,
        seed=config.project.seed,
        max_steps=config.train.max_steps,
    )
    plan.save(run_dir / "batch_plan.npz")
    metadata = _run_metadata(config, bundle, plan)
    with (run_dir / "run_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
    device = resolve_device(config.project.device)
    model = build_model(config, bundle.num_classes).to(device)
    optimizer = build_optimizer(model, config)
    save_anchor(run_dir, 0, model, optimizer, metadata)
    best_metrics = {"loss": float("inf"), "accuracy": 0.0, "step": 0}
    train_range(
        model,
        optimizer,
        bundle,
        plan,
        config,
        0,
        len(plan),
        run_dir=run_dir,
        save_anchors=True,
        save_resume=True,
        validation_loader=eval_loader(bundle.validation, config),
        best_metrics=best_metrics,
    )
    return _finalize_train_project(config, model, bundle, len(plan), resumed=False)


def _contains_only_preflight_artifacts(run_dir: Path) -> bool:
    entries = list(run_dir.iterdir())
    return bool(entries) and all(
        entry.is_file() and entry.name in PREFLIGHT_ARTIFACT_NAMES
        for entry in entries
    )


def _recover_resume_from_anchors(config: Config) -> dict[str, Any] | None:
    run_dir = Path(config.project.run_dir)
    if not (run_dir / "run_metadata.json").exists() or not (run_dir / "batch_plan.npz").exists():
        return None
    anchors = list_anchors(run_dir)
    if not anchors:
        return None
    start_step, anchor_file = max(anchors, key=lambda item: item[0])
    metadata = read_json(run_dir / "run_metadata.json")
    _verify_run_metadata(config, metadata)
    bundle = build_datasets(config, include_annotations=False)
    plan = ExactBatchPlan.load(run_dir / "batch_plan.npz")
    if start_step >= len(plan):
        return None
    device = resolve_device(config.project.device)
    model = build_model(config, bundle.num_classes).to(device)
    optimizer = build_optimizer(model, config)
    load_anchor(anchor_file, model, optimizer, map_location=device)
    best_metrics = {"loss": float("inf"), "accuracy": 0.0, "step": float(start_step)}
    if best_checkpoint_path(run_dir).exists():
        best_payload = torch.load(best_checkpoint_path(run_dir), map_location="cpu", weights_only=False)
        best_metrics = dict(best_payload.get("metrics", best_metrics))
        best_metrics["step"] = float(best_payload.get("step", start_step))
    save_training_resume(run_dir, start_step, model, optimizer, metadata, best_metrics)
    return _resume_train_project(config)


def _resume_train_project(config: Config) -> dict[str, Any]:
    run_dir = Path(config.project.run_dir)
    metadata = read_json(run_dir / "run_metadata.json")
    _verify_run_metadata(config, metadata)
    bundle = build_datasets(config, include_annotations=False)
    plan = ExactBatchPlan.load(run_dir / "batch_plan.npz")
    device = resolve_device(config.project.device)
    model = build_model(config, bundle.num_classes).to(device)
    optimizer = build_optimizer(model, config)
    payload = load_training_resume(run_dir, model, optimizer, map_location=device)
    start_step = int(payload["step"])
    best_metrics = dict(payload.get("best_metrics", {"loss": float("inf"), "accuracy": 0.0, "step": 0}))
    if start_step >= len(plan):
        return _finalize_train_project(config, model, bundle, len(plan), resumed=True)
    train_range(
        model,
        optimizer,
        bundle,
        plan,
        config,
        start_step,
        len(plan),
        run_dir=run_dir,
        save_anchors=True,
        save_resume=True,
        validation_loader=eval_loader(bundle.validation, config),
        best_metrics=best_metrics,
    )
    return _finalize_train_project(config, model, bundle, len(plan), resumed=True)


def _finalize_train_project(
    config: Config,
    model: nn.Module,
    bundle: DatasetBundle,
    steps: int,
    resumed: bool,
) -> dict[str, Any]:
    run_dir = Path(config.project.run_dir)
    device = next(model.parameters()).device
    best_path = best_checkpoint_path(run_dir)
    if best_path.exists():
        best_payload = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(best_payload["model"], strict=True)
        best_step = int(best_payload["step"])
        validation_metrics = dict(best_payload["metrics"])
    else:
        best_step = steps
        validation_metrics = evaluate_model(model, eval_loader(bundle.validation, config), device, config, desc="validation")
    test_metrics = evaluate_model(model, eval_loader(bundle.test, config), device, config, desc="test")
    result = {
        "run_dir": str(run_dir),
        "steps": steps,
        "best_step": best_step,
        "best_model_sha256": state_dict_hash(model.state_dict()),
        "validation": validation_metrics,
        "test": test_metrics,
        "resumed": resumed,
    }
    with (run_dir / "result.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    resume_path = resume_checkpoint_path(run_dir)
    if resume_path.exists():
        resume_path.unlink()
    return result


def train_range(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    bundle: DatasetBundle,
    plan: ExactBatchPlan,
    config: Config,
    start_step: int,
    end_step: int,
    run_dir: str | Path | None = None,
    save_anchors: bool = False,
    save_resume: bool = False,
    validation_loader=None,
    best_metrics: dict[str, float] | None = None,
    loss_weight_function=None,
    image_function=None,
) -> list[dict[str, float]]:
    device = next(model.parameters()).device
    model.train()
    loader = exact_train_loader(bundle.train, plan, config, start_step, end_step)
    rows: list[dict[str, float]] = []
    metadata = None
    if run_dir is not None:
        metadata = read_json(Path(run_dir) / "run_metadata.json")
    if best_metrics is None:
        best_metrics = {"loss": float("inf"), "accuracy": 0.0, "step": 0}
    step_range = range(start_step, end_step)
    for global_step, batch in progress(
        zip(step_range, loader, strict=True),
        total=end_step - start_step,
        desc="train",
    ):
        learning_rate = cosine_learning_rate(
            global_step,
            len(plan),
            config.train.learning_rate,
            config.train.min_learning_rate,
            config.train.warmup_steps,
        )
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        if image_function is not None:
            images = image_function(
                global_step,
                {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()},
            )
        if config.train.channels_last and images.ndim == 4:
            images = images.contiguous(memory_format=torch.channels_last)
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(config, device):
            logits = model(images)
            losses = F.cross_entropy(logits, labels, reduction="none", label_smoothing=config.train.label_smoothing)
        if loss_weight_function is None:
            weights = torch.ones_like(losses)
        else:
            weights = loss_weight_function(
                global_step,
                {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()},
            ).to(losses)
        loss = (losses * weights).sum() / losses.numel()
        loss.backward()
        if config.train.gradient_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.gradient_clip_norm)
        optimizer.step()
        accuracy = (logits.argmax(dim=-1) == labels).float().mean()
        row = {
            "step": float(global_step + 1),
            "epoch": float(batch["epoch"][0].item()),
            "learning_rate": float(learning_rate),
            "loss": float(loss.detach().cpu()),
            "accuracy": float(accuracy.detach().cpu()),
            "weight_sum": float(weights.sum().detach().cpu()),
        }
        rows.append(row)
        completed_step = global_step + 1
        if run_dir is not None and completed_step % config.train.log_every == 0:
            append_jsonl(Path(run_dir) / "telemetry.jsonl", row)
        if run_dir is not None and save_resume and completed_step % config.train.log_every == 0:
            save_training_resume(run_dir, completed_step, model, optimizer, metadata, best_metrics)
        if (
            run_dir is not None
            and validation_loader is not None
            and completed_step % config.train.validation_every == 0
        ):
            metrics = evaluate_model(model, validation_loader, device, config, desc="validation")
            append_jsonl(Path(run_dir) / "validation_telemetry.jsonl", {"step": float(completed_step), **metrics})
            if _is_better_validation(metrics, best_metrics):
                best_metrics.update(metrics)
                best_metrics["step"] = float(completed_step)
                save_training_best(run_dir, completed_step, model, optimizer, metadata, metrics)
        if save_anchors and run_dir is not None and should_save_anchor(completed_step, end_step, config):
            save_anchor(run_dir, completed_step, model, optimizer, metadata)
    if run_dir is not None and validation_loader is not None and end_step > start_step:
        metrics = evaluate_model(model, validation_loader, device, config, desc="validation")
        if _is_better_validation(metrics, best_metrics) or not best_checkpoint_path(run_dir).exists():
            best_metrics.update(metrics)
            best_metrics["step"] = float(end_step)
            save_training_best(run_dir, end_step, model, optimizer, metadata, metrics)
    if run_dir is not None and save_resume:
        save_training_resume(run_dir, end_step, model, optimizer, metadata, best_metrics)
    return rows


def should_save_anchor(completed_step: int, end_step: int, config: Config) -> bool:
    periodic = config.train.anchor_every > 0 and completed_step % config.train.anchor_every == 0
    return periodic or completed_step in set(config.train.analysis_steps) or completed_step == end_step


def _is_better_validation(metrics: dict[str, float], best_metrics: dict[str, float]) -> bool:
    if metrics["loss"] < best_metrics["loss"] - 1e-12:
        return True
    if abs(metrics["loss"] - best_metrics["loss"]) <= 1e-12 and metrics["accuracy"] > best_metrics["accuracy"]:
        return True
    return False


def evaluate_model(model: nn.Module, loader, device: torch.device, config: Config, desc: str = "eval") -> dict[str, Any]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_top5 = 0
    total_count = 0
    class_correct = None
    class_count = None
    with torch.no_grad():
        for batch in progress(loader, desc=desc):
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            if config.train.channels_last and images.ndim == 4:
                images = images.contiguous(memory_format=torch.channels_last)
            with autocast_context(config, device):
                logits = model(images)
                batch_loss = F.cross_entropy(logits, labels, reduction="sum")
            predictions = logits.argmax(dim=-1)
            correct = predictions == labels
            top5 = logits.topk(min(5, logits.shape[-1]), dim=-1).indices
            total_loss += float(batch_loss.cpu())
            total_correct += int(correct.sum().cpu())
            total_top5 += int((top5 == labels[:, None]).any(dim=-1).sum().cpu())
            total_count += labels.numel()
            if class_count is None:
                class_count = torch.zeros(logits.shape[-1], dtype=torch.long)
                class_correct = torch.zeros(logits.shape[-1], dtype=torch.long)
            cpu_labels = labels.detach().cpu()
            class_count += torch.bincount(cpu_labels, minlength=logits.shape[-1])
            class_correct += torch.bincount(cpu_labels[correct.detach().cpu()], minlength=logits.shape[-1])
    if class_count is None or class_correct is None:
        raise RuntimeError("Evaluation loader produced no examples")
    present = class_count > 0
    per_class = class_correct[present].double() / class_count[present].double()
    worst_count = max(1, int((len(per_class) + 9) // 10))
    result = {
        "loss": total_loss / total_count,
        "accuracy": total_correct / total_count,
        "top5_accuracy": total_top5 / total_count,
        "balanced_accuracy": float(per_class.mean()),
        "worst_decile_accuracy": float(per_class.sort().values[:worst_count].mean()),
        "per_class_accuracy": per_class.tolist(),
        "images": total_count,
    }
    model.train()
    return result

def _run_metadata(config: Config, bundle: DatasetBundle, plan: ExactBatchPlan) -> dict[str, Any]:
    resolved = config.model_dump(mode="json")
    training_payload = training_config_payload(config)
    source_manifest = project_source_manifest()
    return {
        "config_sha256": canonical_json_hash(resolved),
        "training_config_sha256": canonical_json_hash(training_payload),
        "training_config": training_payload,
        "batch_plan": plan.metadata(),
        "dataset": bundle.signature,
        "dataset_sha256": canonical_json_hash(bundle.signature),
        "num_classes": bundle.num_classes,
        "class_names": bundle.class_names,
        "attribute_names": bundle.attribute_names,
        "part_names": bundle.part_names,
        "environment": environment_fingerprint(),
        "source_manifest": source_manifest,
        "source_manifest_sha256": canonical_json_hash(source_manifest),
    }


def _verify_run_metadata(config: Config, metadata: dict[str, Any]) -> None:
    if canonical_json_hash(training_config_payload(config)) != metadata.get("training_config_sha256"):
        raise ValueError("Training configuration differs from the saved run")
    expected_source_hash = metadata.get("source_manifest_sha256")
    if expected_source_hash is not None:
        actual_source_hash = canonical_json_hash(project_source_manifest())
        if actual_source_hash != expected_source_hash:
            raise ValueError("Project source differs from the saved run")


def training_config_payload(config: Config) -> dict[str, Any]:
    reproducibility = config.reproducibility.model_dump(mode="json")
    reproducibility.pop("verify_environment", None)
    data = config.data.model_dump(mode="json")
    data.pop("root", None)
    return {
        "project": {"seed": config.project.seed, "device": config.project.device},
        "reproducibility": reproducibility,
        "data": data,
        "model": config.model.model_dump(mode="json"),
        "train": config.train.model_dump(mode="json"),
    }


def project_source_manifest() -> list[dict[str, Any]]:
    project_root = Path(__file__).resolve().parents[3]
    rows = []
    for directory in ["src", "scripts", "configs"]:
        root = project_root / directory
        for row in directory_manifest(
            root,
            suffixes=(".py", ".yaml", ".yml", ".toml"),
        ):
            rows.append(
                {
                    **row,
                    "path": f"{directory}/{row['path']}",
                }
            )
    pyproject = project_root / "pyproject.toml"
    if pyproject.exists():
        rows.extend(
            {
                **row,
                "path": f"project/{row['path']}",
            }
            for row in directory_manifest(
                project_root,
                suffixes=(".toml",),
            )
            if row["path"] == "pyproject.toml"
        )
    return sorted(rows, key=lambda row: row["path"])
