from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import torch
from torch import nn

from feature_genesis.determinism import RNGState, capture_rng_state, restore_rng_state
from feature_genesis.hashing import state_dict_hash


def anchor_path(run_dir: str | Path, step: int) -> Path:
    return Path(run_dir) / "anchors" / f"anchor_step_{step:08d}.pt"


def resume_checkpoint_path(run_dir: str | Path) -> Path:
    return Path(run_dir) / "checkpoints" / "resume.pt"


def best_checkpoint_path(run_dir: str | Path) -> Path:
    return Path(run_dir) / "checkpoints" / "best.pt"


def sae_resume_path(root: str | Path) -> Path:
    return Path(root) / "resume.pt"


def sae_best_path(root: str | Path) -> Path:
    return Path(root) / "best.pt"


def _save_payload(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)
    return path


def save_anchor(
    run_dir: str | Path,
    step: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    metadata: dict[str, Any],
) -> Path:
    path = anchor_path(run_dir, step)
    payload = {
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "rng": capture_rng_state(),
        "model_sha256": state_dict_hash(model.state_dict()),
        "metadata": metadata,
    }
    return _save_payload(path, payload)


def save_training_resume(
    run_dir: str | Path,
    step: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    metadata: dict[str, Any],
    best_metrics: dict[str, float],
) -> Path:
    payload = {
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "rng": capture_rng_state(),
        "model_sha256": state_dict_hash(model.state_dict()),
        "metadata": metadata,
        "best_metrics": best_metrics,
    }
    return _save_payload(resume_checkpoint_path(run_dir), payload)


def save_training_best(
    run_dir: str | Path,
    step: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    metadata: dict[str, Any],
    metrics: dict[str, float],
) -> Path:
    payload = {
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "rng": capture_rng_state(),
        "model_sha256": state_dict_hash(model.state_dict()),
        "metadata": metadata,
        "metrics": metrics,
    }
    return _save_payload(best_checkpoint_path(run_dir), payload)


def load_training_resume(
    run_dir: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    payload = torch.load(resume_checkpoint_path(run_dir), map_location=map_location, weights_only=False)
    model.load_state_dict(payload["model"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    restore_rng_state(payload["rng"])
    return payload


def load_anchor(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    model.load_state_dict(payload["model"], strict=True)
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer"])
    return payload


def list_anchors(run_dir: str | Path) -> list[tuple[int, Path]]:
    output: list[tuple[int, Path]] = []
    for path in sorted((Path(run_dir) / "anchors").glob("anchor_step_*.pt")):
        step = int(path.stem.rsplit("_", 1)[-1])
        output.append((step, path))
    return output


def nearest_anchor(run_dir: str | Path, target_step: int) -> tuple[int, Path]:
    candidates = [(step, path) for step, path in list_anchors(run_dir) if step <= target_step]
    if not candidates:
        raise FileNotFoundError(f"No anchor at or before step {target_step}")
    return max(candidates, key=lambda item: item[0])


def append_jsonl(path: str | Path, row: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
