from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import torch

from feature_genesis.config import Config, SAESection
from feature_genesis.hashing import state_dict_hash
from feature_genesis.sae.core import SparseAutoencoder
from feature_genesis.sae.factory import build_sae
from feature_genesis.training.checkpoints import sae_best_path, sae_resume_path


def sae_root(
    config: Config,
    model_step: int,
    kind: str | None = None,
    sae_seed: int | None = None,
    role: str = "continual",
) -> Path:
    resolved_kind = kind or config.sae.kind
    resolved_seed = config.sae.primary_seed if sae_seed is None else sae_seed
    return (
        Path(config.project.run_dir)
        / "saes"
        / f"model_step_{model_step:08d}"
        / resolved_kind
        / config.sae.artifact_version
        / role
        / f"seed_{resolved_seed:04d}"
    )


def sae_analysis_root(
    config: Config,
    category: str,
    model_step: int | None = None,
    kind: str | None = None,
    sae_seed: int | None = None,
    role: str = "continual",
) -> Path:
    resolved_kind = kind or config.sae.kind
    resolved_seed = config.sae.primary_seed if sae_seed is None else sae_seed
    root = Path(config.project.run_dir) / category
    if model_step is not None:
        root = root / f"step_{model_step:08d}"
    return (
        root
        / resolved_kind
        / config.sae.artifact_version
        / role
        / f"seed_{resolved_seed:04d}"
    )


def save_sae(
    path: str | Path,
    sae: SparseAutoencoder,
    optimizer: torch.optim.Optimizer | None,
    training_step: int,
    metadata: dict[str, Any],
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = sae.state_dict()
    payload = {
        "training_step": training_step,
        "sae_kind": sae.kind,
        "model": state,
        "optimizer": None if optimizer is None else optimizer.state_dict(),
        "model_sha256": state_dict_hash(state),
        "metadata": metadata,
    }
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)
    return path


def load_sae(
    config: Config,
    path: str | Path,
    device: str | torch.device = "cpu",
) -> tuple[SparseAutoencoder, dict[str, Any]]:
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    metadata = payload["metadata"]
    resolved_config = config.model_copy(deep=True)
    resolved_config.sae = SAESection.model_validate(metadata["config"])
    sae = build_sae(resolved_config, int(metadata["input_dim"]), kind=payload["sae_kind"]).to(device)
    sae.load_state_dict(payload["model"], strict=True)
    return sae, payload


def latest_sae_path(
    config: Config,
    model_step: int,
    kind: str | None = None,
    sae_seed: int | None = None,
    role: str = "continual",
) -> Path:
    root = sae_root(config, model_step, kind, sae_seed, role)
    path = root / "final.pt"
    if path.exists():
        return path
    best_path = sae_best_path(root)
    if best_path.exists():
        return best_path
    raise FileNotFoundError(path)


def write_sae_summary(root: str | Path, summary: dict[str, Any]) -> None:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    (root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
