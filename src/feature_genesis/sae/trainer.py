from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
import torch.nn.functional as F

from feature_genesis.activations import ActivationCache
from feature_genesis.config import Config
from feature_genesis.determinism import configure_determinism, resolve_device
from feature_genesis.hashing import canonical_json_hash, sha256_file, stable_seed
from feature_genesis.lineage.alignment import fit_cache_alignment
from feature_genesis.progress import progress
from feature_genesis.sae.core import SparseAutoencoder, geometric_median, low_coherence_frame
from feature_genesis.sae.factory import build_sae
from feature_genesis.sae.io import load_sae, sae_best_path, sae_resume_path, sae_root, save_sae, write_sae_summary
from feature_genesis.sae.metrics import evaluate_reconstruction_loss, evaluate_sae
from feature_genesis.training.checkpoints import read_json
from feature_genesis.training.schedule import cosine_learning_rate
from feature_genesis.training.trainer import project_source_manifest


def train_sae(
    config: Config,
    activation_root: str | Path,
    model_step: int,
    kind: str | None = None,
    previous_sae_path: str | Path | None = None,
    previous_activation_root: str | Path | None = None,
    sae_seed: int | None = None,
    role: str = "continual",
    resume: bool = True,
) -> dict[str, Any]:
    resolved_kind = kind or config.sae.kind
    resolved_seed = config.sae.primary_seed if sae_seed is None else sae_seed
    configure_determinism(
        stable_seed(config.project.seed, "sae", resolved_seed, model_step, resolved_kind, role),
        config.reproducibility,
    )
    device = resolve_device(config.project.device)
    cache = ActivationCache.load(activation_root)
    input_dim = int(cache.activations.shape[1])
    root = sae_root(config, model_step, resolved_kind, resolved_seed, role)
    root.mkdir(parents=True, exist_ok=True)
    summary_path = root / "summary.json"
    resume_path = sae_resume_path(root)
    if summary_path.exists() and not resume_path.exists():
        return read_json(summary_path)

    partitions = partition_cache_rows(
        cache,
        config,
        stable_seed(config.project.seed, "sae-partitions", resolved_seed, model_step, resolved_kind, role),
    )
    sae = build_sae(config, input_dim, resolved_kind).to(device)
    sample_count = min(config.sae.initialization_samples, len(cache.activations))
    sample_count = min(sample_count, len(partitions["train"]))
    generator = torch.Generator().manual_seed(
        stable_seed(config.project.seed, "sae-init", resolved_seed, model_step, resolved_kind, role)
    )
    sample_positions = torch.randperm(len(partitions["train"]), generator=generator)[:sample_count].numpy()
    sample_indices = partitions["train"][sample_positions]
    samples = torch.from_numpy(np.array(cache.activations[sample_indices], dtype=np.float32, copy=True)).to(device)
    reference_decoder = None
    start_step = 0
    best_loss = float("inf")
    best_step = 0
    resumed = resume and resume_path.exists()
    if resumed:
        sae, payload = load_sae(config, resume_path, device)
        optimizer = torch.optim.AdamW(sae.parameters(), lr=config.sae.learning_rate, weight_decay=config.sae.weight_decay)
        if payload.get("optimizer") is not None:
            optimizer.load_state_dict(payload["optimizer"])
        start_step = int(payload.get("training_step", 0))
        sae_metadata_loaded = payload.get("metadata", {})
        best_loss = float(sae_metadata_loaded.get("best_loss", float("inf")))
        best_step = int(sae_metadata_loaded.get("best_step", 0))
    else:
        initialization_generator = torch.Generator(device=device).manual_seed(
            stable_seed(config.project.seed, "sae-directions", resolved_seed, model_step, resolved_kind, role)
        )
        _initialize_sae(
            sae,
            samples,
            config.sae.input_normalization,
            initialization_generator,
        )
        optimizer = torch.optim.AdamW(sae.parameters(), lr=config.sae.learning_rate, weight_decay=config.sae.weight_decay)
    alignment_metadata = None
    if previous_sae_path is not None and config.sae.warm_start_mode == "aligned_directions":
        if previous_activation_root is None:
            raise ValueError("previous_activation_root is required for aligned warm starts")
        previous, _ = load_sae(config, previous_sae_path, device)
        previous_cache = ActivationCache.load(previous_activation_root)
        alignment = fit_cache_alignment(
            previous_cache,
            cache,
            config.sae.warm_start_alignment_rows,
            stable_seed(config.project.seed, "sae-warm-alignment", resolved_seed, model_step, resolved_kind),
        )
        aligned_directions = torch.from_numpy(
            alignment.transform_directions(previous.decoder_directions().detach().cpu().numpy())
        ).to(device)
        aligned_directions = F.normalize(aligned_directions, dim=-1)
        reference_decoder = aligned_directions.detach().clone()
        if not resumed:
            transfer_aligned_directions(sae, aligned_directions)
        alignment_metadata = {
            "source_activation_root": str(previous_cache.root),
            "target_activation_root": str(cache.root),
            "residual_ratio": alignment.residual_ratio,
            "heldout_residual_ratio": alignment.heldout_residual_ratio,
            "rows": min(
                len(previous_cache.activations),
                len(cache.activations),
                config.sae.warm_start_alignment_rows,
            ),
        }

    telemetry_path = root / "telemetry.jsonl"
    rows: list[dict[str, float]] = []
    batches = indexed_activation_batches(
        cache,
        partitions["train"],
        config.sae.batch_size,
        stable_seed(config.project.seed, "sae-batches", resolved_seed, model_step, resolved_kind, role),
        config.sae.steps,
    )
    sae_metadata = _metadata(
        config,
        cache,
        model_step,
        resolved_kind,
        resolved_seed,
        role,
        partitions,
        alignment_metadata,
    )
    sae.train()
    eval_every = max(1, config.sae.checkpoint_every)
    for step, inputs in progress(enumerate(batches), total=config.sae.steps, desc=f"sae {resolved_kind}"):
        if step < start_step:
            continue
        inputs = inputs.to(device)
        learning_rate = cosine_learning_rate(
            step,
            config.sae.steps,
            config.sae.learning_rate,
            config.sae.min_learning_rate,
            config.sae.warmup_steps,
        )
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        optimizer.zero_grad(set_to_none=True)
        if hasattr(sae, "sparsity_scale"):
            warmup = max(1, config.sae.jump_sparsity_warmup_steps)
            sae.sparsity_scale.fill_(min(1.0, (step + 1) / warmup))
        output = sae(inputs)
        temporal_loss = inputs.new_zeros(())
        if reference_decoder is not None and config.sae.temporal_weight > 0:
            temporal_loss = config.sae.temporal_weight * F.mse_loss(sae.decoder_directions(), reference_decoder)
        loss = output.loss + temporal_loss
        loss.backward()
        sae.project_decoder_gradients_()
        if config.sae.gradient_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(sae.parameters(), config.sae.gradient_clip_norm)
        optimizer.step()
        sae.normalize_decoder_()
        row = {
            "step": float(step + 1),
            "learning_rate": float(learning_rate),
            "loss": float(loss.detach().cpu()),
            "reconstruction_loss": float(output.reconstruction_loss.detach().cpu()),
            "auxiliary_loss": float(output.auxiliary_loss.detach().cpu()),
            "temporal_loss": float(temporal_loss.detach().cpu()),
            "l0": float(output.l0.detach().cpu()),
            "dead_features": float(output.dead_features.detach().cpu()),
        }
        rows.append(row)
        completed_step = step + 1
        if completed_step % max(1, config.train.log_every) == 0:
            with telemetry_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
        if completed_step % max(1, config.train.log_every) == 0:
            save_sae(
                resume_path,
                sae,
                optimizer,
                completed_step,
                {
                    **sae_metadata,
                    "best_loss": best_loss,
                    "best_step": best_step,
                },
            )
        if completed_step % eval_every == 0 or completed_step == config.sae.steps:
            sae.eval()
            eval_loss = evaluate_reconstruction_loss(
                sae,
                cache,
                partitions["selection"],
                min(config.sae.batch_size, 2048),
                device,
                inference=False,
            )
            sae.train()
            if eval_loss < best_loss:
                best_loss = eval_loss
                best_step = completed_step
                save_sae(
                    sae_best_path(root),
                    sae,
                    optimizer,
                    completed_step,
                    {
                        **sae_metadata,
                        "best_selection_loss": best_loss,
                        "best_step": best_step,
                    },
                )

    best_path = sae_best_path(root)
    if best_path.exists():
        sae, best_payload = load_sae(config, best_path, device)
        best_step = int(best_payload.get("training_step", best_step))
    calibration_samples = torch.from_numpy(
        np.array(cache.activations[partitions["calibration"]], dtype=np.float32, copy=True)
    ).to(device)
    calibration = None
    if resolved_kind in {"batch_topk", "cosine_batch_topk", "matryoshka"}:
        calibration = calibrate_inference_threshold(
            sae,
            calibration_samples,
            config.sae.k,
            min(config.sae.batch_size, 2048),
        )
    final_metadata = {
        **sae_metadata,
        "best_selection_loss": best_loss,
        "best_step": best_step,
        "calibration": calibration,
        "calibration_saved_in_checkpoint": calibration is not None,
        "final_input_scale": float(sae.input_scale.detach().cpu()),
        "final_inference_threshold": float(sae.inference_threshold.detach().cpu()),
    }
    final_path = save_sae(root / "final.pt", sae, None, best_step, final_metadata)
    sae.eval()
    metrics = evaluate_sae(
        sae,
        cache,
        config.sae.batch_size,
        device,
        row_indices=partitions["evaluation"],
        include_feature_metrics=True,
    )
    summary = {
        "path": str(final_path),
        "best_path": str(best_path) if best_path.exists() else str(final_path),
        "best_step": best_step,
        "best_loss": best_loss,
        "model_step": model_step,
        "kind": resolved_kind,
        "sae_seed": resolved_seed,
        "role": role,
        "artifact_version": config.sae.artifact_version,
        "training_steps": config.sae.steps,
        "inference_calibration": calibration,
        "calibration_saved_in_checkpoint": calibration is not None,
        "warm_start_alignment": alignment_metadata,
        "partitions": {name: len(indices) for name, indices in partitions.items()},
        "metrics": metrics,
        "last_training_row": rows[-1] if rows else {},
    }
    write_sae_summary(root, summary)
    if resume_path.exists():
        resume_path.unlink()
    return summary


@torch.no_grad()
def calibrate_inference_threshold(
    sae: SparseAutoencoder,
    samples: torch.Tensor,
    target_l0: int,
    batch_size: int,
) -> dict[str, float | int]:
    if target_l0 <= 0:
        raise ValueError("target_l0 must be positive")
    sample_count = samples.shape[0]
    target_count = min(sample_count * target_l0, sample_count * sae.dictionary_size)
    retained = samples.new_empty(0)
    for start in range(0, sample_count, batch_size):
        acts = F.relu(sae.preactivations(samples[start : start + batch_size])).reshape(-1)
        combined = torch.cat([retained, acts])
        if combined.numel() > target_count:
            retained = combined.topk(target_count).values
        else:
            retained = combined
    threshold = retained.min().clamp_min(torch.finfo(retained.dtype).tiny)
    sae.inference_threshold.copy_(threshold)
    active = 0
    for start in range(0, sample_count, batch_size):
        acts = F.relu(sae.preactivations(samples[start : start + batch_size]))
        active += int((acts >= threshold).sum().cpu())
    return {
        "rows": sample_count,
        "target_l0": target_l0,
        "realized_l0": active / sample_count,
        "threshold": float(threshold.cpu()),
    }


def train_continual_sae_sequence(
    config: Config,
    activation_roots: dict[int, str | Path],
    kind: str | None = None,
    sae_seed: int | None = None,
    resume: bool = True,
) -> list[dict[str, Any]]:
    previous = None
    previous_activation_root = None
    output = []
    warm_start = config.sae.warm_start_mode == "aligned_directions"
    description = "continual sae" if warm_start else "checkpointwise independent sae"
    for model_step, activation_root in progress(sorted(activation_roots.items()), desc=description):
        summary = train_sae(
            config,
            activation_root,
            model_step,
            kind,
            previous if warm_start else None,
            previous_activation_root if warm_start else None,
            sae_seed,
            "continual",
            resume,
        )
        previous = summary["path"] if warm_start else None
        previous_activation_root = activation_root if warm_start else None
        output.append(summary)
    return output


def partition_cache_rows(
    cache: ActivationCache,
    config: Config,
    seed: int,
) -> dict[str, np.ndarray]:
    image_count = len(cache.image_offsets) - 1
    if image_count < 8:
        raise ValueError("SAE partitioning requires at least eight images")
    generator = torch.Generator().manual_seed(seed)
    image_order = torch.randperm(image_count, generator=generator).numpy()
    selection_count = max(1, int(round(image_count * config.sae.selection_fraction)))
    calibration_count = max(1, int(round(image_count * config.sae.calibration_fraction)))
    evaluation_count = max(1, int(round(image_count * config.sae.evaluation_fraction)))
    reserved = selection_count + calibration_count + evaluation_count
    if reserved >= image_count:
        raise ValueError("SAE data partitions leave no training images")
    selection_images = image_order[:selection_count]
    calibration_images = image_order[selection_count : selection_count + calibration_count]
    evaluation_images = image_order[
        selection_count + calibration_count : selection_count + calibration_count + evaluation_count
    ]
    train_images = image_order[reserved:]
    return {
        "train": rows_for_images(cache, train_images),
        "selection": rows_for_images(cache, selection_images),
        "calibration": rows_for_images(cache, calibration_images),
        "evaluation": rows_for_images(cache, evaluation_images),
    }


def rows_for_images(cache: ActivationCache, image_indices: np.ndarray) -> np.ndarray:
    rows = [
        np.arange(
            int(cache.image_offsets[image_index]),
            int(cache.image_offsets[image_index + 1]),
            dtype=np.int64,
        )
        for image_index in image_indices
    ]
    return np.concatenate(rows) if rows else np.empty(0, dtype=np.int64)


def indexed_activation_batches(
    cache: ActivationCache,
    row_indices: np.ndarray,
    batch_size: int,
    seed: int,
    steps: int,
) -> Iterator[torch.Tensor]:
    indices = np.asarray(row_indices, dtype=np.int64)
    if len(indices) == 0:
        raise ValueError("SAE training requires at least one row")
    yielded = 0
    epoch = 0
    while yielded < steps:
        generator = torch.Generator().manual_seed(stable_seed(seed, "indexed-activation-epoch", epoch))
        order = torch.randperm(len(indices), generator=generator).numpy()
        for start in range(0, len(indices), batch_size):
            if yielded >= steps:
                return
            positions = order[start : start + batch_size]
            batch_indices = indices[positions]
            if len(batch_indices) < batch_size:
                extra_generator = torch.Generator().manual_seed(
                    stable_seed(seed, "indexed-activation-padding", epoch, yielded)
                )
                extra_positions = torch.randint(
                    0,
                    len(indices),
                    (batch_size - len(batch_indices),),
                    generator=extra_generator,
                ).numpy()
                batch_indices = np.concatenate([batch_indices, indices[extra_positions]])
            yield torch.from_numpy(np.array(cache.activations[batch_indices], dtype=np.float32, copy=True))
            yielded += 1
        epoch += 1


@torch.no_grad()
def transfer_aligned_directions(
    sae: SparseAutoencoder,
    aligned_directions: torch.Tensor,
) -> None:
    if aligned_directions.shape != sae.decoder.shape:
        raise ValueError("Aligned warm-start directions do not match SAE dimensions")
    directions = F.normalize(aligned_directions.to(sae.decoder), dim=-1)
    sae.decoder.copy_(directions)
    sae.encoder.copy_(directions.T)
    sae.encoder_bias.zero_()
    sae.inference_threshold.zero_()
    sae.inactive_steps.zero_()


@torch.no_grad()
def _initialize_sae(
    sae: SparseAutoencoder,
    samples: torch.Tensor,
    normalization: str,
    generator: torch.Generator,
) -> None:
    sae.initialize_from_data(samples, generator=generator)
    if normalization == "none":
        center = geometric_median(samples.float())
        normalized = samples.float() - center
        directions = low_coherence_frame(normalized, sae.dictionary_size, generator)
        sae.input_scale.fill_(1)
        sae.decoder_bias.copy_(center)
        sae.decoder.copy_(directions)
        if hasattr(sae, "encoder"):
            sae.encoder.copy_(directions.T)
            sae.encoder_bias.zero_()


def _metadata(
    config: Config,
    cache: ActivationCache,
    model_step: int,
    kind: str,
    sae_seed: int,
    role: str,
    partitions: dict[str, np.ndarray],
    alignment_metadata: dict[str, Any] | None,
) -> dict[str, Any]:
    source_manifest = project_source_manifest()
    return {
        "input_dim": int(cache.activations.shape[1]),
        "dictionary_size": config.sae.dictionary_size or config.sae.expansion_factor * int(cache.activations.shape[1]),
        "model_step": model_step,
        "kind": kind,
        "sae_seed": sae_seed,
        "role": role,
        "artifact_version": config.sae.artifact_version,
        "activation_root": str(cache.root),
        "activation_metadata_sha256": sha256_file(cache.root / "metadata.json"),
        "partition_sha256": {
            name: hashlib.sha256(indices.tobytes()).hexdigest()
            for name, indices in partitions.items()
        },
        "partition_rows": {name: len(indices) for name, indices in partitions.items()},
        "normalization_source_model_step": model_step,
        "warm_start_mode": config.sae.warm_start_mode,
        "warm_start_copied_buffers": False,
        "warm_start_alignment": alignment_metadata,
        "source_manifest": source_manifest,
        "source_manifest_sha256": canonical_json_hash(source_manifest),
        "config": config.sae.model_dump(mode="json"),
    }
