from __future__ import annotations

from typing import Any

import numpy as np
import torch

from feature_genesis.activations import ActivationCache
from feature_genesis.lineage.alignment import OrthogonalAlignment
from feature_genesis.progress import progress
from feature_genesis.sae.core import SparseAutoencoder


@torch.no_grad()
def backward_feature_trajectories(
    final_sae: SparseAutoencoder,
    caches: dict[int, ActivationCache],
    alignments_to_final: dict[int, OrthogonalAlignment | None],
    feature_ids: list[int],
    device: torch.device,
    batch_size: int,
    maximum_alignment_residual_ratio: float,
) -> dict[str, Any]:
    final_sae.eval()
    trajectories = {str(feature): [] for feature in feature_ids}
    for step in progress(sorted(caches), desc="backward trajectories"):
        cache = caches[step]
        alignment = alignments_to_final[step]
        active = torch.zeros(len(feature_ids), dtype=torch.float64, device=device)
        activation_sum = torch.zeros(len(feature_ids), dtype=torch.float64, device=device)
        activation_max = torch.zeros(len(feature_ids), dtype=torch.float32, device=device)
        count = 0
        for start in progress(range(0, len(cache.activations), batch_size), desc=f"backward step {step}", leave=False):
            rows = torch.from_numpy(np.array(cache.activations[start : start + batch_size], dtype=np.float32, copy=True)).to(device)
            if alignment is not None:
                source_mean = torch.from_numpy(alignment.source_mean).to(rows)
                target_mean = torch.from_numpy(alignment.target_mean).to(rows)
                rotation = torch.from_numpy(alignment.rotation).to(rows)
                rows = (rows - source_mean) @ rotation + target_mean
            values = final_sae.encode(rows, inference=True)[:, feature_ids]
            active += values.gt(0).sum(dim=0).double()
            activation_sum += values.sum(dim=0).double()
            activation_max = torch.maximum(activation_max, values.amax(dim=0))
            count += len(rows)
        for local_index, feature in enumerate(feature_ids):
            trajectories[str(feature)].append(
                {
                    "step": step,
                    "positive_frequency": float((active[local_index] / count).cpu()),
                    "mean_positive_response": float((activation_sum[local_index] / active[local_index].clamp_min(1)).cpu()),
                    "maximum_response": float(activation_max[local_index].cpu()),
                    "mean_response": float((activation_sum[local_index] / count).cpu()),
                    "alignment_residual_ratio": 0.0 if alignment is None else alignment.residual_ratio,
                    "alignment_reliable": alignment is None
                    or alignment.residual_ratio <= maximum_alignment_residual_ratio,
                }
            )
    phases = {
        feature: trajectory_phases(rows, maximum_alignment_residual_ratio)
        for feature, rows in trajectories.items()
    }
    return {
        "final_step": max(caches),
        "feature_ids": feature_ids,
        "trajectories": trajectories,
        "phases": phases,
    }


def trajectory_phases(
    rows: list[dict[str, Any]],
    maximum_alignment_residual_ratio: float,
) -> dict[str, Any]:
    reliable_rows = [
        row
        for row in rows
        if row["alignment_residual_ratio"] <= maximum_alignment_residual_ratio
    ]
    if not reliable_rows:
        return {
            "status": "unmeasurable",
            "measurement_window": None,
            "unreliable_before_step": rows[-1]["step"] if rows else None,
            "maximum_alignment_residual_ratio": maximum_alignment_residual_ratio,
        }
    strength = np.asarray(
        [row["mean_response"] for row in reliable_rows],
        dtype=np.float64,
    )
    peak = float(strength.max()) if len(strength) else 0.0
    emergence = sustained_crossing(strength, peak * 0.1)
    consolidation = sustained_crossing(strength, peak * 0.6)
    emergence = 0 if emergence is None else emergence
    consolidation = max(emergence, consolidation if consolidation is not None else len(rows) - 1)
    birth_start = reliable_rows[max(0, emergence - 1)]["step"]
    birth_end = reliable_rows[emergence]["step"]
    left_censored = emergence == 0 and reliable_rows[0]["step"] != rows[0]["step"]
    return {
        "status": "left_censored" if left_censored else "measured",
        "measurement_window": [reliable_rows[0]["step"], reliable_rows[-1]["step"]],
        "unreliable_before_step": reliable_rows[0]["step"],
        "dormant": None if left_censored else [reliable_rows[0]["step"], birth_start],
        "emergence": None
        if left_censored
        else [birth_start, reliable_rows[consolidation]["step"]],
        "consolidation": [
            reliable_rows[consolidation]["step"],
            reliable_rows[-1]["step"],
        ],
        "birth_interval": None if left_censored else [birth_start, birth_end],
        "peak_mean_response": peak,
        "maximum_alignment_residual_ratio": maximum_alignment_residual_ratio,
        "thresholds": {
            "emergence_fraction_of_peak": 0.1,
            "consolidation_fraction_of_peak": 0.6,
            "required_consecutive_checkpoints": 2,
        },
    }


def sustained_crossing(values: np.ndarray, threshold: float) -> int | None:
    if len(values) == 0 or threshold <= 0:
        return None
    for index in range(len(values)):
        end = min(len(values), index + 2)
        if end - index == 2 and np.all(values[index:end] >= threshold):
            return index
    return None
