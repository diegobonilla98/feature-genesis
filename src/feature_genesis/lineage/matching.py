from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from feature_genesis.config import LineageSection
from feature_genesis.lineage.alignment import OrthogonalAlignment
from feature_genesis.lineage.signatures import FeatureSignatures


@dataclass
class FeatureMatch:
    source: int
    target: int
    score: float
    relation: str
    decoder_similarity: float
    activation_similarity: float
    concept_similarity: float


@dataclass
class PairwiseLineage:
    matches: list[FeatureMatch]
    births: list[int]
    deaths: list[int]
    splits: dict[int, list[int]]
    merges: dict[int, list[int]]
    score_matrix: np.ndarray


def match_feature_sets(
    source: FeatureSignatures,
    target: FeatureSignatures,
    config: LineageSection,
    alignment: OrthogonalAlignment | None = None,
    device: torch.device | str = "cpu",
) -> PairwiseLineage:
    source_decoder = source.decoder if alignment is None else alignment.transform_directions(source.decoder)
    decoder = _cosine_matrix(source_decoder, target.decoder, device)
    activation = _cosine_matrix(source.activation_embedding, target.activation_embedding, device)
    concept = _cosine_matrix(source.concept_profile, target.concept_profile, device)
    score = (
        config.decoder_weight * decoder
        + config.activation_weight * activation
        + config.concept_weight * concept
    )
    normalizer = config.decoder_weight + config.activation_weight + config.concept_weight
    score = score / max(normalizer, 1e-12)
    live_sources = np.flatnonzero(
        source.frequency > config.death_frequency_threshold
    )
    live_targets = np.flatnonzero(
        target.frequency > config.death_frequency_threshold
    )
    if len(live_sources) and len(live_targets):
        local_sources, local_targets = linear_sum_assignment(
            -score[np.ix_(live_sources, live_targets)]
        )
        source_indices = live_sources[local_sources]
        target_indices = live_targets[local_targets]
    else:
        source_indices = np.empty(0, dtype=np.int64)
        target_indices = np.empty(0, dtype=np.int64)
    matches = []
    matched_sources = set()
    matched_targets = set()
    for source_index, target_index in zip(source_indices, target_indices, strict=True):
        value = float(score[source_index, target_index])
        if value < config.continue_threshold:
            continue
        matches.append(
            FeatureMatch(
                source=int(source_index),
                target=int(target_index),
                score=value,
                relation="continue",
                decoder_similarity=float(decoder[source_index, target_index]),
                activation_similarity=float(activation[source_index, target_index]),
                concept_similarity=float(concept[source_index, target_index]),
            )
        )
        matched_sources.add(int(source_index))
        matched_targets.add(int(target_index))
    splits: dict[int, list[int]] = {}
    for source_value in live_sources:
        source_index = int(source_value)
        candidates = live_targets[
            score[source_index, live_targets] >= config.secondary_threshold
        ]
        ordered = candidates[np.argsort(score[source_index, candidates])[::-1]]
        if len(ordered) >= 2 and float(score[source_index, ordered[:2]].sum()) >= config.split_mass_threshold:
            splits[source_index] = [int(value) for value in ordered[:4]]
    merges: dict[int, list[int]] = {}
    for target_value in live_targets:
        target_index = int(target_value)
        candidates = live_sources[
            score[live_sources, target_index] >= config.secondary_threshold
        ]
        ordered = candidates[np.argsort(score[candidates, target_index])[::-1]]
        if len(ordered) >= 2 and float(score[ordered[:2], target_index].sum()) >= config.split_mass_threshold:
            merges[target_index] = [int(value) for value in ordered[:4]]
    related_sources = matched_sources | set(splits)
    related_targets = matched_targets | {target_index for targets in splits.values() for target_index in targets}
    related_targets |= set(merges)
    related_sources |= {source_index for sources in merges.values() for source_index in sources}
    births = [
        index
        for index in range(score.shape[1])
        if index not in related_targets and target.frequency[index] > config.death_frequency_threshold
    ]
    deaths = [
        index
        for index in range(score.shape[0])
        if index not in related_sources and source.frequency[index] > config.death_frequency_threshold
    ]
    for source_index, targets in splits.items():
        for target_index in targets:
            if not any(match.source == source_index and match.target == target_index for match in matches):
                matches.append(
                    FeatureMatch(
                        source=source_index,
                        target=target_index,
                        score=float(score[source_index, target_index]),
                        relation="split",
                        decoder_similarity=float(decoder[source_index, target_index]),
                        activation_similarity=float(activation[source_index, target_index]),
                        concept_similarity=float(concept[source_index, target_index]),
                    )
                )
    for target_index, sources in merges.items():
        for source_index in sources:
            if not any(match.source == source_index and match.target == target_index for match in matches):
                matches.append(
                    FeatureMatch(
                        source=source_index,
                        target=target_index,
                        score=float(score[source_index, target_index]),
                        relation="merge",
                        decoder_similarity=float(decoder[source_index, target_index]),
                        activation_similarity=float(activation[source_index, target_index]),
                        concept_similarity=float(concept[source_index, target_index]),
                    )
                )
    return PairwiseLineage(matches, births, deaths, splits, merges, score.astype(np.float32))


def _cosine_matrix(source: np.ndarray, target: np.ndarray, device: torch.device | str) -> np.ndarray:
    source_tensor = F.normalize(torch.from_numpy(np.array(source, dtype=np.float32, copy=True)).to(device), dim=-1)
    target_tensor = F.normalize(torch.from_numpy(np.array(target, dtype=np.float32, copy=True)).to(device), dim=-1)
    output = source_tensor @ target_tensor.T
    return output.clamp_min(0).cpu().numpy()
