from __future__ import annotations

from feature_genesis.config import Config
from feature_genesis.sae.batch_topk import BatchTopKSAE
from feature_genesis.sae.core import SparseAutoencoder
from feature_genesis.sae.cosine_batch_topk import CosineBatchTopKSAE
from feature_genesis.sae.jumprelu import JumpReLUSAE
from feature_genesis.sae.matching_pursuit import MatchingPursuitSAE
from feature_genesis.sae.matryoshka import MatryoshkaBatchTopKSAE
from feature_genesis.sae.topk import TopKSAE


def dictionary_size(config: Config, input_dim: int) -> int:
    return config.sae.dictionary_size or config.sae.expansion_factor * input_dim


def build_sae(config: Config, input_dim: int, kind: str | None = None) -> SparseAutoencoder:
    resolved_kind = kind or config.sae.kind
    size = dictionary_size(config, input_dim)
    common = {
        "input_dim": input_dim,
        "dictionary_size": size,
    }
    if resolved_kind == "jumprelu":
        return JumpReLUSAE(
            **common,
            sparsity_coefficient=config.sae.jump_sparsity_coefficient,
            initial_threshold=config.sae.jump_initial_threshold,
            bandwidth=config.sae.jump_bandwidth,
            dead_steps=config.sae.dead_steps,
            initial_l0=config.sae.jump_initial_l0 if config.sae.jump_initial_l0 is not None else config.sae.k,
        )
    if resolved_kind == "matching_pursuit":
        return MatchingPursuitSAE(
            **common,
            steps=config.sae.mp_steps,
            residual_tolerance=config.sae.mp_residual_tolerance,
            dead_steps=config.sae.dead_steps,
        )
    topk_common = {
        **common,
        "k": config.sae.k,
        "aux_weight": config.sae.aux_weight,
        "aux_k": config.sae.aux_k,
        "dead_steps": config.sae.dead_steps,
        "threshold_ema": config.sae.threshold_ema,
    }
    if resolved_kind == "topk":
        return TopKSAE(**topk_common)
    if resolved_kind == "batch_topk":
        return BatchTopKSAE(**topk_common)
    if resolved_kind == "cosine_batch_topk":
        return CosineBatchTopKSAE(**topk_common, per_feature=config.sae.cosine_per_feature)
    if resolved_kind == "matryoshka":
        return MatryoshkaBatchTopKSAE(
            **topk_common,
            prefixes=config.sae.matryoshka_prefixes,
            prefix_weights=config.sae.matryoshka_weights,
        )
    raise ValueError(f"Unknown SAE kind: {resolved_kind}")
