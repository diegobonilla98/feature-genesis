from __future__ import annotations

from collections.abc import Iterable

import torch

from feature_genesis.models.hooks import rows_to_spatial, spatial_to_rows


def feature_node_id(layer: str, feature_id: int) -> str:
    return f"{layer}:feature:{feature_id}"


def logit_node_id(class_id: int) -> str:
    return f"logit:{class_id}"


def encode_spatial(sae, activation: torch.Tensor) -> tuple[torch.Tensor, tuple[int, ...]]:
    rows, shape = spatial_to_rows(activation)
    codes = sae.encode(rows, inference=True)
    if len(shape) == 2:
        return codes[:, :, None, None], shape
    batch, _, height, width = shape
    return codes.reshape(batch, height, width, sae.dictionary_size).permute(0, 3, 1, 2), shape


def max_feature_values(codes: torch.Tensor, feature_id: int) -> torch.Tensor:
    return codes[:, feature_id].flatten(1).amax(dim=-1)


def decoded_feature_contributions(sae, codes: torch.Tensor, feature_ids: Iterable[int]) -> torch.Tensor:
    ids = sorted(set(int(value) for value in feature_ids))
    if not ids:
        return codes.new_zeros((codes.shape[0], sae.input_dim))
    directions = sae.decoder_directions()[ids].to(codes)
    return sae.input_scale.to(codes) * (codes[:, ids] @ directions)


def modify_sparse_features(
    activation: torch.Tensor,
    sae,
    feature_ids: Iterable[int],
    mode: str,
) -> torch.Tensor:
    rows, shape = spatial_to_rows(activation)
    codes = sae.encode(rows, inference=True)
    selected = sorted(set(int(value) for value in feature_ids))
    if mode == "ablate_selected":
        removed = decoded_feature_contributions(sae, codes, selected)
    elif mode == "keep_selected":
        mask = torch.ones(sae.dictionary_size, dtype=torch.bool, device=codes.device)
        if selected:
            mask[torch.tensor(selected, device=codes.device)] = False
        removed = decoded_feature_contributions(sae, codes, mask.nonzero().flatten().tolist())
    elif mode == "remove_all":
        removed = decoded_feature_contributions(sae, codes, range(sae.dictionary_size))
    else:
        raise ValueError(f"Unknown sparse intervention mode: {mode}")
    return rows_to_spatial(rows - removed.to(rows), shape)


def centered_logit(logits: torch.Tensor, class_id: int) -> torch.Tensor:
    return logits[:, class_id] - logits.mean(dim=-1)


def contribution_scores(
    activation: torch.Tensor,
    gradient: torch.Tensor,
    sae,
) -> torch.Tensor:
    rows, shape = spatial_to_rows(activation)
    gradient_rows, _ = spatial_to_rows(gradient)
    codes = sae.encode(rows, inference=True)
    directions = sae.input_scale.to(rows) * sae.decoder_directions().to(rows)
    effects = codes * (gradient_rows @ directions.T)
    if len(shape) == 2:
        return effects
    batch, _, height, width = shape
    return effects.reshape(batch, height * width, sae.dictionary_size).sum(dim=1)
