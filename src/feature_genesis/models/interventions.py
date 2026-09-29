from __future__ import annotations

import torch

from feature_genesis.models.hooks import rows_to_spatial, spatial_to_rows


def ablate_sae_feature(activation: torch.Tensor, sae, feature_id: int, scale: float = 1.0) -> torch.Tensor:
    rows, shape = spatial_to_rows(activation)
    with torch.no_grad():
        codes = sae.encode(rows, inference=True)
    contribution = (
        sae.input_scale
        * codes[:, feature_id : feature_id + 1]
        * sae.decoder_directions()[feature_id : feature_id + 1]
    )
    modified = rows - scale * contribution.to(rows.dtype)
    return rows_to_spatial(modified, shape)


def steer_sae_feature(activation: torch.Tensor, sae, feature_id: int, amount: float) -> torch.Tensor:
    rows, shape = spatial_to_rows(activation)
    direction = sae.decoder_directions()[feature_id].to(rows)
    modified = rows + amount * direction.unsqueeze(0)
    return rows_to_spatial(modified, shape)
