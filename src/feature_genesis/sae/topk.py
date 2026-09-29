from __future__ import annotations

import torch
import torch.nn.functional as F

from feature_genesis.sae.core import SAEForward, SparseAutoencoder


class TopKSAE(SparseAutoencoder):
    kind = "topk"

    def __init__(
        self,
        input_dim: int,
        dictionary_size: int,
        k: int,
        aux_weight: float,
        aux_k: int,
        dead_steps: int,
        threshold_ema: float = 0.99,
    ):
        super().__init__(input_dim, dictionary_size, threshold_ema)
        self.k = k
        self.aux_weight = aux_weight
        self.aux_k = aux_k
        self.dead_steps = dead_steps

    def encode(self, inputs: torch.Tensor, inference: bool = True) -> torch.Tensor:
        acts = F.relu(self.preactivations(inputs))
        values, indices = torch.topk(acts, min(self.k, self.dictionary_size), dim=-1)
        return torch.zeros_like(acts).scatter(-1, indices, values)

    def forward(self, inputs: torch.Tensor) -> SAEForward:
        normalized = self.normalize_input(inputs)
        acts = F.relu((normalized - self.decoder_bias) @ self.encoder + self.encoder_bias)
        values, indices = torch.topk(acts, min(self.k, self.dictionary_size), dim=-1)
        codes = torch.zeros_like(acts).scatter(-1, indices, values)
        reconstruction_normalized = self.decode_normalized(codes)
        reconstruction_loss = F.mse_loss(reconstruction_normalized.float(), normalized.float())
        auxiliary_loss = self._auxiliary_loss(normalized, reconstruction_normalized, acts)
        loss = reconstruction_loss + auxiliary_loss
        if self.training:
            self.update_inactivity(codes)
        return SAEForward(
            reconstruction=self.denormalize_output(reconstruction_normalized),
            codes=codes,
            loss=loss,
            reconstruction_loss=reconstruction_loss,
            auxiliary_loss=auxiliary_loss,
            l0=codes.ne(0).sum(dim=-1).float().mean(),
            dead_features=(self.inactive_steps >= self.dead_steps).sum(),
            extra={},
        )

    def _auxiliary_loss(self, normalized: torch.Tensor, reconstruction: torch.Tensor, acts: torch.Tensor) -> torch.Tensor:
        dead = self.inactive_steps >= self.dead_steps
        if not dead.any() or self.aux_weight == 0:
            return normalized.new_zeros(())
        residual = normalized - reconstruction.detach()
        dead_acts = acts[:, dead]
        k = min(self.aux_k, dead_acts.shape[1])
        values, indices = torch.topk(dead_acts, k, dim=-1)
        codes = torch.zeros_like(dead_acts).scatter(-1, indices, values)
        auxiliary_reconstruction = codes @ self.decoder[dead]
        return self.aux_weight * F.mse_loss(auxiliary_reconstruction.float(), residual.float())
