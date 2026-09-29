from __future__ import annotations

import torch
import torch.nn.functional as F

from feature_genesis.sae.core import SAEForward
from feature_genesis.sae.topk import TopKSAE


class BatchTopKSAE(TopKSAE):
    kind = "batch_topk"

    def encode(self, inputs: torch.Tensor, inference: bool = True) -> torch.Tensor:
        acts = F.relu(self.preactivations(inputs))
        if inference and self.inference_threshold.item() > 0:
            return acts * (acts >= self.inference_threshold)
        return batch_topk(acts, self.k)

    def forward(self, inputs: torch.Tensor) -> SAEForward:
        normalized = self.normalize_input(inputs)
        acts = F.relu((normalized - self.decoder_bias) @ self.encoder + self.encoder_bias)
        if self.training:
            codes = batch_topk(acts, self.k)
        else:
            codes = acts * (acts >= self.inference_threshold) if self.inference_threshold.item() > 0 else batch_topk(acts, self.k)
        reconstruction_normalized = self.decode_normalized(codes)
        reconstruction_loss = F.mse_loss(reconstruction_normalized.float(), normalized.float())
        auxiliary_loss = self._auxiliary_loss(normalized, reconstruction_normalized, acts)
        loss = reconstruction_loss + auxiliary_loss
        if self.training:
            self.update_threshold(codes)
            self.update_inactivity(codes)
        return SAEForward(
            reconstruction=self.denormalize_output(reconstruction_normalized),
            codes=codes,
            loss=loss,
            reconstruction_loss=reconstruction_loss,
            auxiliary_loss=auxiliary_loss,
            l0=codes.ne(0).sum(dim=-1).float().mean(),
            dead_features=(self.inactive_steps >= self.dead_steps).sum(),
            extra={"threshold": self.inference_threshold.detach().clone()},
        )


def batch_topk(acts: torch.Tensor, k: int) -> torch.Tensor:
    count = min(acts.numel(), acts.shape[0] * k)
    values, indices = torch.topk(acts.reshape(-1), count)
    return torch.zeros_like(acts).reshape(-1).scatter(0, indices, values).reshape_as(acts)
