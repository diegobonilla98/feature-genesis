from __future__ import annotations

import torch
import torch.nn.functional as F

from feature_genesis.sae.batch_topk import BatchTopKSAE, batch_topk
from feature_genesis.sae.core import SAEForward


class MatryoshkaBatchTopKSAE(BatchTopKSAE):
    kind = "matryoshka"

    def __init__(self, *values, prefixes: list[int], prefix_weights: list[float], **named):
        super().__init__(*values, **named)
        if len(prefixes) != len(prefix_weights):
            raise ValueError("prefixes and prefix_weights must have equal length")
        merged: dict[int, float] = {}
        for prefix, weight in zip(prefixes, prefix_weights, strict=True):
            resolved = min(max(1, int(prefix)), self.dictionary_size)
            merged[resolved] = merged.get(resolved, 0.0) + float(weight)
        if self.dictionary_size not in merged:
            merged[self.dictionary_size] = max(merged.values())
        self.prefixes = sorted(merged)
        total = sum(merged.values())
        self.prefix_weights = [merged[prefix] / total for prefix in self.prefixes]

    def forward(self, inputs: torch.Tensor) -> SAEForward:
        normalized = self.normalize_input(inputs)
        acts = F.relu((normalized - self.decoder_bias) @ self.encoder + self.encoder_bias)
        if self.training:
            full_codes = batch_topk(acts, self.k)
        else:
            full_codes = acts * (acts >= self.inference_threshold) if self.inference_threshold.item() > 0 else batch_topk(acts, self.k)
        prefix_losses = []
        full_reconstruction = None
        for prefix, weight in zip(self.prefixes, self.prefix_weights, strict=True):
            reconstruction = full_codes[:, :prefix] @ self.decoder[:prefix] + self.decoder_bias
            prefix_losses.append(weight * F.mse_loss(reconstruction.float(), normalized.float()))
            if prefix == self.dictionary_size:
                full_reconstruction = reconstruction
        if full_reconstruction is None:
            raise RuntimeError("Full dictionary prefix was not evaluated")
        reconstruction_loss = torch.stack(prefix_losses).sum()
        auxiliary_loss = self._auxiliary_loss(normalized, full_reconstruction, acts)
        loss = reconstruction_loss + auxiliary_loss
        if self.training:
            self.update_threshold(full_codes)
            self.update_inactivity(full_codes)
        return SAEForward(
            reconstruction=self.denormalize_output(full_reconstruction),
            codes=full_codes,
            loss=loss,
            reconstruction_loss=reconstruction_loss,
            auxiliary_loss=auxiliary_loss,
            l0=full_codes.ne(0).sum(dim=-1).float().mean(),
            dead_features=(self.inactive_steps >= self.dead_steps).sum(),
            extra={f"prefix_{prefix}": value.detach() for prefix, value in zip(self.prefixes, prefix_losses, strict=True)},
        )
