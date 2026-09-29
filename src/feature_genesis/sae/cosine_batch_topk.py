from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from feature_genesis.sae.batch_topk import BatchTopKSAE, batch_topk
from feature_genesis.sae.core import SAEForward


class CosineBatchTopKSAE(BatchTopKSAE):
    kind = "cosine_batch_topk"

    def __init__(self, *values, per_feature: bool = True, **named):
        super().__init__(*values, **named)
        self.per_feature = per_feature
        self.norm_exponent_base = nn.Parameter(torch.zeros(()))
        if per_feature:
            self.norm_exponent_delta = nn.Parameter(torch.zeros(self.dictionary_size))
            self.log_scale = nn.Parameter(torch.full((self.dictionary_size,), 0.5 * math.log(self.input_dim)))
        else:
            self.register_parameter("norm_exponent_delta", None)
            self.log_scale = nn.Parameter(torch.tensor(0.5 * math.log(self.input_dim)))

    @torch.no_grad()
    def initialize_from_data(
        self,
        samples: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> None:
        super().initialize_from_data(samples, generator=generator)
        normalized = self.normalize_input(samples.float())
        centered = normalized - self.decoder_bias
        mean_norm = centered.norm(dim=-1).mean().clamp_min(1e-6)
        default_scale = self.input_dim**0.5
        resolved_scale = float(mean_norm) if default_scale > 2 * float(mean_norm) else default_scale
        self.norm_exponent_base.zero_()
        if self.norm_exponent_delta is not None:
            self.norm_exponent_delta.zero_()
        self.log_scale.fill_(math.log(resolved_scale))

    def preactivations(self, inputs: torch.Tensor) -> torch.Tensor:
        normalized = self.normalize_input(inputs)
        centered = normalized - self.decoder_bias
        norms = centered.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        input_unit = centered / norms
        encoder_unit = F.normalize(self.encoder, dim=0)
        cosine = input_unit @ encoder_unit
        exponent = self.norm_exponent_base
        if self.norm_exponent_delta is not None:
            exponent = exponent + self.norm_exponent_delta
        scale = torch.exp(norms.log() * exponent + self.log_scale)
        return scale * cosine + self.encoder_bias

    def encode(self, inputs: torch.Tensor, inference: bool = True) -> torch.Tensor:
        acts = F.relu(self.preactivations(inputs))
        if inference and self.inference_threshold.item() > 0:
            return acts * (acts >= self.inference_threshold)
        return batch_topk(acts, self.k)

    def forward(self, inputs: torch.Tensor) -> SAEForward:
        normalized = self.normalize_input(inputs)
        acts = F.relu(self.preactivations(inputs))
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
        exponent = self.norm_exponent_base
        if self.norm_exponent_delta is not None:
            exponent = exponent + self.norm_exponent_delta
        return SAEForward(
            reconstruction=self.denormalize_output(reconstruction_normalized),
            codes=codes,
            loss=loss,
            reconstruction_loss=reconstruction_loss,
            auxiliary_loss=auxiliary_loss,
            l0=codes.ne(0).sum(dim=-1).float().mean(),
            dead_features=(self.inactive_steps >= self.dead_steps).sum(),
            extra={
                "norm_exponent_mean": exponent.detach().float().mean(),
                "norm_exponent_max": exponent.detach().float().max(),
            },
        )

    def metadata(self) -> dict[str, object]:
        metadata = super().metadata()
        exponent = self.norm_exponent_base.detach()
        if self.norm_exponent_delta is not None:
            exponent = exponent + self.norm_exponent_delta.detach()
        metadata.update(
            {
                "per_feature": self.per_feature,
                "norm_exponent_mean": float(exponent.float().mean().cpu()),
                "norm_exponent_max": float(exponent.float().max().cpu()),
            }
        )
        return metadata
