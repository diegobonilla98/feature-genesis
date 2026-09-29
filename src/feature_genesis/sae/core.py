from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class SAEForward:
    reconstruction: torch.Tensor
    codes: torch.Tensor
    loss: torch.Tensor
    reconstruction_loss: torch.Tensor
    auxiliary_loss: torch.Tensor
    l0: torch.Tensor
    dead_features: torch.Tensor
    extra: dict[str, torch.Tensor]


class SparseAutoencoder(nn.Module):
    kind = "base"

    def __init__(self, input_dim: int, dictionary_size: int, threshold_ema: float = 0.99):
        super().__init__()
        self.input_dim = input_dim
        self.dictionary_size = dictionary_size
        self.threshold_ema = threshold_ema
        self.encoder = nn.Parameter(torch.empty(input_dim, dictionary_size))
        self.decoder = nn.Parameter(torch.empty(dictionary_size, input_dim))
        self.encoder_bias = nn.Parameter(torch.zeros(dictionary_size))
        self.decoder_bias = nn.Parameter(torch.zeros(input_dim))
        self.register_buffer("input_scale", torch.ones(()))
        self.register_buffer("inference_threshold", torch.zeros(()))
        self.register_buffer("inactive_steps", torch.zeros(dictionary_size, dtype=torch.long))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.encoder)
        with torch.no_grad():
            self.decoder.copy_(self.encoder.T)
            self.normalize_decoder_()

    @torch.no_grad()
    def initialize_from_data(
        self,
        samples: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> None:
        samples = samples.float()
        center = geometric_median(samples)
        centered = samples - center
        scale = centered.pow(2).mean().sqrt().clamp_min(1e-6)
        normalized = centered / scale
        if generator is None:
            generator = torch.Generator(device=normalized.device).manual_seed(414213)
        directions = low_coherence_frame(normalized, self.dictionary_size, generator)
        self.input_scale.copy_(scale)
        self.decoder_bias.copy_(center / scale)
        self.decoder.copy_(directions)
        self.encoder.copy_(directions.T)
        self.encoder_bias.zero_()
        self.inference_threshold.zero_()
        self.inactive_steps.zero_()

    def normalize_input(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs / self.input_scale

    def denormalize_output(self, outputs: torch.Tensor) -> torch.Tensor:
        return outputs * self.input_scale

    def preactivations(self, inputs: torch.Tensor) -> torch.Tensor:
        normalized = self.normalize_input(inputs)
        return (normalized - self.decoder_bias) @ self.encoder + self.encoder_bias

    def decode_normalized(self, codes: torch.Tensor) -> torch.Tensor:
        return codes @ self.decoder + self.decoder_bias

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        return self.denormalize_output(self.decode_normalized(codes))

    def encode(self, inputs: torch.Tensor, inference: bool = True) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, inputs: torch.Tensor) -> SAEForward:
        raise NotImplementedError

    def decoder_directions(self) -> torch.Tensor:
        return F.normalize(self.decoder, dim=-1)

    @torch.no_grad()
    def project_decoder_gradients_(self) -> None:
        if self.decoder.grad is None:
            return
        directions = self.decoder_directions()
        parallel = (self.decoder.grad * directions).sum(dim=-1, keepdim=True) * directions
        self.decoder.grad.sub_(parallel)

    @torch.no_grad()
    def normalize_decoder_(self) -> None:
        self.decoder.copy_(F.normalize(self.decoder, dim=-1))

    @torch.no_grad()
    def update_inactivity(self, codes: torch.Tensor) -> None:
        active = codes.detach().ne(0).any(dim=0)
        self.inactive_steps.add_(1)
        self.inactive_steps[active] = 0

    @torch.no_grad()
    def update_threshold(self, selected_codes: torch.Tensor) -> None:
        positive = selected_codes[selected_codes > 0]
        if positive.numel() == 0:
            return
        batch_threshold = positive.min()
        if self.inference_threshold.item() == 0:
            self.inference_threshold.copy_(batch_threshold)
        else:
            self.inference_threshold.mul_(self.threshold_ema).add_(batch_threshold * (1 - self.threshold_ema))

    def metadata(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "input_dim": self.input_dim,
            "dictionary_size": self.dictionary_size,
            "input_scale": float(self.input_scale.detach().cpu()),
            "inference_threshold": float(self.inference_threshold.detach().cpu()),
        }


def geometric_median(values: torch.Tensor, iterations: int = 32, epsilon: float = 1e-6) -> torch.Tensor:
    estimate = values.mean(dim=0)
    for _ in range(iterations):
        distances = (values - estimate).norm(dim=-1).clamp_min(epsilon)
        weights = distances.reciprocal()
        updated = (values * weights[:, None]).sum(dim=0) / weights.sum()
        if (updated - estimate).norm() < epsilon:
            estimate = updated
            break
        estimate = updated
    return estimate


@torch.no_grad()
def low_coherence_frame(
    normalized_samples: torch.Tensor,
    dictionary_size: int,
    generator: torch.Generator,
) -> torch.Tensor:
    input_dim = normalized_samples.shape[1]
    covariance = normalized_samples.T @ normalized_samples / max(1, len(normalized_samples))
    _, eigenvectors = torch.linalg.eigh(covariance)
    blocks = [eigenvectors.flip(dims=[1]).T]
    while sum(block.shape[0] for block in blocks) < dictionary_size:
        random_matrix = torch.randn(
            input_dim,
            input_dim,
            generator=generator,
            device=normalized_samples.device,
            dtype=normalized_samples.dtype,
        )
        orthogonal, _ = torch.linalg.qr(random_matrix)
        blocks.append(orthogonal.T)
    return torch.cat(blocks, dim=0)[:dictionary_size].contiguous()
