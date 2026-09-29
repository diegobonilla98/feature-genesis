from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Function

from feature_genesis.sae.core import SAEForward, SparseAutoencoder


class HeavisideSTE(Function):
    @staticmethod
    def forward(ctx, values: torch.Tensor, bandwidth: float) -> torch.Tensor:
        ctx.save_for_backward(values)
        ctx.bandwidth = bandwidth
        return values.gt(0).to(values.dtype)

    @staticmethod
    def backward(ctx, gradient: torch.Tensor) -> tuple[torch.Tensor, None]:
        (values,) = ctx.saved_tensors
        bandwidth = ctx.bandwidth
        rectangle = (values.abs() < 0.5 * bandwidth).to(values.dtype)
        return gradient * rectangle / bandwidth, None


class JumpReLUFunction(Function):
    @staticmethod
    def forward(ctx, values: torch.Tensor, threshold: torch.Tensor, bandwidth: float) -> torch.Tensor:
        ctx.save_for_backward(values, threshold)
        ctx.bandwidth = bandwidth
        return values * values.gt(threshold).to(values.dtype)

    @staticmethod
    def backward(ctx, gradient: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, None]:
        values, threshold = ctx.saved_tensors
        bandwidth = ctx.bandwidth
        active = values.gt(threshold).to(values.dtype)
        rectangle = ((values - threshold).abs() < 0.5 * bandwidth).to(values.dtype)
        value_gradient = gradient * active
        threshold_gradient = gradient * (-threshold / bandwidth) * rectangle
        return value_gradient, threshold_gradient, None


class JumpReLUSAE(SparseAutoencoder):
    kind = "jumprelu"

    def __init__(
        self,
        input_dim: int,
        dictionary_size: int,
        sparsity_coefficient: float,
        initial_threshold: float,
        bandwidth: float,
        dead_steps: int,
        initial_l0: int | None = None,
    ):
        super().__init__(input_dim, dictionary_size)
        if sparsity_coefficient < 0:
            raise ValueError("sparsity_coefficient must be non-negative")
        if initial_threshold <= 0 or bandwidth <= 0:
            raise ValueError("initial_threshold and bandwidth must be positive")
        self.sparsity_coefficient = sparsity_coefficient
        self.bandwidth = bandwidth
        self.dead_steps = dead_steps
        self.initial_l0 = initial_l0
        self.log_threshold = nn.Parameter(torch.full((dictionary_size,), math.log(initial_threshold)))
        self.register_buffer("sparsity_scale", torch.ones(()))

    @property
    def threshold(self) -> torch.Tensor:
        return self.log_threshold.exp()

    @torch.no_grad()
    def initialize_from_data(
        self,
        samples: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> None:
        super().initialize_from_data(samples, generator=generator)
        if self.initial_l0 is None:
            return
        target_l0 = min(max(1, self.initial_l0), self.dictionary_size)
        preactivations = F.relu(self.preactivations(samples.float())).reshape(-1)
        keep = min(preactivations.numel(), samples.shape[0] * target_l0)
        threshold = preactivations.topk(keep).values.min().clamp_min(1e-6)
        self.log_threshold.fill_(threshold.log())

    def encode(self, inputs: torch.Tensor, inference: bool = True) -> torch.Tensor:
        preactivations = F.relu(self.preactivations(inputs))
        return preactivations * preactivations.gt(self.threshold).to(preactivations.dtype)

    def forward(self, inputs: torch.Tensor) -> SAEForward:
        normalized = self.normalize_input(inputs)
        preactivations = F.relu((normalized - self.decoder_bias) @ self.encoder + self.encoder_bias)
        codes = JumpReLUFunction.apply(preactivations, self.threshold, self.bandwidth)
        reconstruction_normalized = self.decode_normalized(codes)
        reconstruction_loss = F.mse_loss(reconstruction_normalized.float(), normalized.float())
        active = HeavisideSTE.apply(preactivations - self.threshold, self.bandwidth)
        l0 = active.sum(dim=-1).mean()
        sparsity_loss = self.sparsity_scale * self.sparsity_coefficient * l0
        loss = reconstruction_loss + sparsity_loss
        if self.training:
            self.update_inactivity(codes)
        return SAEForward(
            reconstruction=self.denormalize_output(reconstruction_normalized),
            codes=codes,
            loss=loss,
            reconstruction_loss=reconstruction_loss,
            auxiliary_loss=sparsity_loss,
            l0=l0.detach(),
            dead_features=(self.inactive_steps >= self.dead_steps).sum(),
            extra={
                "threshold_mean": self.threshold.detach().mean(),
                "threshold_median": self.threshold.detach().median(),
                "sparsity_scale": self.sparsity_scale.detach().clone(),
            },
        )

    def metadata(self) -> dict[str, object]:
        metadata = super().metadata()
        metadata.update(
            {
                "sparsity_coefficient": self.sparsity_coefficient,
                "bandwidth": self.bandwidth,
                "initial_l0": self.initial_l0,
                "threshold_mean": float(self.threshold.detach().mean().cpu()),
                "threshold_median": float(self.threshold.detach().median().cpu()),
            }
        )
        return metadata
