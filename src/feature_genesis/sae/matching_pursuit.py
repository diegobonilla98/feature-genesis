from __future__ import annotations

import torch
import torch.nn.functional as F

from feature_genesis.sae.core import SAEForward, SparseAutoencoder, low_coherence_frame


class MatchingPursuitSAE(SparseAutoencoder):
    kind = "matching_pursuit"

    def __init__(
        self,
        input_dim: int,
        dictionary_size: int,
        steps: int,
        residual_tolerance: float = 0.0,
        dead_steps: int = 200,
    ):
        super().__init__(input_dim, dictionary_size)
        del self.encoder
        del self.encoder_bias
        self.steps = steps
        self.residual_tolerance = residual_tolerance
        self.dead_steps = dead_steps

    def reset_parameters(self) -> None:
        torch.nn.init.kaiming_uniform_(self.decoder)
        with torch.no_grad():
            self.normalize_decoder_()

    @torch.no_grad()
    def initialize_from_data(
        self,
        samples: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> None:
        samples = samples.float()
        center = samples.mean(dim=0)
        centered = samples - center
        scale = centered.pow(2).mean().sqrt().clamp_min(1e-6)
        normalized = centered / scale
        if generator is None:
            generator = torch.Generator(device=normalized.device).manual_seed(271828)
        directions = low_coherence_frame(normalized, self.dictionary_size, generator)
        self.input_scale.copy_(scale)
        self.decoder_bias.copy_(center / scale)
        self.decoder.copy_(directions)
        self.inactive_steps.zero_()

    def preactivations(self, inputs: torch.Tensor) -> torch.Tensor:
        normalized = self.normalize_input(inputs)
        return F.relu((normalized - self.decoder_bias) @ self.decoder_directions().T)

    def encode(self, inputs: torch.Tensor, inference: bool = True) -> torch.Tensor:
        normalized = self.normalize_input(inputs)
        codes, _ = self._decompose(normalized)
        return codes

    def forward(self, inputs: torch.Tensor) -> SAEForward:
        normalized = self.normalize_input(inputs)
        codes, reconstruction = self._decompose(normalized)
        reconstruction_loss = F.mse_loss(reconstruction.float(), normalized.float())
        if self.training:
            self.update_inactivity(codes)
        return SAEForward(
            reconstruction=self.denormalize_output(reconstruction),
            codes=codes,
            loss=reconstruction_loss,
            reconstruction_loss=reconstruction_loss,
            auxiliary_loss=normalized.new_zeros(()),
            l0=codes.ne(0).sum(dim=-1).float().mean(),
            dead_features=(self.inactive_steps >= self.dead_steps).sum(),
            extra={},
        )

    def decompose_trace(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        normalized = self.normalize_input(inputs)
        return self._decompose(normalized, return_trace=True)

    def _decompose(
        self,
        normalized: torch.Tensor,
        return_trace: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        directions = self.decoder_directions()
        residual = normalized - self.decoder_bias
        reconstruction = self.decoder_bias.unsqueeze(0).expand_as(normalized).clone()
        codes = normalized.new_zeros((normalized.shape[0], self.dictionary_size))
        trace = [residual.detach().clone()] if return_trace else []
        for _ in range(self.steps):
            scores = residual @ directions.T
            values, indices = scores.max(dim=-1)
            values = F.relu(values)
            if self.residual_tolerance > 0:
                values = values * (residual.norm(dim=-1) > self.residual_tolerance)
            selected = directions[indices]
            contribution = values[:, None] * selected
            reconstruction = reconstruction + contribution
            residual = residual - contribution
            codes = codes.scatter_add(1, indices[:, None], values[:, None])
            if return_trace:
                trace.append(residual.detach().clone())
        if return_trace:
            return codes, reconstruction, trace
        return codes, reconstruction
