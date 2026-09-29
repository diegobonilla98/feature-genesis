from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from torch import nn

from feature_genesis.models.factory import resolve_module


class ActivationCapture:
    def __init__(self, model: nn.Module, layer: str, detach: bool = False):
        self.model = model
        self.layer = layer
        self.detach = detach
        self.value: torch.Tensor | None = None
        module = resolve_module(model, layer)
        self.handle = module.register_forward_hook(self._hook)

    def _hook(self, module: nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        self.value = output.detach() if self.detach else output

    def close(self) -> None:
        self.handle.remove()

    def __enter__(self) -> "ActivationCapture":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


class ActivationModifier:
    def __init__(self, model: nn.Module, layer: str, function: Callable[[torch.Tensor], torch.Tensor]):
        self.function = function
        module = resolve_module(model, layer)
        self.handle = module.register_forward_hook(self._hook)

    def _hook(self, module: nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> torch.Tensor:
        return self.function(output)

    def close(self) -> None:
        self.handle.remove()

    def __enter__(self) -> "ActivationModifier":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


class MultiActivationCapture:
    def __init__(self, model: nn.Module, layers: list[str], detach: bool = False):
        self.model = model
        self.layers = list(layers)
        self.detach = detach
        self.values: dict[str, torch.Tensor] = {}
        self.handles = [
            resolve_module(model, layer).register_forward_hook(self._hook(layer))
            for layer in self.layers
        ]

    def _hook(self, layer: str):
        def capture(module: nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
            self.values[layer] = output.detach() if self.detach else output

        return capture

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()

    def __enter__(self) -> "MultiActivationCapture":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


class MultiActivationModifier:
    def __init__(self, model: nn.Module, functions: dict[str, Callable[[torch.Tensor], torch.Tensor]]):
        self.functions = dict(functions)
        self.handles = [
            resolve_module(model, layer).register_forward_hook(self._hook(layer))
            for layer in self.functions
        ]

    def _hook(self, layer: str):
        def modify(module: nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> torch.Tensor:
            return self.functions[layer](output)

        return modify

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()

    def __enter__(self) -> "MultiActivationModifier":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


def spatial_to_rows(activation: torch.Tensor) -> tuple[torch.Tensor, tuple[int, ...]]:
    if activation.ndim == 2:
        return activation, tuple(activation.shape)
    if activation.ndim != 4:
        raise ValueError(f"Expected rank-2 or rank-4 activation, got {activation.shape}")
    batch, channels, height, width = activation.shape
    rows = activation.permute(0, 2, 3, 1).reshape(batch * height * width, channels)
    return rows, (batch, channels, height, width)


def rows_to_spatial(rows: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
    if len(shape) == 2:
        return rows
    batch, channels, height, width = shape
    return rows.reshape(batch, height, width, channels).permute(0, 3, 1, 2).contiguous()
