from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn
from torchvision.models import resnet18, resnet34

from feature_genesis.config import Config


class TinyCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 24, 3, padding=1, bias=False),
            nn.GroupNorm(6, 24),
            nn.SiLU(),
            nn.Conv2d(24, 48, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, 48),
            nn.SiLU(),
            nn.Conv2d(48, 64, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.features(inputs)
        pooled = self.pool(features).flatten(1)
        return self.classifier(pooled)


class QuickDrawCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 64, 5, padding=2, bias=False),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(16, 128),
            nn.SiLU(),
            nn.Conv2d(128, 128, 3, padding=1, bias=False),
            nn.GroupNorm(16, 128),
            nn.SiLU(),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(32, 256),
            nn.SiLU(),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.GroupNorm(32, 256),
            nn.SiLU(),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        values = self.block1(inputs)
        values = self.block2(values)
        values = self.block3(values)
        return self.classifier(self.pool(values).flatten(1))


def build_model(config: Config, num_classes: int) -> nn.Module:
    resolved_classes = config.model.num_classes or num_classes
    if config.model.name == "quickdraw_cnn":
        return QuickDrawCNN(resolved_classes)
    if config.model.name == "tiny_cnn":
        return TinyCNN(resolved_classes)
    groups = config.model.group_norm_groups

    def norm_layer(channels: int) -> nn.GroupNorm:
        resolved_groups = min(groups, channels)
        while channels % resolved_groups != 0:
            resolved_groups -= 1
        return nn.GroupNorm(resolved_groups, channels)

    if config.model.name == "quickdraw_resnet34_gn":
        model = resnet34(weights=None, norm_layer=norm_layer, zero_init_residual=True)
        model.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        model.fc = nn.Linear(model.fc.in_features, resolved_classes)
        if config.train.channels_last:
            model = model.to(memory_format=torch.channels_last)
        return model

    model = resnet18(weights=None, norm_layer=norm_layer, zero_init_residual=True)
    model.fc = nn.Linear(model.fc.in_features, resolved_classes)
    return model


def resolve_module(model: nn.Module, path: str) -> nn.Module:
    current: nn.Module = model
    for component in path.split("."):
        if component.isdigit():
            current = current[int(component)]
        else:
            current = getattr(current, component)
    return current


def parameter_scope(model: nn.Module, hook_layer: str, scope: str) -> list[tuple[str, nn.Parameter]]:
    if scope == "all":
        return [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    if scope == "classifier":
        prefixes = ("fc.", "classifier.")
        return [(name, parameter) for name, parameter in model.named_parameters() if name.startswith(prefixes)]
    prefix = hook_layer.split(".")[0] + "."
    return [(name, parameter) for name, parameter in model.named_parameters() if name.startswith(prefix)]
