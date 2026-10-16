"""Классификаторы спектрограмм с нуля, без аудио-предобучения."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from stage2.bounds import (
    CONV_KERNEL,
    FAMILY_LCNN,
    FAMILY_RESNET18,
    LCNN_BLOCKS,
    MAXPOOL_KERNEL,
    MAXPOOL_PADDING,
    MAXPOOL_STRIDE,
    POOL_KERNEL,
    POOL_STRIDE,
    RESNET18_BLOCKS_PER_STAGE,
    RESNET18_CHANNEL_FACTORS,
    STAGE_STRIDE,
    STEM_KERNEL,
    STEM_STRIDE,
)


class MaxFeatureMap(nn.Module):
    """Активация Light-CNN: попарный максимум по каналам."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        left, right = torch.chunk(x, 2, dim=1)
        return torch.max(left, right)


class SafePool2d(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=POOL_KERNEL, stride=POOL_STRIDE)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] < POOL_KERNEL or x.shape[-2] < POOL_KERNEL:
            return x
        return self.pool(x)


class LCNN(nn.Module):
    """CNN с max-feature-map. width: число каналов после каждого MFM."""

    def __init__(self, width: int, dropout: float) -> None:
        super().__init__()
        blocks = []
        in_ch = 1
        for _ in range(LCNN_BLOCKS):
            blocks.extend(
                [
                    nn.Conv2d(in_ch, width * 2, CONV_KERNEL, padding=CONV_KERNEL // 2, bias=False),
                    nn.BatchNorm2d(width * 2),
                    MaxFeatureMap(),
                    SafePool2d(),
                ]
            )
            in_ch = width
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(width, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.pool(self.features(x)).flatten(1)
        return self.fc(self.drop(h)).view(-1)


class BasicBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, CONV_KERNEL, stride=stride, padding=CONV_KERNEL // 2, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, CONV_KERNEL, padding=CONV_KERNEL // 2, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x), inplace=True)


class ResNet18(nn.Module):
    """Схема ResNet-18 на одноканальной спектрограмме. Веса случайные."""

    def __init__(self, width: int, dropout: float) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, width, STEM_KERNEL, stride=STEM_STRIDE, padding=STEM_KERNEL // 2, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
        )
        self.maxpool = nn.MaxPool2d(kernel_size=MAXPOOL_KERNEL, stride=MAXPOOL_STRIDE, padding=MAXPOOL_PADDING)
        stages = []
        in_ch = width
        for factor, n_blocks in zip(RESNET18_CHANNEL_FACTORS, RESNET18_BLOCKS_PER_STAGE, strict=True):
            out_ch = width * factor
            stride = 1 if factor == RESNET18_CHANNEL_FACTORS[0] else STAGE_STRIDE
            layers = [BasicBlock(in_ch, out_ch, stride)]
            for _ in range(1, n_blocks):
                layers.append(BasicBlock(out_ch, out_ch, 1))
            stages.append(nn.Sequential(*layers))
            in_ch = out_ch
        self.stages = nn.Sequential(*stages)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(in_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.stages(self.maxpool(self.stem(x)))
        h = self.pool(h).flatten(1)
        return self.fc(self.drop(h)).view(-1)


def build_spectrogram_model(family: str, lcnn_width: int, resnet_width: int, dropout: float) -> nn.Module:
    if family == FAMILY_LCNN:
        return LCNN(width=int(lcnn_width), dropout=float(dropout))
    if family == FAMILY_RESNET18:
        return ResNet18(width=int(resnet_width), dropout=float(dropout))
    raise ValueError(f"unknown family {family}")


def spectrogram_factories(params: dict, specs, labels):
    """Фабрики модели и датасета по зафиксированному словарю параметров этапа 2."""
    import numpy as np

    from stage2.bounds import LCNN_WIDTH_CHOICES, RESNET_WIDTH_CHOICES
    from stage2.datautil import SpecDataset

    label_values = np.asarray(labels, dtype=np.float32)

    def model_fn():
        return build_spectrogram_model(
            str(params["family"]),
            int(params.get("lcnn_width", LCNN_WIDTH_CHOICES[0])),
            int(params.get("resnet_width", RESNET_WIDTH_CHOICES[0])),
            float(params["dropout"]),
        )

    def dataset_fn(indices):
        return SpecDataset(specs, label_values, np.asarray(indices, dtype=np.int64))

    return model_fn, dataset_fn
