"""CNN по сырой волне, обучение с нуля."""

from __future__ import annotations

import torch
from torch import nn

from stage3.bounds import BLOCK_KERNEL, BLOCK_STRIDE


class RawCNNFrontend(nn.Module):
    """Conv1d: волна (B, T) в эмбеддинг (B, width)."""

    def __init__(self, width: int, kernel: int, stride: int, n_blocks: int) -> None:
        super().__init__()
        layers = []
        in_ch = 1
        for index in range(int(n_blocks)):
            block_kernel = int(kernel) if index == 0 else BLOCK_KERNEL
            block_stride = int(stride) if index == 0 else BLOCK_STRIDE
            layers.extend(
                [
                    nn.Conv1d(
                        in_ch,
                        int(width),
                        block_kernel,
                        stride=block_stride,
                        padding=block_kernel // 2,
                        bias=False,
                    ),
                    nn.BatchNorm1d(int(width)),
                    nn.ReLU(inplace=True),
                ]
            )
            in_ch = int(width)
        self.blocks = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, wave: torch.Tensor) -> torch.Tensor:
        if wave.dim() == 2:
            wave = wave.unsqueeze(1)
        return self.pool(self.blocks(wave)).flatten(1)


class RawWaveCNN(nn.Module):
    def __init__(self, width: int, kernel: int, stride: int, n_blocks: int, dropout: float) -> None:
        super().__init__()
        self.frontend = RawCNNFrontend(width, kernel, stride, n_blocks)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(int(width), 1)

    def forward(self, wave: torch.Tensor) -> torch.Tensor:
        return self.fc(self.drop(self.frontend(wave))).view(-1)


def frontend_report() -> dict:
    return {
        "class": "RawCNNFrontend",
        "source": "stage3.frontend",
        "external": False,
        "note": "Stage 3 defines RawCNNFrontend and trains the CNN from scratch.",
    }


def build_raw_cnn(params: dict) -> nn.Module:
    return RawWaveCNN(
        width=int(params["width"]),
        kernel=int(params["kernel"]),
        stride=int(params["stride"]),
        n_blocks=int(params["n_blocks"]),
        dropout=float(params["dropout"]),
    )
