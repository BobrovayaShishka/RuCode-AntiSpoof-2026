"""CUDA и masked mean-pool для нейронных этапов."""

from __future__ import annotations

import gc

import torch


def is_oom(exc: BaseException) -> bool:
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    return "out of memory" in str(exc).lower()


def release_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def masked_mean_pool(hidden: torch.Tensor, frame_mask: torch.Tensor) -> torch.Tensor:
    """Среднее по времени. Паддинг не входит. Пустая маска нормируется единицей."""
    values = hidden.float()
    mask = frame_mask.unsqueeze(-1).to(dtype=values.dtype)
    denom = mask.sum(dim=1).clamp(min=1.0)
    return (values * mask).sum(dim=1) / denom
