"""Forward замороженного SSL. Mean-pool каждого скрытого состояния. Нужна gpu_lock."""

from __future__ import annotations

import numpy as np
import torch
from transformers import AutoFeatureExtractor, AutoModel

from torchutil import is_oom, masked_mean_pool, release_cuda

from stage4.constants import (
    ALLOW_TF32,
    AUDIO_MAX_SECONDS_CANDIDATES,
    GPU_DTYPE_NAME,
    HF_HUB_CACHE,
)


def _dtype() -> torch.dtype:
    if GPU_DTYPE_NAME == "float16":
        return torch.float16
    raise ValueError(GPU_DTYPE_NAME)


def load_frozen(checkpoint: str, device: torch.device):
    if ALLOW_TF32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    extractor = AutoFeatureExtractor.from_pretrained(
        checkpoint, cache_dir=str(HF_HUB_CACHE), local_files_only=True
    )
    model = AutoModel.from_pretrained(
        checkpoint,
        cache_dir=str(HF_HUB_CACHE),
        dtype=_dtype(),
        local_files_only=True,
    )
    model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if not hasattr(model, "_get_feature_vector_attention_mask"):
        raise TypeError(f"{checkpoint} has no frame attention mask")
    return extractor, model


def _frame_mask(model, attention_mask: torch.Tensor, frames: int) -> torch.Tensor:
    mask = model._get_feature_vector_attention_mask(frames, attention_mask)
    if mask.shape[1] != frames:
        raise RuntimeError(f"frame mask {tuple(mask.shape)} does not match hidden time {frames}")
    return mask


def _mean_pool(hidden: torch.Tensor, frame_mask: torch.Tensor) -> np.ndarray:
    pooled = masked_mean_pool(hidden, frame_mask)
    array = pooled.detach().cpu().numpy().astype(np.float32, copy=False)
    if not np.isfinite(array).all():
        raise RuntimeError("non-finite pooled features")
    return array


@torch.inference_mode()
def encode_waves(extractor, model, waves: list[np.ndarray], device: torch.device) -> list[np.ndarray]:
    sample_rate = int(extractor.sampling_rate)
    inputs = extractor(
        waves,
        sampling_rate=sample_rate,
        return_tensors="pt",
        padding=True,
        return_attention_mask=True,
    )
    dtype = next(model.parameters()).dtype
    input_values = inputs["input_values"].to(device=device, dtype=dtype)
    attention_mask = inputs["attention_mask"].to(device=device)
    outputs = model(
        input_values,
        attention_mask=attention_mask,
        output_hidden_states=True,
        return_dict=True,
    )
    hidden_states = outputs.hidden_states
    if not hidden_states:
        raise RuntimeError("model returned no hidden states")
    frame_mask = _frame_mask(model, attention_mask, hidden_states[0].shape[1])
    pooled = [_mean_pool(hidden, frame_mask) for hidden in hidden_states]
    del outputs, hidden_states, input_values, attention_mask
    return pooled


def fit_max_seconds(extractor, model, device: torch.device) -> float:
    sample_rate = int(extractor.sampling_rate)
    last_error: Exception | None = None
    for cap in AUDIO_MAX_SECONDS_CANDIDATES:
        wave = np.zeros(int(round(cap * sample_rate)), dtype=np.float32)
        try:
            encode_waves(extractor, model, [wave], device)
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            return float(cap)
        except torch.cuda.OutOfMemoryError as exc:
            last_error = exc
            release_cuda()
    raise RuntimeError(
        f"encoder does not fit even at {AUDIO_MAX_SECONDS_CANDIDATES[-1]} s"
    ) from last_error


