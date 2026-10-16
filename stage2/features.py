"""Фронтенды LFCC и log-mel. Спектрограммы кэшируются на диск в порядке манифеста."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torchaudio
from torch import nn

from stage2.bounds import (
    APPLY_ZNORM,
    DB_MULTIPLIER,
    FRONTEND_LFCC,
    FRONTEND_LOGMEL,
    LFCC_DCT_TYPE,
    LFCC_LOG_LF,
    LFCC_N_FILTER,
    LFCC_NORM,
    MEL_NORM,
    MEL_POWER,
    MEL_SCALE,
    POWER_FLOOR,
    SPEC_CACHE_VERSION,
    SPEC_CENTER,
    SPEC_COMPUTE_BATCH,
    SPEC_LOG_EVERY_BATCHES,
    TARGET_SR,
    ZNORM_EPS,
)


def znorm(feat: torch.Tensor) -> torch.Tensor:
    reduce = tuple(range(1, feat.dim()))
    mean = feat.mean(dim=reduce, keepdim=True)
    std = feat.std(dim=reduce, keepdim=True, unbiased=False)
    return (feat - mean) / (std + ZNORM_EPS)


class SpecFrontend(nn.Module):
    def __init__(self, frontend: str, n_bins: int, n_fft: int, hop: int) -> None:
        super().__init__()
        self.frontend = frontend
        speckwargs = {
            "n_fft": int(n_fft),
            "win_length": int(n_fft),
            "hop_length": int(hop),
            "center": SPEC_CENTER,
        }
        if frontend == FRONTEND_LOGMEL:
            self.body = torchaudio.transforms.MelSpectrogram(
                sample_rate=TARGET_SR,
                n_mels=int(n_bins),
                power=MEL_POWER,
                mel_scale=MEL_SCALE,
                norm=MEL_NORM,
                **speckwargs,
            )
        elif frontend == FRONTEND_LFCC:
            self.body = torchaudio.transforms.LFCC(
                sample_rate=TARGET_SR,
                n_filter=LFCC_N_FILTER,
                n_lfcc=int(n_bins),
                dct_type=LFCC_DCT_TYPE,
                norm=LFCC_NORM,
                log_lf=LFCC_LOG_LF,
                speckwargs=speckwargs,
            )
        else:
            raise ValueError(f"unknown frontend {frontend}")

    def forward(self, wave: torch.Tensor) -> torch.Tensor:
        if self.frontend == FRONTEND_LOGMEL:
            power = self.body(wave).clamp_min(POWER_FLOOR)
            ref = power.amax(dim=(-2, -1), keepdim=True).clamp_min(POWER_FLOOR)
            feat = DB_MULTIPLIER * torch.log10(power / ref)
        else:
            feat = self.body(wave)
        if APPLY_ZNORM:
            feat = znorm(feat)
        return feat


def cache_key(frontend: str, n_bins: int, n_fft: int, hop: int) -> str:
    return f"{SPEC_CACHE_VERSION}_{frontend}_bins{int(n_bins)}_fft{int(n_fft)}_hop{int(hop)}"


def ensure_spec_memmap(store, frontend: str, n_bins: int, n_fft: int, hop: int, cache_dir: Path) -> np.ndarray:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{cache_key(frontend, n_bins, n_fft, hop)}.npy"
    n = store.n
    if path.is_file():
        cached = np.load(path, mmap_mode="r")
        if cached.shape[0] == n and cached.ndim == 3:
            print(f"spec cache hit {path.name} {tuple(cached.shape)}", flush=True)
            return cached
    print(f"building spec cache {path.name} for {n} rows", flush=True)
    model = SpecFrontend(frontend, n_bins, n_fft, hop).eval()
    with torch.no_grad():
        sample = model(torch.from_numpy(store.wave(0)).unsqueeze(0))
    n_freq, n_frames = int(sample.shape[-2]), int(sample.shape[-1])
    out = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=(n, n_freq, n_frames))
    out[0] = sample.squeeze(0).numpy()
    step = SPEC_COMPUTE_BATCH
    log_every = step * SPEC_LOG_EVERY_BATCHES
    for start in range(1, n, step):
        stop = min(n, start + step)
        waves = np.stack([store.wave(i) for i in range(start, stop)], axis=0)
        with torch.no_grad():
            feat = model(torch.from_numpy(waves))
        out[start:stop] = feat.numpy()
        if start % log_every == 1 or stop == n:
            print(f"  specs {stop}/{n}", flush=True)
    out.flush()
    del out, model
    return np.load(path, mmap_mode="r")
