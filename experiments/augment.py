"""Аугментация волны на обучении, до LFCC.

RawBoost: FIR-окраска реплики и импульсы, зависящие от сигнала.
Симуляция кодека: узкополосный телефонный тракт, без ffmpeg.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from experiments.config import (
    ALIGN_CORNERS,
    BATCH_DIM,
    CHANNEL_DIM,
    CODEC_BANDLIMIT,
    CODEC_MULAW,
    CODEC_TELEPHONE,
    CONV_IN_CHANNELS,
    CONV_SAME_PAD_DIVISOR,
    INTERPOLATE_MODE,
    LAST_DIM,
    MULAW_MAX_CODE,
    MULAW_MU,
    NORM_EPS,
    ONE,
    PAD_LEFT,
    PEAK_LIMIT,
    QUANT_MIN,
    RAWBOOST_CONVOLUTIVE,
    RAWBOOST_IMPULSIVE,
    RAWBOOST_SERIES,
    REMAINDER_NONE,
    START_INDEX,
    TARGET_LENGTH,
    TARGET_SR,
    TELEPHONE_BAND_HIGH_HZ,
    TELEPHONE_BAND_LOW_HZ,
    TELEPHONY_SAMPLE_RATE,
    TWO,
    WAVE_RANK,
)


def telephony_factor() -> int:
    remainder = int(TARGET_SR) % int(TELEPHONY_SAMPLE_RATE)
    if remainder != REMAINDER_NONE:
        raise RuntimeError("telephony sample rate must divide the training sample rate")
    return int(TARGET_SR) // int(TELEPHONY_SAMPLE_RATE)


def fit_wave_length(wave: torch.Tensor) -> torch.Tensor:
    """Подогнать длину обучения, пик не выше 1."""
    length = int(wave.shape[LAST_DIM])
    if length < int(TARGET_LENGTH):
        wave = F.pad(wave, (PAD_LEFT, int(TARGET_LENGTH) - length))
    elif length > int(TARGET_LENGTH):
        wave = wave.narrow(LAST_DIM, START_INDEX, int(TARGET_LENGTH))
    return limit_peak(wave)


def limit_peak(wave: torch.Tensor) -> torch.Tensor:
    peak = wave.abs().amax(dim=LAST_DIM, keepdim=True)
    scale = torch.where(peak > PEAK_LIMIT, peak, torch.ones_like(peak))
    return wave / scale


def apply_rawboost(wave: torch.Tensor, params: dict) -> torch.Tensor:
    if wave.dim() != WAVE_RANK:
        raise ValueError("RawBoost expects (batch, time)")
    variant = str(params["variant"])
    if variant == RAWBOOST_CONVOLUTIVE:
        augmented = _convolutive(wave, params)
    elif variant == RAWBOOST_IMPULSIVE:
        augmented = _impulsive(wave, params)
    elif variant == RAWBOOST_SERIES:
        augmented = _impulsive(_convolutive(wave, params), params)
    else:
        raise ValueError(f"unknown RawBoost variant {variant}")
    return _gate(wave, augmented, float(params["apply_probability"]))


def apply_codec(wave: torch.Tensor, params: dict) -> torch.Tensor:
    if wave.dim() != WAVE_RANK:
        raise ValueError("codec simulation expects (batch, time)")
    choice = str(params["choice"])
    if choice == CODEC_BANDLIMIT:
        augmented = bandlimit(wave, int(TARGET_SR), TELEPHONE_BAND_LOW_HZ, TELEPHONE_BAND_HIGH_HZ)
    elif choice == CODEC_MULAW:
        augmented = mulaw_roundtrip(wave)
    elif choice == CODEC_TELEPHONE:
        augmented = telephone_path(wave)
    else:
        raise ValueError(f"unknown codec choice {choice}")
    return _gate(wave, augmented, float(params["apply_probability"]))


def telephone_path(wave: torch.Tensor) -> torch.Tensor:
    """Ограничить полосу, сузить частоту, mu-law, вернуть частоту обучения."""
    limited = bandlimit(wave, int(TARGET_SR), TELEPHONE_BAND_LOW_HZ, TELEPHONE_BAND_HIGH_HZ)
    factor = telephony_factor()
    down = limited[..., ::factor]
    coded = mulaw_roundtrip(down)
    restored = F.interpolate(
        coded.unsqueeze(CHANNEL_DIM),
        scale_factor=factor,
        mode=INTERPOLATE_MODE,
        align_corners=ALIGN_CORNERS,
    ).squeeze(CHANNEL_DIM)
    return restored


def bandlimit(wave: torch.Tensor, sample_rate: int, low_hz: float, high_hz: float) -> torch.Tensor:
    spectrum = torch.fft.rfft(wave, dim=LAST_DIM)
    freqs = torch.fft.rfftfreq(int(wave.shape[LAST_DIM]), d=ONE / float(sample_rate), device=wave.device)
    keep = ((freqs >= float(low_hz)) & (freqs <= float(high_hz))).to(spectrum.dtype)
    return torch.fft.irfft(spectrum * keep, n=int(wave.shape[LAST_DIM]), dim=LAST_DIM)


def mulaw_roundtrip(wave: torch.Tensor) -> torch.Tensor:
    """8-битный mu-law. MULAW_MU: константа G.711, не подобранный коэффициент."""
    mu = wave.new_tensor(float(MULAW_MU))
    compressed = torch.sign(wave.clamp(-PEAK_LIMIT, PEAK_LIMIT)) * torch.log1p(mu * wave.clamp(-PEAK_LIMIT, PEAK_LIMIT).abs()) / torch.log1p(mu)
    codes = torch.round((compressed + PEAK_LIMIT) / TWO * float(MULAW_MAX_CODE)).clamp(QUANT_MIN, float(MULAW_MAX_CODE))
    decoded = codes / float(MULAW_MAX_CODE) * TWO - PEAK_LIMIT
    return torch.sign(decoded) * (torch.pow(ONE + mu, decoded.abs()) - ONE) / mu


class WaveAugmentor:
    """Применить выбранные в hparams блоки RawBoost и кодека. Только train."""

    def __init__(self, hparams: dict) -> None:
        self.rawboost = hparams.get("rawboost")
        self.codec = hparams.get("codec")

    def __call__(self, wave: torch.Tensor) -> torch.Tensor:
        if self.rawboost:
            wave = apply_rawboost(wave, self.rawboost)
        if self.codec:
            wave = apply_codec(wave, self.codec)
        return fit_wave_length(wave)


def _convolutive(wave: torch.Tensor, params: dict) -> torch.Tensor:
    batch = wave.shape[BATCH_DIM]
    n_taps = int(params["n_taps"])
    strength = float(params["strength"])
    taps = torch.randn(batch, CONV_IN_CHANNELS, n_taps, device=wave.device, dtype=wave.dtype)
    taps = taps / taps.abs().sum(dim=LAST_DIM, keepdim=True).clamp_min(NORM_EPS)
    pad = n_taps // CONV_SAME_PAD_DIVISOR
    colored = F.conv1d(wave.unsqueeze(BATCH_DIM), taps, padding=pad, groups=int(batch)).squeeze(BATCH_DIM)
    shaped = (ONE - strength) * colored + strength * torch.tanh(colored)
    return (ONE - strength) * wave + strength * shaped


def _impulsive(wave: torch.Tensor, params: dict) -> torch.Tensor:
    strength = float(params["strength"])
    mask = (torch.rand_like(wave) < float(params["impulse_rate"])).to(wave.dtype)
    noise = torch.randn_like(wave) * wave.abs()
    return wave + strength * mask * noise


def _gate(clean: torch.Tensor, augmented: torch.Tensor, probability: float) -> torch.Tensor:
    clean = fit_wave_length(clean)
    augmented = fit_wave_length(augmented)
    draw = torch.rand(clean.shape[BATCH_DIM], device=clean.device, dtype=clean.dtype)
    gate = (draw < float(probability)).to(clean.dtype).unsqueeze(LAST_DIM)
    return gate * augmented + (ONE - gate) * clean
