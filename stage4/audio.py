"""Загрузка волны. Только отсчёты, без длительности, тишины, громкости и частоты."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from stage4.constants import MIN_AUDIO_SECONDS


def load_wave(
    path: Path,
    sample_rate: int,
    max_samples: int,
    min_samples: int | None = None,
) -> tuple[np.ndarray, bool]:
    if min_samples is None:
        min_samples = max(1, int(round(MIN_AUDIO_SECONDS * sample_rate)))
    audio, file_rate = sf.read(str(path), always_2d=False, dtype="float32")
    if np.asarray(audio).ndim > 1:
        audio = np.mean(audio, axis=1, dtype=np.float32)
    audio = np.asarray(audio, dtype=np.float32)
    cropped = False
    if int(file_rate) != int(sample_rate):
        from af_common import resample_audio

        audio = resample_audio(audio, int(file_rate), int(sample_rate))
    if audio.size < min_samples:
        audio = np.pad(audio, (0, min_samples - int(audio.size)))
    if audio.size > max_samples:
        audio = audio[:max_samples]
        cropped = True
    if audio.size == 0:
        audio = np.zeros(min_samples, dtype=np.float32)
    return np.ascontiguousarray(audio, dtype=np.float32), cropped


def wave_sample_count(path: Path, sample_rate: int, max_samples: int, min_samples: int) -> int:
    info = sf.info(str(path))
    if info.samplerate:
        count = int(round(float(info.duration) * sample_rate))
    else:
        count = int(info.frames)
    count = max(count, min_samples)
    return min(count, max_samples)
