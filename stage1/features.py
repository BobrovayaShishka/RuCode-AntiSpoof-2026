"""Моменты MFCC и LFCC для этапа 1.

Сетка фронтенда извлекается один раз. Испытания Optuna режут коэффициенты
и моменты из этого кэша, поэтому оба классификатора делят одни матрицы.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.fft import dct
from scipy.signal import windows

from af_common import resample_audio

# Границы поиска. Извлечение считает полную сетку заранее.
N_FFT_CHOICES = (512, 1024)
HOP_LENGTH_CHOICES = (160, 320)
N_MELS_CHOICES = (40, 80)
N_MFCC_CHOICES = (13, 20, 40)
N_LFCC_CHOICES = (13, 20)
CEPSTRAL_SOURCES = ("mfcc", "lfcc", "both")
AGGREGATION_CHOICES = (
    "mean_std",
    "mean_std_skew",
    "mean_std_delta",
    "mean_std_skew_delta",
)

MOMENT_NAMES = ("mean", "std", "skew", "delta_mean", "delta_std")
MOMENT_GROUPS = {
    "mean_std": ("mean", "std"),
    "mean_std_skew": ("mean", "std", "skew"),
    "mean_std_delta": ("mean", "std", "delta_mean", "delta_std"),
    "mean_std_skew_delta": ("mean", "std", "skew", "delta_mean", "delta_std"),
}

# Фиксированные константы анализа, не подобранная рабочая точка.
PRE_EMPHASIS = 0.97
DELTA_WIDTH = 9
POWER_FLOOR = 1e-10
SKEW_STD_EPS = 1e-8
FILTER_NORM = "slaney"
MEL_SCALE = "htk"
MEL_HTK_SCALE = 2595.0
MEL_HTK_BREAK_HZ = 700.0
WINDOW_PERIODIC = True
CHANNEL_REDUCE = "mean"
FEATURE_CACHE_VERSION = 1
RESERVED_CORES = 1
EXTRACT_BATCH_FILES = 32
EXTRACT_LOG_EVERY_BATCHES = 10
FILTER_BANDWIDTH_EPS = 1e-6

MAX_N_MFCC = max(N_MFCC_CHOICES)
MAX_N_LFCC = max(N_LFCC_CHOICES)

_BANKS: dict[tuple[int, int, str], np.ndarray] | None = None
_SAMPLE_RATE: int | None = None


def _validate_bounds() -> None:
    if any(hop >= n_fft for hop in HOP_LENGTH_CHOICES for n_fft in N_FFT_CHOICES):
        raise RuntimeError("every hop length must be shorter than every n_fft")
    if DELTA_WIDTH < 3 or DELTA_WIDTH % 2 != 1:
        raise RuntimeError("DELTA_WIDTH must be an odd integer >= 3")
    if MAX_N_MFCC > min(N_MELS_CHOICES):
        raise RuntimeError("n_mfcc cannot exceed the smallest n_mels")
    if MAX_N_LFCC > min(N_MELS_CHOICES):
        raise RuntimeError("n_lfcc cannot exceed the smallest n_mels")
    missing = [name for name in AGGREGATION_CHOICES if name not in MOMENT_GROUPS]
    if missing:
        raise RuntimeError(f"aggregation choices missing moment groups: {missing}")
    for group in MOMENT_GROUPS.values():
        unknown = [name for name in group if name not in MOMENT_NAMES]
        if unknown:
            raise RuntimeError(f"unknown moments: {unknown}")


_validate_bounds()


def extract_worker_count() -> int:
    cpu_count = os.cpu_count() or 1
    return max(1, cpu_count - RESERVED_CORES)


def _hz_to_mel(hz: np.ndarray) -> np.ndarray:
    return MEL_HTK_SCALE * np.log10(1.0 + hz / MEL_HTK_BREAK_HZ)


def _mel_to_hz(mel: np.ndarray) -> np.ndarray:
    return MEL_HTK_BREAK_HZ * (np.power(10.0, mel / MEL_HTK_SCALE) - 1.0)


def _band_edges(sample_rate: int, n_bands: int, scale: str) -> np.ndarray:
    nyquist = sample_rate / 2.0
    if scale == "mel":
        return _mel_to_hz(np.linspace(_hz_to_mel(0.0), _hz_to_mel(nyquist), n_bands + 2))
    if scale == "hz":
        return np.linspace(0.0, nyquist, n_bands + 2)
    raise ValueError(f"unknown filter scale: {scale}")


def triangular_filterbank(sample_rate: int, n_fft: int, n_bands: int, scale: str) -> np.ndarray:
    """Треугольный банк фильтров. scale: mel для MFCC, hz для LFCC."""
    n_freq = n_fft // 2 + 1
    edges_hz = _band_edges(sample_rate, n_bands, scale)
    bins = np.floor((n_fft + 1) * edges_hz / sample_rate).astype(int)
    bins = np.clip(bins, 0, n_freq - 1)
    weights = np.zeros((n_bands, n_freq), dtype=np.float32)
    for band in range(n_bands):
        left = int(bins[band])
        center = int(bins[band + 1])
        right = int(bins[band + 2])
        if center <= left:
            center = min(left + 1, n_freq - 1)
        if right <= center:
            right = min(center + 1, n_freq)
        rising = np.arange(left, center, dtype=np.float32)
        if rising.size and center != left:
            weights[band, left:center] = (rising - left) / float(center - left)
        falling = np.arange(center, right, dtype=np.float32)
        if falling.size and right != center:
            weights[band, center:right] = (right - falling) / float(right - center)
    if FILTER_NORM == "slaney":
        bandwidth = np.maximum(edges_hz[2:] - edges_hz[:-2], FILTER_BANDWIDTH_EPS)
        weights *= (2.0 / bandwidth).astype(np.float32)[:, None]
    elif FILTER_NORM != "none":
        raise ValueError(f"unknown FILTER_NORM: {FILTER_NORM}")
    if not np.isfinite(weights).all() or weights.shape != (n_bands, n_freq):
        raise RuntimeError("filterbank is not finite")
    if np.any(weights.sum(axis=1) <= 0):
        raise RuntimeError(f"empty {scale} filter at n_fft={n_fft}, n_bands={n_bands}")
    return weights


def build_filterbanks(sample_rate: int) -> dict[tuple[int, int, str], np.ndarray]:
    banks: dict[tuple[int, int, str], np.ndarray] = {}
    for n_fft in N_FFT_CHOICES:
        for n_mels in N_MELS_CHOICES:
            banks[(n_fft, n_mels, "mel")] = triangular_filterbank(sample_rate, n_fft, n_mels, "mel")
            banks[(n_fft, n_mels, "hz")] = triangular_filterbank(sample_rate, n_fft, n_mels, "hz")
    return banks


def _limit_worker_threads() -> None:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    try:
        import threadpoolctl

        threadpoolctl.threadpool_limits(limits=1)
    except ImportError:
        return


def _init_worker(sample_rate: int) -> None:
    global _BANKS, _SAMPLE_RATE
    _limit_worker_threads()
    _SAMPLE_RATE = int(sample_rate)
    _BANKS = build_filterbanks(_SAMPLE_RATE)


def _periodic_hann(n_fft: int) -> np.ndarray:
    sym = not WINDOW_PERIODIC
    return windows.hann(n_fft, sym=sym).astype(np.float32)


def power_spectrogram(y: np.ndarray, n_fft: int, hop: int, window: np.ndarray) -> np.ndarray:
    if y.size < n_fft:
        y = np.pad(y, (0, n_fft - int(y.size)))
    pad = n_fft // 2
    if y.size <= pad:
        y = np.pad(y, (0, pad + 1 - int(y.size)), mode="edge")
    padded = np.pad(y, (pad, pad), mode="reflect")
    frames = np.lib.stride_tricks.sliding_window_view(padded, n_fft)[::hop]
    weighted = np.asarray(frames, dtype=np.float32) * window
    spectrum = np.fft.rfft(weighted, axis=1)
    power = np.square(spectrum.real) + np.square(spectrum.imag)
    return power.T.astype(np.float32, copy=False)


def _cepstra(power: np.ndarray, bank: np.ndarray, n_ceps: int) -> np.ndarray:
    filtered = bank @ power
    logged = np.log(filtered + np.float32(POWER_FLOOR))
    coeffs = dct(logged, type=2, axis=0, norm="ortho")
    return np.asarray(coeffs[:n_ceps], dtype=np.float32)


def delta_features(cepstra: np.ndarray, width: int = DELTA_WIDTH) -> np.ndarray:
    half = width // 2
    steps = np.arange(1, half + 1, dtype=np.float32)
    denom = np.float32(2.0 * np.sum(np.square(steps)))
    padded = np.pad(cepstra, ((0, 0), (half, half)), mode="edge")
    n_frames = cepstra.shape[1]
    accumulated = np.zeros_like(cepstra, dtype=np.float32)
    for step, weight in enumerate(steps, start=1):
        accumulated += weight * (
            padded[:, half + step : half + step + n_frames]
            - padded[:, half - step : half - step + n_frames]
        )
    return accumulated / denom


def moment_blocks(cepstra: np.ndarray) -> dict[str, np.ndarray]:
    mean = cepstra.mean(axis=1)
    std = cepstra.std(axis=1)
    safe_std = np.maximum(std, np.float32(SKEW_STD_EPS))
    centered = (cepstra - mean[:, None]) / safe_std[:, None]
    skew = np.mean(np.power(centered, 3), axis=1)
    skew = np.where(std < np.float32(SKEW_STD_EPS), np.float32(0.0), skew)
    delta = delta_features(cepstra)
    return {
        "mean": mean.astype(np.float32),
        "std": std.astype(np.float32),
        "skew": skew.astype(np.float32),
        "delta_mean": delta.mean(axis=1).astype(np.float32),
        "delta_std": delta.std(axis=1).astype(np.float32),
    }


def prepare_waveform(y: np.ndarray, file_rate: int, target_rate: int) -> np.ndarray:
    if y.ndim == 2:
        if CHANNEL_REDUCE != "mean":
            raise ValueError(f"unknown CHANNEL_REDUCE: {CHANNEL_REDUCE}")
        y = y.mean(axis=1)
    y = np.nan_to_num(np.asarray(y, dtype=np.float32))
    if file_rate != target_rate:
        y = resample_audio(y, int(file_rate), int(target_rate))
    if y.size == 0:
        y = np.zeros(max(N_FFT_CHOICES), dtype=np.float32)
    if PRE_EMPHASIS != 0.0 and y.size > 1:
        emphasized = np.empty_like(y)
        emphasized[0] = y[0]
        emphasized[1:] = y[1:] - np.float32(PRE_EMPHASIS) * y[:-1]
        y = emphasized
    return np.ascontiguousarray(y, dtype=np.float32)


def extract_waveform(y: np.ndarray, banks: dict[tuple[int, int, str], np.ndarray]) -> dict[str, np.ndarray]:
    windows = {n_fft: _periodic_hann(n_fft) for n_fft in N_FFT_CHOICES}
    flat: dict[str, np.ndarray] = {}
    for n_fft in N_FFT_CHOICES:
        for hop in HOP_LENGTH_CHOICES:
            power = power_spectrogram(y, n_fft, hop, windows[n_fft])
            for n_mels in N_MELS_CHOICES:
                mfcc = moment_blocks(_cepstra(power, banks[(n_fft, n_mels, "mel")], MAX_N_MFCC))
                lfcc = moment_blocks(_cepstra(power, banks[(n_fft, n_mels, "hz")], MAX_N_LFCC))
                for moment in MOMENT_NAMES:
                    flat[moment_key(n_fft, hop, n_mels, "mfcc", moment)] = mfcc[moment]
                    flat[moment_key(n_fft, hop, n_mels, "lfcc", moment)] = lfcc[moment]
    return flat


def moment_key(n_fft: int, hop: int, n_mels: int, prefix: str, moment: str) -> str:
    return f"nfft{int(n_fft)}_hop{int(hop)}_mels{int(n_mels)}__{prefix}_{moment}"


def _extract_batch(paths: list[str]) -> dict[str, np.ndarray]:
    if _BANKS is None or _SAMPLE_RATE is None:
        raise RuntimeError("feature worker was not initialized")
    columns: dict[str, list[np.ndarray]] = {}
    try:
        import threadpoolctl

        limiter = threadpoolctl.threadpool_limits(limits=1)
    except ImportError:
        limiter = None
    if limiter is None:
        return _stack_batch(paths, columns)
    with limiter:
        return _stack_batch(paths, columns)


def _stack_batch(paths: list[str], columns: dict[str, list[np.ndarray]]) -> dict[str, np.ndarray]:
    for path in paths:
        waveform, file_rate = sf.read(path, dtype="float32", always_2d=False)
        prepared = prepare_waveform(np.asarray(waveform), int(file_rate), _SAMPLE_RATE)
        if not np.isfinite(prepared).all():
            raise RuntimeError(f"non-finite waveform: {path}")
        extracted = extract_waveform(prepared, _BANKS)
        for key, values in extracted.items():
            columns.setdefault(key, []).append(values)
    stacked = {key: np.stack(rows, axis=0) for key, rows in columns.items()}
    for key, values in stacked.items():
        if not np.isfinite(values).all():
            raise RuntimeError(f"non-finite features in {key}")
    return stacked


def _sample_rate(path: str) -> int:
    return int(sf.info(path).samplerate)


def choose_sample_rate(rates: list[int]) -> int:
    if not rates:
        raise ValueError("no audio files")
    counts = Counter(int(rate) for rate in rates)
    # Самая частая частота. При ничьей берётся меньшая, выбор стабилен.
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]


def _chunks(paths: list[str], size: int) -> list[list[str]]:
    return [paths[start : start + size] for start in range(0, len(paths), size)]


def moment_schema() -> list[tuple[str, int]]:
    schema: list[tuple[str, int]] = []
    for n_fft in N_FFT_CHOICES:
        for hop in HOP_LENGTH_CHOICES:
            for n_mels in N_MELS_CHOICES:
                for prefix, width in (("mfcc", MAX_N_MFCC), ("lfcc", MAX_N_LFCC)):
                    for moment in MOMENT_NAMES:
                        schema.append((moment_key(n_fft, hop, n_mels, prefix, moment), width))
    return schema


def cache_signature(paths: list[str]) -> str:
    payload = {
        "version": FEATURE_CACHE_VERSION,
        "paths": paths,
        "n_fft": list(N_FFT_CHOICES),
        "hop": list(HOP_LENGTH_CHOICES),
        "n_mels": list(N_MELS_CHOICES),
        "max_n_mfcc": MAX_N_MFCC,
        "max_n_lfcc": MAX_N_LFCC,
        "moments": list(MOMENT_NAMES),
        "pre_emphasis": PRE_EMPHASIS,
        "delta_width": DELTA_WIDTH,
        "power_floor": POWER_FLOOR,
        "skew_std_eps": SKEW_STD_EPS,
        "filter_norm": FILTER_NORM,
        "mel_scale": MEL_SCALE,
        "mel_htk_scale": MEL_HTK_SCALE,
        "mel_htk_break_hz": MEL_HTK_BREAK_HZ,
        "window_periodic": WINDOW_PERIODIC,
        "channel_reduce": CHANNEL_REDUCE,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def extract_paths(paths: list[str], workers: int | None = None) -> dict:
    if not paths:
        raise ValueError("extract_paths received no files")
    worker_count = extract_worker_count() if workers is None else int(workers)
    with ThreadPoolExecutor(max_workers=max(1, worker_count)) as pool:
        rates = list(pool.map(_sample_rate, paths))
    sample_rate = choose_sample_rate(rates)
    schema = moment_schema()
    stores = {key: np.empty((len(paths), width), dtype=np.float32) for key, width in schema}
    batches = _chunks(paths, EXTRACT_BATCH_FILES)
    offset = 0

    def consume(batch_result: dict[str, np.ndarray], batch_index: int) -> None:
        nonlocal offset
        n_rows = next(iter(batch_result.values())).shape[0]
        for key, _width in schema:
            stores[key][offset : offset + n_rows] = batch_result[key]
        offset += n_rows
        done = offset
        if batch_index == 0 or (batch_index + 1) % EXTRACT_LOG_EVERY_BATCHES == 0 or done == len(paths):
            print(f"features {done}/{len(paths)}", flush=True)

    if worker_count <= 1:
        _init_worker(sample_rate)
        for batch_index, batch in enumerate(batches):
            consume(_extract_batch(batch), batch_index)
    else:
        with ProcessPoolExecutor(
            max_workers=worker_count,
            initializer=_init_worker,
            initargs=(sample_rate,),
        ) as pool:
            for batch_index, batch_result in enumerate(pool.map(_extract_batch, batches)):
                consume(batch_result, batch_index)

    if offset != len(paths):
        raise RuntimeError("feature extraction row count does not match the manifest")
    return {"sample_rate": sample_rate, "paths": list(paths), "arrays": stores}


def save_cache(cache: dict, path: Path, signature: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "signature": np.array(signature),
        "sample_rate": np.int64(cache["sample_rate"]),
        "paths": np.array(cache["paths"]),
    }
    payload.update(cache["arrays"])
    np.savez(path, **payload)


def load_cache(path: Path, signature: str, paths: list[str]) -> dict | None:
    if not path.is_file():
        return None
    with np.load(path, allow_pickle=False) as blob:
        cached_signature = str(blob["signature"].item())
        cached_paths = blob["paths"].astype(str).tolist()
        if cached_signature != signature or cached_paths != list(paths):
            return None
        arrays = {
            key: np.array(blob[key], dtype=np.float32, copy=True)
            for key, _width in moment_schema()
        }
        sample_rate = int(blob["sample_rate"])
    return {"sample_rate": sample_rate, "paths": cached_paths, "arrays": arrays}


def feature_names(frontend: dict) -> list[str]:
    names: list[str] = []
    for prefix, count in _active_sources(frontend):
        for moment in MOMENT_GROUPS[frontend["aggregation"]]:
            for index in range(count):
                names.append(f"{prefix}_c{index}_{moment}")
    return names


def _active_sources(frontend: dict) -> list[tuple[str, int]]:
    source = frontend["cepstral_source"]
    active: list[tuple[str, int]] = []
    if source in ("mfcc", "both"):
        active.append(("mfcc", int(frontend["n_mfcc"])))
    if source in ("lfcc", "both"):
        active.append(("lfcc", int(frontend["n_lfcc"])))
    if not active:
        raise ValueError("cepstral source produced no coefficients")
    return active


def assemble(cache: dict, frontend: dict) -> tuple[np.ndarray, list[str]]:
    blocks: list[np.ndarray] = []
    for prefix, count in _active_sources(frontend):
        for moment in MOMENT_GROUPS[frontend["aggregation"]]:
            key = moment_key(
                int(frontend["n_fft"]),
                int(frontend["hop_length"]),
                int(frontend["n_mels"]),
                prefix,
                moment,
            )
            blocks.append(cache["arrays"][key][:, :count])
    matrix = np.hstack(blocks).astype(np.float32, copy=False)
    if not np.isfinite(matrix).all():
        raise RuntimeError("assembled feature matrix is not finite")
    return matrix, feature_names(frontend)


def suggest_frontend(trial) -> dict:
    """Настройки фронтенда для одного испытания Optuna. Границы на уровне модуля."""
    return {
        "n_fft": int(trial.suggest_categorical("n_fft", list(N_FFT_CHOICES))),
        "hop_length": int(trial.suggest_categorical("hop_length", list(HOP_LENGTH_CHOICES))),
        "n_mels": int(trial.suggest_categorical("n_mels", list(N_MELS_CHOICES))),
        "cepstral_source": trial.suggest_categorical("cepstral_source", list(CEPSTRAL_SOURCES)),
        "n_mfcc": int(trial.suggest_categorical("n_mfcc", list(N_MFCC_CHOICES))),
        "n_lfcc": int(trial.suggest_categorical("n_lfcc", list(N_LFCC_CHOICES))),
        "aggregation": trial.suggest_categorical("aggregation", list(AGGREGATION_CHOICES)),
    }
