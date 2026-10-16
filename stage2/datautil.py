"""Волны манифеста в порядке LOSO. Тестовое аудио и ablation-кэш не используются."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from torch.utils.data import Dataset

import af_common
from stage2.bounds import TARGET_LENGTH, TARGET_SR, WAVE_CACHE_SCALE

DECODE_WORKERS = 4
WAVE_MEMMAP_NAME = "waves_int16_resample_poly.npy"


def _fit_length(audio: np.ndarray) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if audio.shape[0] < TARGET_LENGTH:
        audio = np.pad(audio, (0, TARGET_LENGTH - audio.shape[0]))
    else:
        audio = audio[:TARGET_LENGTH]
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 1.0:
        audio = audio / peak
    return audio


def _on_colab_drive(path: Path) -> bool:
    return "/content/drive/" in Path(path).as_posix()


def _read_soundfile(path: Path):
    """Декодировать аудио. libsndfile даёт System error, если открывает путь Drive сам."""
    import io
    import time

    path = Path(path)
    attempts = 5 if _on_colab_drive(path) else 1
    last_error = None
    for attempt in range(attempts):
        try:
            if _on_colab_drive(path):
                payload = path.read_bytes()
                return sf.read(io.BytesIO(payload), always_2d=False)
            return sf.read(path, always_2d=False)
        except (OSError, sf.LibsndfileError) as exc:
            last_error = exc
            if attempt + 1 == attempts:
                break
            time.sleep(float(attempt + 1))
    raise last_error


def load_wave_from_path(path: Path) -> np.ndarray:
    path = Path(path)
    if path.suffix.lower() == ".npy":
        audio = np.load(path)
        if getattr(audio, "ndim", 1) > 1:
            audio = np.mean(audio, axis=-1 if audio.shape[-1] < audio.shape[0] else 0)
        return _fit_length(audio)
    audio, sr = _read_soundfile(path)
    if getattr(audio, "ndim", 1) > 1:
        audio = np.mean(audio, axis=1)
    audio = np.asarray(audio, dtype=np.float32)
    if sr != TARGET_SR:
        audio = af_common.resample_audio(audio, sr, TARGET_SR)
    return _fit_length(audio)


def _as_int16(audio: np.ndarray) -> np.ndarray:
    scaled = np.round(audio * WAVE_CACHE_SCALE)
    return np.clip(scaled, -WAVE_CACHE_SCALE, WAVE_CACHE_SCALE).astype(np.int16)


class WaveStore:
    """Волны в порядке манифеста. prepare пишет int16 memmap рядом с выводами этапа."""

    def __init__(self, table: pd.DataFrame) -> None:
        self.paths = extract_paths(table)
        self.n = len(self.paths)
        self.local = None
        self.local_scale = WAVE_CACHE_SCALE
        self.source = "manifest_audio"

    def prepare(self, dest_dir: Path) -> None:
        if self.local is not None:
            return
        dest_dir.mkdir(parents=True, exist_ok=True)
        path = dest_dir / WAVE_MEMMAP_NAME
        if path.is_file():
            try:
                cached = np.load(path, mmap_mode="r")
            except (OSError, ValueError):
                cached = None
            if cached is not None and cached.shape == (self.n, TARGET_LENGTH):
                self.local = cached
                self.source = str(path)
                return
        print(f"decoding {self.n} manifest waveforms into {path.name}", flush=True)
        out = np.lib.format.open_memmap(path, mode="w+", dtype=np.int16, shape=(self.n, TARGET_LENGTH))
        on_drive = bool(self.paths) and _on_colab_drive(self.paths[0])
        workers = 1 if on_drive else max(1, min(DECODE_WORKERS, self.n))

        def _one(index: int) -> tuple[int, np.ndarray]:
            return index, _as_int16(load_wave_from_path(self.paths[index]))

        done = 0
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for index, audio in pool.map(_one, range(self.n)):
                out[index] = audio
                done += 1
                if done == 1 or done % 1000 == 0 or done == self.n:
                    print(f"  waves {done}/{self.n}", flush=True)
        out.flush()
        del out
        self.local = np.load(path, mmap_mode="r")
        self.source = str(path)

    def wave(self, index: int) -> np.ndarray:
        if self.local is not None:
            return np.asarray(self.local[index], dtype=np.float32) / self.local_scale
        return load_wave_from_path(self.paths[index])


class SpecDataset(Dataset):
    def __init__(self, specs: np.ndarray, labels: np.ndarray, indices: np.ndarray) -> None:
        self.specs = specs
        self.labels = labels
        self.indices = np.asarray(indices)

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def __getitem__(self, i: int):
        j = int(self.indices[i])
        spec = torch.tensor(np.asarray(self.specs[j]), dtype=torch.float32).unsqueeze(0)
        label = torch.tensor(float(self.labels[j]), dtype=torch.float32)
        return spec, label


class WaveDataset(Dataset):
    def __init__(self, store: WaveStore, labels: np.ndarray, indices: np.ndarray) -> None:
        self.store = store
        self.labels = labels
        self.indices = np.asarray(indices)

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def __getitem__(self, i: int):
        j = int(self.indices[i])
        wave = torch.tensor(self.store.wave(j), dtype=torch.float32)
        label = torch.tensor(float(self.labels[j]), dtype=torch.float32)
        return wave, label


def coerce_table(table) -> pd.DataFrame:
    if isinstance(table, pd.DataFrame):
        return table.reset_index(drop=True)
    return pd.DataFrame(table).reset_index(drop=True)


def resolve_path(value, root: Path | None = None) -> Path | None:
    if value is None or value == "":
        return None
    path = Path(str(value))
    if not path.is_absolute():
        path = (af_common.PROJECT_ROOT if root is None else root) / path
    return path


def make_store(table: pd.DataFrame) -> WaveStore:
    return WaveStore(table)


def open_training_table(verdict: dict) -> tuple[pd.DataFrame, str]:
    """Размеченные строки ASVspoof для LOSO.

    data_manifest: протокол train или CSV этапа 0.
    af_common.load_loso_table читает оба и заполняет path и label.
    """
    manifest = resolve_path(verdict.get("data_manifest"))
    if manifest is None or not manifest.is_file():
        raise FileNotFoundError("verdict.data_manifest is missing")
    return coerce_table(af_common.load_loso_table(manifest)), str(manifest)


def extract_labels(table: pd.DataFrame) -> np.ndarray:
    if af_common.LABEL_COLUMN not in table.columns:
        raise KeyError(f"no {af_common.LABEL_COLUMN} column in {list(table.columns)}")
    labels = table[af_common.LABEL_COLUMN].to_numpy(dtype=np.float32)
    uniq = set(np.unique(labels).tolist())
    if not uniq <= {0.0, 1.0}:
        raise ValueError(f"labels must be binary, got {uniq}")
    return labels


def extract_paths(table: pd.DataFrame) -> list[Path]:
    if af_common.PATH_COLUMN not in table.columns:
        raise KeyError(f"no {af_common.PATH_COLUMN} column in {list(table.columns)}")
    return [Path(str(value)) for value in table[af_common.PATH_COLUMN].tolist()]


def validate_folds(folds, n_rows: int) -> None:
    seen = set()
    if not folds:
        raise RuntimeError("LOSO produced no folds")
    for fold in folds:
        fold_id = str(fold.fold_id)
        if fold_id in seen:
            raise RuntimeError(f"duplicate fold id {fold_id}")
        seen.add(fold_id)
        train_index = np.asarray(fold.train_index)
        val_index = np.asarray(fold.val_index)
        if train_index.size == 0 or val_index.size == 0:
            raise RuntimeError(f"empty fold {fold_id}")
        if (
            int(train_index.min()) < 0
            or int(val_index.min()) < 0
            or int(train_index.max()) >= n_rows
            or int(val_index.max()) >= n_rows
        ):
            raise RuntimeError(f"fold {fold_id} index outside 0..{n_rows - 1}")
