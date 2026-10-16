"""Только размеченный train. Shortcut-признаки и тестовое аудио не используются."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from af_common import (
    LABEL_COLUMN,
    PATH_COLUMN,
    PROJECT_ROOT,
    UTTERANCE_COLUMN,
    load_loso_table,
)


def resolve_manifest(manifest: str | Path) -> Path:
    path = Path(manifest)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def load_training_table(manifest: str | Path) -> pd.DataFrame:
    """Строки манифеста этапа 0 через общий загрузчик LOSO."""
    return load_loso_table(resolve_manifest(manifest))


def table_fingerprint(table: pd.DataFrame) -> str:
    """Отпечаток байтов аудио и общего ресемплера, из которых собран кэш.

    Размер и mtime входят, чтобы обрезанная замена того же id не переиспользовала признаки.
    """
    hasher = hashlib.sha256()
    utterances = table[UTTERANCE_COLUMN].astype(str).tolist()
    labels = table[LABEL_COLUMN].tolist()
    paths = table[PATH_COLUMN].astype(str).tolist()
    for utterance, label, raw_path in zip(utterances, labels, paths):
        path = Path(raw_path)
        stat = path.stat()
        line = (
            f"{utterance}\t{int(label)}\t{path.resolve()}\t"
            f"{stat.st_size}\t{stat.st_mtime_ns}\n"
        )
        hasher.update(line.encode("utf-8"))
    hasher.update(f"n={len(table)}\n".encode())
    hasher.update(b"resample=scipy.signal.resample_poly\n")
    return hasher.hexdigest()
