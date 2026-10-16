"""Кэши mean-pooled слоёв. Один memmap на слой, порядок строк как в таблице LOSO."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from jsonutil import read_json, write_json
from stage4.constants import ACTIVE_FILE, CACHE_ROOT, LAYER_FILE, META_FILE


def slug_for(checkpoint: str) -> str:
    return checkpoint.split("/")[-1]


def pointer_path(checkpoint: str) -> Path:
    return CACHE_ROOT / slug_for(checkpoint) / ACTIVE_FILE


def read_pointer(checkpoint: str) -> dict | None:
    path = pointer_path(checkpoint)
    if not path.is_file():
        return None
    return read_json(path)


def write_pointer(checkpoint: str, fingerprint: str, directory: Path, complete: bool) -> None:
    write_json(
        pointer_path(checkpoint),
        {
            "checkpoint": checkpoint,
            "fingerprint": fingerprint,
            "directory": directory.name,
            "complete": complete,
        },
    )


def cache_dir_from_pointer(checkpoint: str, fingerprint: str) -> tuple[Path, dict] | None:
    pointer = read_pointer(checkpoint)
    if pointer is None or pointer.get("fingerprint") != fingerprint:
        return None
    directory = CACHE_ROOT / slug_for(checkpoint) / str(pointer["directory"])
    meta_path = directory / META_FILE
    if not meta_path.is_file():
        return None
    meta = read_json(meta_path)
    if meta.get("fingerprint") != fingerprint:
        return None
    return directory, meta


def allocate_cache_dir(checkpoint: str, cap_seconds: float, fingerprint: str) -> Path:
    root = CACHE_ROOT / slug_for(checkpoint)
    root.mkdir(parents=True, exist_ok=True)
    cap_name = f"cap{cap_seconds:g}"
    candidate = root / cap_name
    meta_path = candidate / META_FILE
    if meta_path.is_file():
        existing = read_json(meta_path)
        if existing.get("fingerprint") != fingerprint:
            candidate = root / f"{cap_name}_{fingerprint[:8]}"
    candidate.mkdir(parents=True, exist_ok=True)
    return candidate


def layer_path(directory: Path, index: int) -> Path:
    return directory / LAYER_FILE.format(index=index)


class LayerStore:
    def __init__(self, directory: Path):
        self.directory = directory
        self.meta_path = directory / META_FILE
        self.meta: dict = {}
        self.maps: list[np.memmap] = []

    def create(self, meta: dict, hidden: int) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        n_rows = int(meta["n_rows"])
        n_layers = int(meta["n_layers"])
        self.maps = []
        for index in range(n_layers):
            path = layer_path(self.directory, index)
            array = np.memmap(path, dtype=np.float32, mode="w+", shape=(n_rows, hidden))
            array.flush()
            self.maps.append(array)
        meta = dict(meta)
        meta["hidden"] = int(hidden)
        meta["rows_done"] = 0
        self.meta = meta
        self.save_meta()

    def open_existing(self, meta: dict) -> None:
        self.meta = dict(meta)
        n_rows = int(meta["n_rows"])
        hidden = int(meta["hidden"])
        n_layers = int(meta["n_layers"])
        self.maps = []
        for index in range(n_layers):
            path = layer_path(self.directory, index)
            self.maps.append(
                np.memmap(path, dtype=np.float32, mode="r+", shape=(n_rows, hidden))
            )

    def write_rows(self, start: int, pooled: list[np.ndarray], n_cropped: int) -> None:
        count = int(pooled[0].shape[0])
        if len(pooled) != len(self.maps):
            raise RuntimeError(
                f"expected {len(self.maps)} layers, forward returned {len(pooled)}"
            )
        for index, values in enumerate(pooled):
            if values.shape != (count, int(self.meta["hidden"])):
                raise RuntimeError(f"layer {index} shape {values.shape}")
            self.maps[index][start : start + count] = values
            self.maps[index].flush()
        self.meta["rows_done"] = start + count
        self.meta["n_cropped"] = int(n_cropped)
        self.save_meta()

    def save_meta(self) -> None:
        write_json(self.meta_path, self.meta)

    def close(self) -> None:
        for array in self.maps:
            array.flush()
        self.maps = []


def read_layer(directory: Path, index: int, n_rows: int, hidden: int) -> np.ndarray:
    path = layer_path(directory, index)
    mapped = np.memmap(path, dtype=np.float32, mode="r", shape=(n_rows, hidden))
    copied = np.array(mapped, dtype=np.float32, copy=True)
    del mapped
    return copied


def cache_is_complete(meta: dict) -> bool:
    return int(meta.get("rows_done", 0)) >= int(meta.get("n_rows", 1)) and int(meta.get("n_rows", 0)) > 0
