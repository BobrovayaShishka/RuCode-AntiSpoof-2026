"""Найти корень кода и каталог аудио для ноутбуков этапов.

Ячейка ищет только этот файл. Обход Drive, zip и вложенность Data/Data живут здесь.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _visible(path: Path) -> bool:
    """Drive скрывает файлы от is_file(), пока не обойти каждый родительский каталог."""
    if not path.is_absolute():
        return path.is_file()
    cursor = Path(path.anchor)
    for part in path.parts[1:-1]:
        cursor = cursor / part
        if not cursor.is_dir():
            return False
        try:
            os.listdir(cursor)
        except OSError:
            return False
    return path.is_file()


def _dataset_root(folder: Path) -> Path | None:
    """Принять Data/train и Data/Data/train: в архиве уже есть каталог Data."""
    for candidate in (folder, folder / "Data"):
        protocol = candidate / "train" / "targets.txt"
        audio = candidate / "train" / "flac"
        if not _visible(protocol) or not audio.is_dir():
            continue
        try:
            has_audio = next(audio.glob("*.flac"), None) is not None
        except OSError:
            has_audio = False
        if has_audio:
            return candidate
    return None


def resolve_data_dir(repo_root: Path, requested: Path) -> Path:
    """Оставить путь Drive, если он смонтирован. Не подменять отсутствующим локальным Data/."""
    local_data = repo_root / "Data"
    found = _dataset_root(requested)
    if found is not None:
        return found
    found = _dataset_root(local_data)
    if found is not None:
        return found
    drive_protocol = requested / "train" / "targets.txt"
    local_protocol = local_data / "train" / "targets.txt"
    if _visible(drive_protocol):
        return requested
    if _visible(local_protocol):
        return local_data
    try:
        import google.colab  # noqa: F401

        in_colab = True
    except ImportError:
        in_colab = False
    if in_colab or Path("/content/drive").is_dir():
        raise FileNotFoundError(
            f"Не найден {drive_protocol}. Смонтируйте Drive и положите датасет в {requested}. "
            "Каталог кода на этот путь не подменяется."
        )
    return local_data


def boot(needed: tuple[str, ...], data_dir: Path) -> tuple[Path, Path]:
    """Проверить, что в корне есть needed, затем найти каталог аудио."""
    root = Path(__file__).resolve().parent
    missing = [item for item in needed if not (root / item).exists()]
    if missing:
        raise FileNotFoundError(f"В {root} не хватает: {', '.join(missing)}")
    resolved = resolve_data_dir(root, Path(data_dir))
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    print("code", root)
    print("data", resolved)
    return root, resolved
