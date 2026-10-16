"""Общие JSON и heartbeat для всех этапов."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np


def to_jsonable(value):
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, payload: dict, *, ensure_ascii: bool = True) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(to_jsonable(payload), indent=2, ensure_ascii=ensure_ascii)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text + "\n", encoding="utf-8")
    temporary.replace(path)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def beat(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(message, encoding="utf-8")
    print(message, flush=True)


def flag_is_true(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return value is True or value == 1


def wait_until_ready(verdict_path: Path, poll_s: float) -> dict:
    """Ждать, пока этап 0 выставит ready_for_training. Флаг утечки обучение не останавливает.

    В готовом вердикте должен быть data_manifest. Повреждённый файл считается неготовым.
    """
    verdict_path = Path(verdict_path)
    while True:
        verdict = None
        if verdict_path.is_file():
            try:
                loaded = json.loads(verdict_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                loaded = None
            if isinstance(loaded, dict):
                verdict = loaded
        if verdict is not None and flag_is_true(verdict.get("ready_for_training")):
            if not verdict.get("data_manifest"):
                raise RuntimeError("verdict is ready_for_training but has no data_manifest")
            print(
                "ready_for_training "
                f"manifest={verdict.get('data_manifest')} "
                f"leak={verdict.get('leak')} "
                f"acoustic_shortcut={verdict.get('acoustic_shortcut')}",
                flush=True,
            )
            return verdict
        if verdict is None:
            print(f"waiting for {verdict_path}", flush=True)
        else:
            print(
                "waiting for ready_for_training "
                f"leak={verdict.get('leak')} "
                f"manifest={verdict.get('data_manifest')} "
                f"ready_for_training={verdict.get('ready_for_training')!r}",
                flush=True,
            )
        time.sleep(poll_s)
