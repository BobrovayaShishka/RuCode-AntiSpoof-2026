"""Общий протокол проверки антиспуфинга.

Одна реализация LOSO по синтезатору и одна реализация метрик для всех этапов.
Импортируйте этот модуль, не дублируйте фолды и AUC.

Разметка: Data/train/targets.txt, пробелы, без заголовка.
Колонки: speaker_id, utterance_id, неиспользуемый дефис,
synthesizer (A01, A02, ...; для bona fide стоит -), label (bonafide / spoof).

Публичный тест не загружается. Фолд задаёт удержанный synthesizer.
Строки bona fide режутся на смежные равные куски в порядке протокола,
чтобы в валидации были оба класса.

Гиперпараметры: именованные границы Optuna. Пороги в коде этапов не вводить.
"""

from __future__ import annotations

import json
import os
import time
from collections import namedtuple
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd


SEEDS = (0, 1, 2)
N_OPTUNA_TRIALS = 20
MIN_MEAN_AUC_GAIN = 0.005
BOOTSTRAP_RESAMPLES = 1000
BOOTSTRAP_CONFIDENCE = 0.95
BOOTSTRAP_SEED = 0
SHORTCUT_LEAK_AUC = 0.55
CHANCE_AUC = 0.5
STD_DDOF = 1

GPU_LOCK_TIMEOUT_SECONDS = 43200
GPU_LOCK_POLL_SECONDS = 2.0
GPU_LOCK_EMPTY_GRACE_SECONDS = 5.0
PROCESS_QUERY_LIMITED_INFORMATION = 0x1000

LABEL_BONAFIDE = 0
LABEL_SPOOF = 1
BONAFIDE_NAME = "bonafide"
SPOOF_NAME = "spoof"
BONAFIDE_SYSTEM_ID = "-"
TRAIN_UTTERANCE_PREFIX = "LA_T_"
AUDIO_SUFFIX = ".flac"

SPEAKER_COLUMN = "speaker_id"
UTTERANCE_COLUMN = "utterance_id"
UNUSED_PROTOCOL_COLUMN = "protocol_unused"
SYNTHESIZER_COLUMN = "synthesizer"
RAW_LABEL_COLUMN = "label_name"
LABEL_COLUMN = "label"
FILE_COLUMN = "file"
PATH_COLUMN = "path"

PROTOCOL_COLUMNS = (
    SPEAKER_COLUMN,
    UTTERANCE_COLUMN,
    UNUSED_PROTOCOL_COLUMN,
    SYNTHESIZER_COLUMN,
    RAW_LABEL_COLUMN,
)

# Этап 0: границы поиска Optuna, не выбранная рабочая точка.
# Длина кадра переводит порог тишины в долю молчания.
STAGE0_SILENCE_THRESHOLD_LOW = 1e-4
STAGE0_SILENCE_THRESHOLD_HIGH = 1e-1
STAGE0_LOGREG_C_LOW = 1e-3
STAGE0_LOGREG_C_HIGH = 1e2
STAGE0_CLASS_WEIGHT_NONE = "none"
STAGE0_CLASS_WEIGHT_POLICIES = ("balanced", STAGE0_CLASS_WEIGHT_NONE)
STAGE0_LOGREG_SOLVERS = ("lbfgs", "liblinear")
STAGE0_LOGREG_MAX_ITER = 1000
STAGE0_LOGREG_TOL = 1e-4
STAGE0_FRAME_SECONDS = 0.02
STAGE0_LOUDNESS_EPS = 1e-12
STAGE0_AMPLITUDE_DB_FACTOR = 20.0
STAGE0_FEATURE_NAMES = ("duration", "silence_ratio", "loudness", "sample_rate")
STAGE0_SCALE_FEATURES = True
STAGE0_OPTUNA_SAMPLER_SEED = 0
STAGE0_OBJECTIVE_ATOL = 1e-12
AUC_MATCH_ATOL = 1e-9

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "Data"
TRAIN_PROTOCOL_PATH = DATA_DIR / "train" / "targets.txt"
TRAIN_AUDIO_DIR = DATA_DIR / "train" / "flac"
TEST_AUDIO_DIR = DATA_DIR / "public_test"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
STAGE0_DIR = OUTPUT_DIR / "stage0"
TRIMMED_CACHE_DIR = STAGE0_DIR / "trimmed"
GPU_LOCK_PATH = OUTPUT_DIR / "gpu.lock"
STAGE0_VERDICT_PATH = STAGE0_DIR / "verdict.json"
STAGE0_METRICS_PATH = STAGE0_DIR / "metrics.json"
STAGE1_DIR = OUTPUT_DIR / "stage1"
STAGE1_METRICS_PATH = STAGE1_DIR / "metrics.json"
STAGE1_CACHE_PATH = STAGE1_DIR / "cepstral_moments.npz"
STAGE2_DIR = OUTPUT_DIR / "stage2"
STAGE2_METRICS_PATH = STAGE2_DIR / "metrics.json"
STAGE3_DIR = OUTPUT_DIR / "stage3"
STAGE3_METRICS_PATH = STAGE3_DIR / "metrics.json"
STAGE4_DIR = OUTPUT_DIR / "stage4"
STAGE4_METRICS_PATH = STAGE4_DIR / "metrics.json"
SAMPLE_SUBMISSION_PATH = PROJECT_ROOT / "public_sample_submission.csv"
SUBMISSION_ID_COLUMN = "idx"
SUBMISSION_SCORE_COLUMN = "score"
SUBMISSION_COLUMNS = (SUBMISSION_ID_COLUMN, SUBMISSION_SCORE_COLUMN)
SCORE_FLOAT_FORMAT = ".8f"

# Сводка по семенам: среднее каждого фолда, затем summarize_fold_aucs.
# worst_auc: худшее из средних по фолдам, не минимум худших по семенам.
SEED_AGGREGATION = "af_common.mean_fold_aucs + summarize_fold_aucs"


def use_data_dir(data_dir: os.PathLike | str | None) -> None:
    """Перенаправить train и тест на другой каталог. Выводы остаются рядом с этим файлом."""
    global DATA_DIR, TRAIN_PROTOCOL_PATH, TRAIN_AUDIO_DIR, TEST_AUDIO_DIR
    if data_dir is None:
        return
    root = Path(data_dir)
    DATA_DIR = root
    TRAIN_PROTOCOL_PATH = root / "train" / "targets.txt"
    TRAIN_AUDIO_DIR = root / "train" / "flac"
    TEST_AUDIO_DIR = root / "public_test"

Fold = namedtuple("Fold", ("fold_id", "train_index", "val_index"))
Fold.__doc__ = (
    "Один фолд LOSO. fold_id: удержанный синтезатор. "
    "train_index и val_index: позиции в таблице load_loso_table()."
)


def repo_relative(path: Path) -> str:
    """Путь относительно корня репозитория, если файл внутри него."""
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def audio_path(utterance_id: str) -> Path:
    name = utterance_id if str(utterance_id).endswith(AUDIO_SUFFIX) else f"{utterance_id}{AUDIO_SUFFIX}"
    return TRAIN_AUDIO_DIR / name


def submission_path(seed: int) -> Path:
    """Путь outputs/submission_seed_{seed}.csv. Его пишут следующие этапы."""
    return OUTPUT_DIR / f"submission_seed_{int(seed)}.csv"


def _assert_not_test(path: Path) -> None:
    resolved = Path(path).resolve()
    test_root = TEST_AUDIO_DIR.resolve()
    if resolved == test_root or test_root in resolved.parents:
        raise RuntimeError(f"test split is off-limits: {resolved}")


def _coerce_labels(series: pd.Series) -> np.ndarray:
    if pd.api.types.is_numeric_dtype(series):
        values = series.to_numpy(dtype=np.int64)
    else:
        mapped = series.astype(str).map(
            {BONAFIDE_NAME: LABEL_BONAFIDE, SPOOF_NAME: LABEL_SPOOF}
        )
        if mapped.isna().any():
            unknown = sorted(set(series.astype(str)[mapped.isna()].tolist()))
            raise ValueError(f"unknown labels: {unknown}")
        values = mapped.to_numpy(dtype=np.int64)
    allowed = np.isin(values, (LABEL_BONAFIDE, LABEL_SPOOF))
    if not bool(np.all(allowed)):
        raise ValueError("labels must be bonafide (0) or spoof (1)")
    return values


def _read_labeled_table(path: Path) -> pd.DataFrame:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    _assert_not_test(path)
    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(path)
        if RAW_LABEL_COLUMN in frame.columns and LABEL_COLUMN not in frame.columns:
            frame = frame.rename(columns={RAW_LABEL_COLUMN: LABEL_COLUMN})
    else:
        frame = pd.read_csv(
            path,
            sep=r"\s+",
            header=None,
            names=list(PROTOCOL_COLUMNS),
            engine="python",
        )
        frame = frame.drop(columns=[UNUSED_PROTOCOL_COLUMN])
        frame = frame.rename(columns={RAW_LABEL_COLUMN: LABEL_COLUMN})

    missing = [column for column in (SPEAKER_COLUMN, UTTERANCE_COLUMN, SYNTHESIZER_COLUMN, LABEL_COLUMN) if column not in frame.columns]
    if missing:
        raise ValueError(f"{path} is missing columns {missing}")

    frame[SPEAKER_COLUMN] = frame[SPEAKER_COLUMN].astype(str)
    frame[UTTERANCE_COLUMN] = frame[UTTERANCE_COLUMN].astype(str)
    frame[SYNTHESIZER_COLUMN] = frame[SYNTHESIZER_COLUMN].astype(str)
    if FILE_COLUMN not in frame.columns:
        frame[FILE_COLUMN] = frame[UTTERANCE_COLUMN].map(
            lambda utterance: utterance if utterance.endswith(AUDIO_SUFFIX) else f"{utterance}{AUDIO_SUFFIX}"
        )
    if PATH_COLUMN not in frame.columns:
        frame[PATH_COLUMN] = frame[UTTERANCE_COLUMN].map(lambda utterance: str(audio_path(utterance)))
    else:
        frame[PATH_COLUMN] = frame[PATH_COLUMN].astype(str)

    labels = _coerce_labels(frame[LABEL_COLUMN])
    frame[LABEL_COLUMN] = labels

    prefix_ok = frame[UTTERANCE_COLUMN].str.startswith(TRAIN_UTTERANCE_PREFIX)
    if not bool(prefix_ok.all()):
        raise RuntimeError("manifest contains non-train utterances; test is off-limits")

    test_root = str(TEST_AUDIO_DIR.resolve())
    allowed_roots = (
        str(TRAIN_AUDIO_DIR.resolve()),
        str(TRIMMED_CACHE_DIR.resolve()),
    )
    for raw_path in frame[PATH_COLUMN].tolist():
        resolved = str(Path(raw_path).resolve())
        if resolved == test_root or resolved.startswith(test_root + os.sep):
            raise RuntimeError(f"test audio is off-limits: {resolved}")
        if not any(
            resolved == root or resolved.startswith(root + os.sep) for root in allowed_roots
        ):
            raise RuntimeError(f"audio path is outside the train set: {resolved}")

    keep = [
        SPEAKER_COLUMN,
        UTTERANCE_COLUMN,
        FILE_COLUMN,
        PATH_COLUMN,
        SYNTHESIZER_COLUMN,
        LABEL_COLUMN,
    ]
    extra = [column for column in frame.columns if column not in keep and column != RAW_LABEL_COLUMN]
    ordered = frame[keep + extra].copy()
    return ordered.reset_index(drop=True)


def _resolve_manifest(manifest: os.PathLike | None) -> Path:
    if manifest is not None:
        return Path(manifest)
    if STAGE0_VERDICT_PATH.is_file():
        payload = json.loads(STAGE0_VERDICT_PATH.read_text(encoding="utf-8"))
        recorded = payload.get("data_manifest")
        if recorded:
            recorded_path = Path(recorded)
            if not recorded_path.is_absolute():
                recorded_path = PROJECT_ROOT / recorded_path
            if recorded_path.is_file():
                return recorded_path
    return TRAIN_PROTOCOL_PATH


def load_loso_table(manifest: os.PathLike | None = None) -> pd.DataFrame:
    """Только размеченный train.

    Без аргумента берётся манифест вердикта этапа 0, иначе исходный протокол.
    TRAIN_PROTOCOL_PATH принудительно читает исходную таблицу. Тест не включается.
    """
    return _read_labeled_table(_resolve_manifest(manifest))


def _contiguous_slices(n_items: int, n_bins: int) -> list[tuple[int, int]]:
    if n_bins <= 0:
        raise ValueError("n_bins must be positive")
    base = n_items // n_bins
    remainder = n_items % n_bins
    slices: list[tuple[int, int]] = []
    start = 0
    for bin_index in range(n_bins):
        count = base + (1 if bin_index < remainder else 0)
        stop = start + count
        slices.append((start, stop))
        start = stop
    return slices


def iter_loso_folds(table: pd.DataFrame) -> list[Fold]:
    """По одному фолду на spoof-синтезатор, идентификаторы по возрастанию.

    Валидация фолда S: все его строки плюс непересекающийся кусок bona fide.
    Обучение: дополнение.
    """
    if not isinstance(table.index, pd.RangeIndex) or table.index.start != 0 or table.index.step != 1:
        raise ValueError("table must use a reset RangeIndex; use load_loso_table()")
    if len(table.index) != len(table):
        raise ValueError("table index does not match row count")

    synthesizer = table[SYNTHESIZER_COLUMN].to_numpy()
    labels = table[LABEL_COLUMN].to_numpy(dtype=np.int64)
    is_bonafide = synthesizer == BONAFIDE_SYSTEM_ID
    if is_bonafide.shape[0] == 0 or not bool(np.any(is_bonafide)):
        raise ValueError("no bonafide rows")
    if not bool(np.all(labels[is_bonafide] == LABEL_BONAFIDE)):
        raise ValueError("bonafide sentinel rows must be labeled bonafide")
    if not bool(np.all(labels[~is_bonafide] == LABEL_SPOOF)):
        raise ValueError("non-bonafide synthesizer rows must be labeled spoof")

    spoof_ids = sorted(set(synthesizer[~is_bonafide].tolist()))
    bonafide_positions = np.flatnonzero(is_bonafide)
    slices = _contiguous_slices(int(bonafide_positions.size), len(spoof_ids))

    folds: list[Fold] = []
    covered = np.zeros(len(table), dtype=bool)
    for fold_index, fold_id in enumerate(spoof_ids):
        val_mask = synthesizer == fold_id
        start, stop = slices[fold_index]
        val_mask = val_mask.copy()
        val_mask[bonafide_positions[start:stop]] = True
        if bool(np.any(covered & val_mask)):
            raise RuntimeError("LOSO validation folds overlap")
        covered |= val_mask
        val_index = np.flatnonzero(val_mask).astype(np.int64, copy=False)
        train_index = np.flatnonzero(~val_mask).astype(np.int64, copy=False)
        if labels[val_index].min() == labels[val_index].max():
            raise ValueError(f"fold {fold_id} validation has a single class")
        if labels[train_index].min() == labels[train_index].max():
            raise ValueError(f"fold {fold_id} train has a single class")
        folds.append(Fold(str(fold_id), train_index, val_index))

    if not bool(np.all(covered)):
        raise RuntimeError("LOSO folds do not cover every labeled row")
    return folds


def class_weight_arg(policy: str):
    """class_weight sklearn: none даёт None, остальные политики передаются как есть."""
    policy = str(policy)
    if policy == STAGE0_CLASS_WEIGHT_NONE:
        return None
    if policy not in STAGE0_CLASS_WEIGHT_POLICIES:
        raise ValueError(f"unknown class_weight policy: {policy}")
    return policy


def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Полифазный ресемплер. Этапы 0-4 делят один фильтр, чтобы волна не зависела от этапа."""
    from math import gcd

    from scipy.signal import resample_poly

    values = np.asarray(audio, dtype=np.float32)
    orig = int(orig_sr)
    target = int(target_sr)
    if orig == target or values.size == 0:
        return values
    divisor = gcd(orig, target)
    return np.asarray(resample_poly(values, target // divisor, orig // divisor), dtype=np.float32)


def average_ranks(scores) -> np.ndarray:
    """Средние ранги с единицы. Ничьи делят среднее мест. Сортировка mergesort."""
    values = np.asarray(scores, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    sorted_scores = values[order]
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    n_scores = int(values.size)
    while start < n_scores:
        stop = start + 1
        while stop < n_scores and sorted_scores[stop] == sorted_scores[start]:
            stop += 1
        ranks[order[start:stop]] = (start + 1 + stop) / 2.0
        start = stop
    return ranks


def binary_auc(y_true, y_score) -> float:
    """ROC AUC по статистике Манна-Уитни. Больший скор означает spoof."""
    y_raw = np.asarray(y_true)
    scores = np.asarray(y_score, dtype=np.float64)
    if y_raw.shape[0] != scores.shape[0]:
        raise ValueError("y_true and y_score must have the same length")
    if y_raw.size == 0:
        raise ValueError("binary_auc received no rows")
    if not np.isfinite(scores).all():
        raise ValueError("y_score contains non-finite values")
    if np.issubdtype(y_raw.dtype, np.floating):
        labels = np.rint(y_raw).astype(np.int64)
    else:
        labels = y_raw.astype(np.int64, copy=False)
    positive = labels == LABEL_SPOOF
    negative = labels == LABEL_BONAFIDE
    if int(positive.sum() + negative.sum()) != int(labels.size):
        raise ValueError("labels must be 0 (bonafide) or 1 (spoof)")
    n_pos = int(positive.sum())
    n_neg = int(negative.sum())
    if n_pos == 0 or n_neg == 0:
        raise ValueError("binary_auc requires both classes")

    ranks = average_ranks(scores)
    sum_positive_ranks = float(ranks[positive].sum())
    auc = (sum_positive_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def _auc_vector(fold_aucs: dict[str, float]) -> tuple[list[str], np.ndarray]:
    if not fold_aucs:
        raise ValueError("fold_aucs is empty")
    keys = sorted(fold_aucs)
    values = np.array([float(fold_aucs[key]) for key in keys], dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("fold AUCs must be finite")
    return keys, values


def summarize_fold_aucs(fold_aucs: dict[str, float]) -> dict:
    """Среднее, худший фолд, выборочное СКО и перцентильный бутстреп-интервал среднего AUC."""
    keys, values = _auc_vector(fold_aucs)
    mean_auc = float(values.mean())
    worst_auc = float(values.min())
    if values.size > STD_DDOF:
        std_auc = float(values.std(ddof=STD_DDOF))
    else:
        std_auc = 0.0
    generator = np.random.default_rng(BOOTSTRAP_SEED)
    draw = generator.integers(0, values.size, size=(BOOTSTRAP_RESAMPLES, values.size))
    boot_means = values[draw].mean(axis=1)
    tail = (1.0 - BOOTSTRAP_CONFIDENCE) / 2.0
    ci_low = float(np.quantile(boot_means, tail))
    ci_high = float(np.quantile(boot_means, 1.0 - tail))
    return {
        "mean_auc": mean_auc,
        "worst_auc": worst_auc,
        "std_auc": std_auc,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "fold_aucs": {key: float(fold_aucs[key]) for key in keys},
    }


def mean_fold_aucs(fold_maps) -> dict[str, float]:
    """Среднее каждого фолда по семенам. Наборы идентификаторов должны совпадать."""
    maps = [dict(item) for item in fold_maps]
    if not maps:
        raise ValueError("no seeds")
    keys = set(maps[0])
    for other in maps[1:]:
        if set(other) != keys:
            raise RuntimeError("fold ids differ across seeds")
    return {key: float(np.mean([float(item[key]) for item in maps])) for key in sorted(keys)}


def read_submission_ids(
    sample_path: os.PathLike | str | None = None,
    test_dir: os.PathLike | str | None = None,
) -> list[tuple[str, Path]]:
    """Идентификаторы из образца сдачи, сопоставленные с файлами публичного теста.

    Колонка score не читается.
    """
    sample = Path(sample_path) if sample_path is not None else SAMPLE_SUBMISSION_PATH
    audio_dir = Path(test_dir) if test_dir is not None else TEST_AUDIO_DIR
    rows: list[tuple[str, Path]] = []
    with sample.open(encoding="utf-8", newline="") as handle:
        header = handle.readline().strip().split(",")
        if SUBMISSION_ID_COLUMN not in header:
            raise RuntimeError(f"sample submission has no {SUBMISSION_ID_COLUMN} column")
        idx_at = header.index(SUBMISSION_ID_COLUMN)
        for line in handle:
            if not line.strip():
                continue
            name = line.split(",")[idx_at].strip()
            path = audio_dir / name
            if not path.is_file():
                raise FileNotFoundError(path)
            rows.append((name, path))
    if not rows:
        raise RuntimeError("sample submission has no rows")
    return rows


def write_submission_csv(path: os.PathLike | str, ids: list[tuple[str, Path]], scores) -> None:
    """CSV с колонками idx,score и восемью знаками после запятой."""
    import csv

    values = np.asarray(scores, dtype=np.float64)
    if int(values.shape[0]) != len(ids):
        raise RuntimeError("score count does not match the sample submission")
    if not np.isfinite(values).all():
        raise RuntimeError("submission scores must be finite")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(list(SUBMISSION_COLUMNS))
        for (name, _audio), score in zip(ids, values, strict=True):
            writer.writerow([name, format(float(score), SCORE_FLOAT_FORMAT)])


def is_shortcut_leak(mean_auc: float) -> bool:
    """True, если средний AUC этапа 0 выше порога shortcut/утечки."""
    return float(mean_auc) > SHORTCUT_LEAK_AUC


def set_global_seed(seed: int) -> None:
    """Зафиксировать зерно Python, NumPy и PyTorch, если он установлен."""
    import random

    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def accept_stage(previous: dict, candidate: dict) -> bool:
    """Оставить следующий этап, только если средний AUC вырос достаточно, а худший фолд не упал."""
    mean_gain = float(candidate["mean_auc"]) - float(previous["mean_auc"])
    worst_held = float(candidate["worst_auc"]) >= float(previous["worst_auc"])
    return mean_gain >= MIN_MEAN_AUC_GAIN and worst_held


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes

        handle = ctypes.windll.kernel32.OpenProcess(
            PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid)
        )
        if handle:
            ctypes.windll.kernel32.CloseHandle(handle)
            return True
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _lock_holder_pid(path: Path) -> int | None:
    try:
        text = path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        return None


def _lock_is_stale(path: Path) -> bool:
    holder = _lock_holder_pid(path)
    if holder is not None:
        return not _pid_alive(holder)
    try:
        age = time.time() - path.stat().st_mtime
    except OSError:
        return True
    return age >= GPU_LOCK_EMPTY_GRACE_SECONDS


def _release_lock_if_owner(path: Path) -> None:
    if _lock_holder_pid(path) != os.getpid():
        return
    try:
        os.remove(path)
    except FileNotFoundError:
        return


@contextmanager
def gpu_lock():
    """Исключительная блокировка outputs/gpu.lock: этапы GPU не делят устройство.

    Файл создаётся с O_EXCL и хранит pid. Живой держатель ждётся до таймаута.
    Мёртвая блокировка снимается. Освобождение в finally.
    """
    lock_path = GPU_LOCK_PATH
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + GPU_LOCK_TIMEOUT_SECONDS
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    if hasattr(os, "O_BINARY"):
        flags |= os.O_BINARY
    fd = None
    while fd is None:
        try:
            fd = os.open(str(lock_path), flags)
        except FileExistsError:
            if _lock_is_stale(lock_path):
                try:
                    os.remove(lock_path)
                except FileNotFoundError:
                    pass
                continue
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"timed out after {GPU_LOCK_TIMEOUT_SECONDS}s waiting for {lock_path}"
                )
            time.sleep(GPU_LOCK_POLL_SECONDS)
    try:
        os.write(fd, f"{os.getpid()}\n".encode("ascii"))
        yield lock_path
    finally:
        try:
            os.close(fd)
        except OSError:
            pass
        _release_lock_if_owner(lock_path)


__all__ = [
    "BOOTSTRAP_CONFIDENCE",
    "BOOTSTRAP_RESAMPLES",
    "BOOTSTRAP_SEED",
    "CHANCE_AUC",
    "LABEL_COLUMN",
    "MIN_MEAN_AUC_GAIN",
    "N_OPTUNA_TRIALS",
    "SEEDS",
    "SHORTCUT_LEAK_AUC",
    "SYNTHESIZER_COLUMN",
    "TRAIN_PROTOCOL_PATH",
    "Fold",
    "SCORE_FLOAT_FORMAT",
    "SEED_AGGREGATION",
    "SUBMISSION_COLUMNS",
    "SUBMISSION_ID_COLUMN",
    "SUBMISSION_SCORE_COLUMN",
    "accept_stage",
    "audio_path",
    "average_ranks",
    "binary_auc",
    "class_weight_arg",
    "gpu_lock",
    "is_shortcut_leak",
    "iter_loso_folds",
    "load_loso_table",
    "mean_fold_aucs",
    "read_submission_ids",
    "repo_relative",
    "resample_audio",
    "set_global_seed",
    "submission_path",
    "summarize_fold_aucs",
    "use_data_dir",
    "write_submission_csv",
]
