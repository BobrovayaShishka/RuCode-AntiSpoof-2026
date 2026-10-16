"""Общие LOSO, метрики, смешивание рангов и сдача для этапов 10 и 11.

Фолды и сводки AUC не переопределяются. Тест соревнования метрикой не скорится.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

import af_common
from experiments.config import (
    DISPLAY_METRIC_FORMAT,
    DISPLAY_SECONDS_FORMAT,
    DISPLAY_THRESHOLD_FORMAT,
    START_INDEX,
)
from experiments.decisions import accept_loso, explain_decision
from jsonutil import read_json, write_json
from ssl_stages.constants import (
    BASELINE_STAGE,
    BASELINE_STAGE_NAME,
    BASELINE_STAGE_REASON,
    DECISION_ACCEPT,
    DECISION_REJECT,
    FROZEN_SSL_BASELINE_NAME,
    FUSION_NAME,
    FUSION_WEIGHTS_FIT,
    INNER_MONITOR_POSITION,
    MIN_MEAN_AUC_GAIN,
    MOST_FOLDS_THRESHOLD,
    PRIMARY_AGGREGATION,
    PRIMARY_MEAN_NOTE,
    RANK_BASE,
    RANK_KIND,
    SEEDS,
    SIDECAR_SUFFIX,
    STRONG_MEAN_AUC_GAIN,
)
from stage4.constants import HEAD_RESULT_PATH, METRICS_PATH
from stage4.summaries import summarize_seeds

DEVICE_INDEX = 0


def emit(text: str) -> None:
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    safe = text.encode(encoding, errors="replace").decode(encoding, errors="replace")
    print(safe, flush=True)


def stage_dir(stage: int) -> Path:
    return Path(af_common.OUTPUT_DIR) / f"stage{int(stage)}"


def prepare_session(repo_root: str | None = None, data_dir: str | None = None, require_cuda: bool = True):
    """Зафиксировать семя, направить аудио на data_dir в Colab и вернуть устройство CUDA."""
    import torch

    from stage4.gpu_env import configure_process_env

    configure_process_env()
    root = Path(repo_root) if repo_root else af_common.PROJECT_ROOT
    if Path(root).resolve() != Path(af_common.PROJECT_ROOT).resolve():
        raise RuntimeError(f"imported af_common lives in {af_common.PROJECT_ROOT}, not {root}")
    if str(root) not in sys.path:
        sys.path.insert(START_INDEX, str(root))
    af_common.use_data_dir(data_dir)
    emit(f"data {af_common.DATA_DIR}")
    emit(f"protocol {af_common.TRAIN_PROTOCOL_PATH}")
    af_common.set_global_seed(int(SEEDS[START_INDEX]))
    if not torch.cuda.is_available():
        if require_cuda:
            raise RuntimeError(
                "Stages 10-11 need a CUDA GPU. In Colab use Runtime → Change runtime type → GPU, then rerun from the top."
            )
        return torch.device("cpu")
    emit(f"GPU {torch.cuda.get_device_name(DEVICE_INDEX)}")
    return torch.device("cuda", DEVICE_INDEX)


def configure_determinism() -> None:
    import torch

    from ssl_stages.constants import DETERMINISTIC_ALGORITHMS, TRAIN_ALLOW_TF32

    torch.backends.cuda.matmul.allow_tf32 = bool(TRAIN_ALLOW_TF32)
    torch.backends.cudnn.allow_tf32 = bool(TRAIN_ALLOW_TF32)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if not DETERMINISTIC_ALGORITHMS:
        return
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        try:
            torch.use_deterministic_algorithms(True)
        except Exception as exc:
            emit(f"deterministic algorithms unavailable: {exc}")
    except Exception as exc:
        emit(f"deterministic algorithms unavailable: {exc}")


def load_protocol_table():
    """Размеченный train и общие фолды LOSO. Публичный тест не загружается."""
    if not af_common.STAGE0_VERDICT_PATH.is_file():
        table = af_common.load_loso_table(af_common.TRAIN_PROTOCOL_PATH)
    else:
        table = af_common.load_loso_table()
    folds = af_common.iter_loso_folds(table)
    return table, folds


def _worst_fold_id(fold_aucs: dict) -> str:
    return str(min(fold_aucs, key=lambda key: float(fold_aucs[key])))


def acceptance_view(summary: dict) -> dict:
    """Словарь метрик для accept_loso. Полей теста и лидерборда нет."""
    aggregate = summary["aggregate"]
    fold_aucs = aggregate["fold_aucs"]
    per_seed = summary["per_seed"]
    return {
        "mean_auc": float(aggregate["mean_auc"]),
        "worst_auc": float(aggregate["worst_auc"]),
        "std_auc": float(aggregate["std_auc"]),
        "ci_low": float(aggregate["ci_low"]),
        "ci_high": float(aggregate["ci_high"]),
        "fold_aucs": {str(key): float(value) for key, value in fold_aucs.items()},
        "worst_fold_id": _worst_fold_id(fold_aucs),
        "worst_auc_min_over_seeds": float(min(float(item["worst_auc"]) for item in per_seed.values())),
        "aggregation": PRIMARY_AGGREGATION,
        "aggregation_note": PRIMARY_MEAN_NOTE,
    }


def summarize_loso(fold_aucs_by_seed: dict[int, dict[str, float]]) -> dict:
    summary = summarize_seeds(fold_aucs_by_seed)
    summary["view"] = acceptance_view(summary)
    return summary


def frozen_ssl_baseline() -> dict:
    """Замороженная линейная голова этапа 4. Загружается, не переобучается."""
    if not HEAD_RESULT_PATH.is_file():
        raise FileNotFoundError(HEAD_RESULT_PATH)
    payload = read_json(HEAD_RESULT_PATH)
    aggregate = payload["aggregate"]
    view = {
        "mean_auc": float(aggregate["mean_auc"]),
        "worst_auc": float(aggregate["worst_auc"]),
        "std_auc": float(aggregate["std_auc"]),
        "ci_low": float(aggregate["ci_low"]),
        "ci_high": float(aggregate["ci_high"]),
        "fold_aucs": {str(key): float(value) for key, value in aggregate["fold_aucs"].items()},
        "worst_fold_id": _worst_fold_id(aggregate["fold_aucs"]),
        "aggregation": PRIMARY_AGGREGATION,
        "aggregation_note": PRIMARY_MEAN_NOTE,
        "source": str(HEAD_RESULT_PATH),
    }
    return {
        "name": FROZEN_SSL_BASELINE_NAME,
        "checkpoint": payload["checkpoint"],
        "layer": int(payload["layer"]),
        "best_params": payload["best_params"],
        "metrics": view,
        "per_seed": payload["per_seed"],
    }


def stage2_record() -> dict:
    if not af_common.STAGE2_METRICS_PATH.is_file():
        raise FileNotFoundError(af_common.STAGE2_METRICS_PATH)
    payload = read_json(af_common.STAGE2_METRICS_PATH)
    if payload.get("kept_vs_stage1") is not True:
        raise RuntimeError("stage 2 is not the kept neural model")
    if af_common.STAGE3_METRICS_PATH.is_file():
        stage3 = read_json(af_common.STAGE3_METRICS_PATH)
        if stage3.get("kept_vs_stage2") is not False:
            raise RuntimeError("stage 3 was kept; BASELINE_STAGE must be revisited")
    if METRICS_PATH.is_file():
        stage4 = read_json(METRICS_PATH)
        if stage4.get("kept_vs_neural") is not False:
            raise RuntimeError("stage 4 was kept against the neural model; BASELINE_STAGE must be revisited")
    if int(BASELINE_STAGE) != 2:
        raise RuntimeError(BASELINE_STAGE_REASON)
    return payload


def decide_against(reference_name: str, candidate_name: str, reference: dict, candidate: dict) -> dict:
    """Приёмка только по LOSO. reference и candidate не должны нести тестовые скоры."""
    accepted = bool(accept_loso(reference, candidate))
    gain = float(candidate["mean_auc"]) - float(reference["mean_auc"])
    worst_delta = float(candidate["worst_auc"]) - float(reference["worst_auc"])
    worst_ok = float(candidate["worst_auc"]) >= float(reference["worst_auc"])
    accepted_at_strong = gain >= float(STRONG_MEAN_AUC_GAIN) and worst_ok
    text = explain_decision(reference_name, candidate_name, reference, candidate, accepted)
    return {
        "decision": DECISION_ACCEPT if accepted else DECISION_REJECT,
        "accepted_new": accepted,
        "mean_auc_delta": gain,
        "worst_auc_delta": worst_delta,
        "min_mean_auc_gain": float(MIN_MEAN_AUC_GAIN),
        "strong_mean_auc_gain": float(STRONG_MEAN_AUC_GAIN),
        "strong_threshold_would_change_decision": bool(accepted) != bool(accepted_at_strong),
        "summary_ru": text,
        "reference_name": reference_name,
        "candidate_name": candidate_name,
    }


def print_decision(report: dict) -> None:
    emit(f"РЕШЕНИЕ: {report['decision']}")
    emit(report["summary_ru"])
    emit(
        "delta mean AUC "
        + format(float(report["mean_auc_delta"]), DISPLAY_METRIC_FORMAT)
        + " (порог "
        + format(float(report["min_mean_auc_gain"]), DISPLAY_THRESHOLD_FORMAT)
        + ", сильная полоса "
        + format(float(report["strong_mean_auc_gain"]), DISPLAY_THRESHOLD_FORMAT)
        + ")"
    )
    emit("delta worst fold " + format(float(report["worst_auc_delta"]), DISPLAY_METRIC_FORMAT))
    changed = "да" if report["strong_threshold_would_change_decision"] else "нет"
    emit(f"Порог {format(float(STRONG_MEAN_AUC_GAIN), DISPLAY_THRESHOLD_FORMAT)} изменил бы решение: {changed}")
    emit(PRIMARY_MEAN_NOTE)


def inner_monitor_split(table, train_index: np.ndarray, loso_fold_id: str) -> tuple[np.ndarray, np.ndarray]:
    """Один синтезатор train плюс кусок bona fide. Holdout LOSO не входит.

    Early stopping может смотреть этот сплит, но не loso_fold_id и не тест.
    """
    synthesizer = table[af_common.SYNTHESIZER_COLUMN].to_numpy()
    train_index = np.asarray(train_index, dtype=np.int64)
    train_synth = synthesizer[train_index]
    spoof_ids = sorted(
        {value for value in train_synth.tolist() if value != af_common.BONAFIDE_SYSTEM_ID}
    )
    if loso_fold_id in spoof_ids:
        raise RuntimeError("LOSO holdout synthesizer is inside the training indices")
    if len(spoof_ids) < 2:
        raise RuntimeError("inner monitor needs at least two training synthesizers")
    monitor_id = spoof_ids[INNER_MONITOR_POSITION]
    spoof_rows = train_index[train_synth == monitor_id]
    bonafide_rows = train_index[train_synth == af_common.BONAFIDE_SYSTEM_ID]
    slices = af_common._contiguous_slices(int(bonafide_rows.size), len(spoof_ids))
    start, stop = slices[INNER_MONITOR_POSITION]
    bona_rows = bonafide_rows[start:stop]
    if spoof_rows.size == 0 or bona_rows.size == 0:
        raise RuntimeError("inner monitor split is empty")
    inner_val = np.sort(np.concatenate([spoof_rows, bona_rows]))
    held = set(int(value) for value in inner_val.tolist())
    inner_train = np.asarray([int(value) for value in train_index.tolist() if int(value) not in held], dtype=np.int64)
    labels = table[af_common.LABEL_COLUMN].to_numpy()
    for name, index in (("inner train", inner_train), ("inner monitor", inner_val)):
        if labels[index].min() == labels[index].max():
            raise RuntimeError(f"{name} has a single class")
    if bool(np.any(synthesizer[inner_val] == loso_fold_id)):
        raise RuntimeError("inner monitor contains the LOSO holdout")
    return inner_train, inner_val


def rank_normalize(scores: np.ndarray) -> np.ndarray:
    """Средние ранги при ничье, деление на длину оценки. Выше скор, выше ранг."""
    if RANK_KIND != "mergesort" or int(RANK_BASE) != 1:
        raise RuntimeError("rank fusion expects mergesort and 1-based average ranks")
    values = np.asarray(scores, dtype=np.float64)
    count = int(values.size)
    if count == 0:
        raise ValueError("rank_normalize received no rows")
    if not np.isfinite(values).all():
        raise ValueError("scores must be finite")
    return af_common.average_ranks(values) / float(count)


def equal_rank_average(columns: list[np.ndarray]) -> np.ndarray:
    """Равное среднее рангов. Вес смеси не подбирается."""
    if FUSION_WEIGHTS_FIT:
        raise RuntimeError(f"{FUSION_NAME} does not fit fusion weights")
    if len(columns) < 2:
        raise ValueError("rank averaging needs at least two models")
    ranked = [rank_normalize(column) for column in columns]
    width = int(ranked[0].shape[0])
    for column in ranked:
        if int(column.shape[0]) != width:
            raise ValueError("rank columns differ in length")
    return np.mean(np.stack(ranked, axis=0), axis=0)


def fold_win_report(ensemble_folds: dict[str, float], single_folds: dict[str, float]) -> dict:
    """По удержанному синтезатору: обгоняет ли ансамбль выбранную одиночную модель."""
    keys = sorted(set(ensemble_folds) & set(single_folds))
    if not keys:
        raise ValueError("no shared folds")
    wins = []
    rows = []
    for key in keys:
        delta = float(ensemble_folds[key]) - float(single_folds[key])
        won = delta > 0.0
        wins.append(won)
        rows.append({"fold_id": key, "delta": delta, "ensemble_wins": won})
    fraction = float(np.mean(np.asarray(wins, dtype=np.float64)))
    return {
        "folds": rows,
        "win_fraction": fraction,
        "better_in_most_folds": fraction > float(MOST_FOLDS_THRESHOLD),
        "most_folds_threshold": float(MOST_FOLDS_THRESHOLD),
        "n_folds": len(keys),
        "n_wins": int(sum(wins)),
    }


def print_fold_wins(report: dict, single_name: str) -> None:
    emit(
        f"Ансамбль лучше {single_name} на {report['n_wins']}/{report['n_folds']} фолдах "
        f"(доля {format(float(report['win_fraction']), DISPLAY_METRIC_FORMAT)}). "
        f"Лучше на большинстве фолдов: {'да' if report['better_in_most_folds'] else 'нет'}."
    )
    for row in report["folds"]:
        mark = "лучше" if row["ensemble_wins"] else "не лучше"
        emit(f"  {row['fold_id']}: {mark}, delta {format(float(row['delta']), DISPLAY_METRIC_FORMAT)}")


def write_score_csv(path: Path, ids: list[tuple[str, Path]], scores: np.ndarray) -> None:
    af_common.write_submission_csv(path, ids, scores)


def sidecar_path(seed: int) -> Path:
    return af_common.submission_path(int(seed)).with_suffix(SIDECAR_SUFFIX)


def write_sidecar(seed: int, payload: dict) -> Path:
    path = sidecar_path(int(seed))
    body = dict(payload)
    body["seed"] = int(seed)
    write_json(path, body)
    return path


def submission_ids() -> list[tuple[str, Path]]:
    return af_common.read_submission_ids()


def format_seconds(seconds: float) -> str:
    return format(float(seconds), DISPLAY_SECONDS_FORMAT)


def print_metrics(name: str, view: dict) -> None:
    emit(
        f"{name}: mean {format(float(view['mean_auc']), DISPLAY_METRIC_FORMAT)} "
        f"worst {view.get('worst_fold_id', '')} {format(float(view['worst_auc']), DISPLAY_METRIC_FORMAT)} "
        f"std {format(float(view['std_auc']), DISPLAY_METRIC_FORMAT)} "
        f"CI [{format(float(view['ci_low']), DISPLAY_METRIC_FORMAT)}, "
        f"{format(float(view['ci_high']), DISPLAY_METRIC_FORMAT)}]"
    )


def baseline_stage_banner() -> str:
    return (
        f"BASELINE_STAGE={int(BASELINE_STAGE)} ({BASELINE_STAGE_NAME}). {BASELINE_STAGE_REASON}."
    )
