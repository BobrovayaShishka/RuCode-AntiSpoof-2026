"""decisions.json: цепочка, которую этапы 5-9 читают и дополняют.

Приёмка: af_common.accept_stage. Видны только среднее LOSO и худший фолд.
Метки теста и лидерборд из словаря метрик отбрасываются.
"""

from __future__ import annotations

import json
from pathlib import Path

import af_common
from experiments.config import (
    BASELINE_NAME,
    BASELINE_SOURCE,
    CONTINUATION_CODE_VERSION,
    DECISIONS_FILENAME,
    DECISIONS_VERSION,
    DISPLAY_METRIC_FORMAT,
    DISPLAY_THRESHOLD_FORMAT,
    FIRST_STAGE,
    FORBIDDEN_DECISION_KEYS,
    LAST_STAGE,
    MIN_MEAN_AUC_GAIN,
    STAGE_NOTEBOOKS,
    STALE_STEP,
    START_INDEX,
    STATUS_COMPLETE,
    STRONG_MEAN_AUC_GAIN,
    baseline_hparams,
    baseline_metrics,
    protocol_block,
)
from jsonutil import write_json


def decisions_path() -> Path:
    return Path(af_common.OUTPUT_DIR) / DECISIONS_FILENAME


def config_name(hparams: dict) -> str:
    parts = [
        f"lcnn{int(hparams['lcnn_width'])}",
        str(hparams["frontend"]),
        str(hparams["pooling"]),
        str(hparams["loss"]),
    ]
    if hparams.get("rawboost"):
        parts.append("rawboost")
    if hparams.get("codec"):
        parts.append("codec")
    return "_".join(parts)


def fresh_payload() -> dict:
    hparams = baseline_hparams()
    name = config_name(hparams)
    if name != BASELINE_NAME:
        raise RuntimeError(f"baseline name {BASELINE_NAME} does not match {name}")
    winner = {
        "name": name,
        "stage": "baseline",
        "source": BASELINE_SOURCE,
        "hparams": hparams,
        "metrics": baseline_metrics(),
    }
    return {
        "version": DECISIONS_VERSION,
        "code_version": CONTINUATION_CODE_VERSION,
        "protocol": protocol_block(),
        "baseline": {
            "name": name,
            "source": BASELINE_SOURCE,
            "hparams": hparams,
            "metrics": baseline_metrics(),
        },
        "current_winner": winner,
        "stages": {},
    }


def load_decisions() -> dict | None:
    path = decisions_path()
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def save_decisions(payload: dict) -> None:
    write_json(decisions_path(), payload, ensure_ascii=False)


def assert_baseline_matches_stage2() -> None:
    """Именованный baseline должен совпасть со сводкой этапа 2, если файл уже есть."""
    metrics_path = af_common.STAGE2_METRICS_PATH
    if not metrics_path.is_file():
        return
    recorded = json.loads(metrics_path.read_text(encoding="utf-8"))
    aggregate = recorded.get("aggregate") or {}
    expected = baseline_metrics()
    for key in ("mean_auc", "worst_auc"):
        if key not in aggregate:
            raise RuntimeError(f"stage-2 metrics have no aggregate.{key}")
        if abs(float(aggregate[key]) - float(expected[key])) > float(af_common.AUC_MATCH_ATOL):
            raise RuntimeError(
                f"experiments.config baseline {key} does not match outputs/stage2/metrics.json"
            )


def ensure_baseline() -> dict:
    assert_baseline_matches_stage2()
    payload = load_decisions()
    if payload is None:
        payload = fresh_payload()
        save_decisions(payload)
        return payload
    if payload.get("version") != DECISIONS_VERSION:
        raise RuntimeError(
            f"{decisions_path()} has version {payload.get('version')}; expected {DECISIONS_VERSION}"
        )
    if "baseline" not in payload or "current_winner" not in payload or "stages" not in payload:
        raise RuntimeError(f"{decisions_path()} is missing baseline, current_winner, or stages")
    return payload


def require_prior_stage(payload: dict, stage: int) -> None:
    if int(stage) == FIRST_STAGE:
        return
    prior = int(stage) - int(STALE_STEP)
    record = payload.get("stages", {}).get(str(prior))
    if not isinstance(record, dict) or record.get("status") != STATUS_COMPLETE:
        notebook = STAGE_NOTEBOOKS.get(prior, f"stage {prior}")
        raise RuntimeError(
            f"Stage {int(stage)} needs a completed stage {prior} decision in {decisions_path()}. "
            f"Run {notebook} to completion first."
        )


def accept_loso(reference: dict, candidate: dict) -> bool:
    """True только если средний AUC LOSO проходит порог и худший фолд не падает."""
    for blob in (reference, candidate):
        overlap = set(FORBIDDEN_DECISION_KEYS) & set(blob)
        if overlap:
            raise RuntimeError(f"acceptance cannot see test or leaderboard fields: {sorted(overlap)}")
    return bool(af_common.accept_stage(reference, candidate))


def explain_decision(
    reference_name: str,
    candidate_name: str,
    reference: dict,
    candidate: dict,
    accepted: bool,
) -> str:
    gain = float(candidate["mean_auc"]) - float(reference["mean_auc"])
    worst_ok = float(candidate["worst_auc"]) >= float(reference["worst_auc"])
    gain_text = format(gain, DISPLAY_METRIC_FORMAT)
    threshold_text = format(float(MIN_MEAN_AUC_GAIN), DISPLAY_THRESHOLD_FORMAT)
    strong_text = format(float(STRONG_MEAN_AUC_GAIN), DISPLAY_THRESHOLD_FORMAT)
    if accepted and gain >= float(STRONG_MEAN_AUC_GAIN):
        band = f"прирост среднего AUC {gain_text} не меньше сильной полосы {strong_text}"
    elif accepted:
        band = (
            f"прирост среднего AUC {gain_text} не меньше порога {threshold_text}, "
            f"но ниже сильной полосы {strong_text}"
        )
    elif not worst_ok:
        band = (
            f"худший фолд {format(float(candidate['worst_auc']), DISPLAY_METRIC_FORMAT)} "
            f"ниже эталона {format(float(reference['worst_auc']), DISPLAY_METRIC_FORMAT)}; "
            f"прирост среднего {gain_text}"
        )
    else:
        band = f"прирост среднего AUC {gain_text} меньше порога {threshold_text}"
    worst_id = candidate.get("worst_fold_id", "")
    worst_note = f" Худший фолд кандидата: {worst_id}." if worst_id else ""
    if accepted:
        return f"Принимаем {candidate_name} вместо {reference_name}: {band}. Худший фолд не ухудшился.{worst_note}"
    return f"Оставляем {reference_name}: {band}. Новая модель {candidate_name} не принята.{worst_note}"


def choose_winner(reference: dict, new_arms: list[dict]) -> tuple[dict | None, str]:
    """Выбрать новую ветку, прошедшую accept_loso. Ничья: средний AUC, затем худший фолд."""
    accepted = [arm for arm in new_arms if accept_loso(reference["metrics"], arm["metrics"])]
    if not accepted:
        reasons = [
            explain_decision(reference["name"], arm["name"], reference["metrics"], arm["metrics"], False)
            for arm in new_arms
        ]
        return None, " ".join(reasons)
    accepted.sort(key=lambda arm: (float(arm["metrics"]["mean_auc"]), float(arm["metrics"]["worst_auc"])), reverse=True)
    best = accepted[START_INDEX]
    reason = explain_decision(reference["name"], best["name"], reference["metrics"], best["metrics"], True)
    skipped = [arm["name"] for arm in new_arms if arm["name"] != best["name"]]
    if skipped:
        reason = f"{reason} Другие кандидаты этого этапа: {', '.join(skipped)}."
    return best, reason


def preview_text(stage: int) -> str:
    if int(stage) < FIRST_STAGE or int(stage) > LAST_STAGE:
        raise ValueError(f"stage must be {FIRST_STAGE}..{LAST_STAGE}")
    payload = ensure_baseline()
    require_prior_stage(payload, int(stage))
    winner = payload["current_winner"]
    metrics = winner["metrics"]
    return (
        f"Этап {int(stage)}. Эталон: {winner['name']} "
        f"(зафиксирован на {winner.get('stage')}). "
        f"LOSO mean AUC {format(float(metrics['mean_auc']), DISPLAY_METRIC_FORMAT)}, "
        f"худший фолд {metrics.get('worst_fold_id', '')} "
        f"{format(float(metrics['worst_auc']), DISPLAY_METRIC_FORMAT)}. "
        f"Порог прироста {format(float(MIN_MEAN_AUC_GAIN), DISPLAY_THRESHOLD_FORMAT)}, "
        f"сильная полоса {format(float(STRONG_MEAN_AUC_GAIN), DISPLAY_THRESHOLD_FORMAT)}. "
        f"Решение только по LOSO. Файл: {decisions_path()}."
    )


def rewind_to(stage: int) -> dict:
    """Сбросить решения начиная с stage и вернуть предыдущего победителя.

    Только если этап нужно прогнать заново с нуля.
    """
    payload = ensure_baseline()
    stages = payload["stages"]
    for key in list(stages):
        if int(key) >= int(stage):
            del stages[key]
    winner = {
        "name": payload["baseline"]["name"],
        "stage": "baseline",
        "source": payload["baseline"]["source"],
        "hparams": payload["baseline"]["hparams"],
        "metrics": payload["baseline"]["metrics"],
    }
    for key in sorted(stages, key=int):
        record = stages[key]
        if record.get("status") == STATUS_COMPLETE and record.get("accepted_new") and record.get("winner_config"):
            winner = record["winner_config"]
    payload["current_winner"] = winner
    save_decisions(payload)
    return payload
