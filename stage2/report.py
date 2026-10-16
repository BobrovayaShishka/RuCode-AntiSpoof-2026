"""Сводки LOSO для этапов по спектрограмме и по сырой волне."""

from __future__ import annotations

import json
from pathlib import Path

import af_common

SEED_PROTOCOL_NOTE = (
    "Each Optuna trial uses SEEDS[0] and the full LOSO fold list shared with every "
    "other model. The trial budget is N_OPTUNA_TRIALS for this stage alone. After the "
    "best trial is chosen, that frozen config is retrained on every seed in SEEDS. "
    "The aggregate is summarize_fold_aucs on the seed-averaged fold AUCs: mean_auc is "
    "the mean of those folds (equal to the mean of the per-seed means), worst_auc is "
    "the worst seed-averaged fold, and std_auc plus the bootstrap CI are over the folds. "
    "No fold, trial, or seed is dropped. Early stopping uses EARLY_STOP_PATIENCE on the "
    "held-out fold AUC."
)

AGGREGATE_NOTE = (
    "Each fold AUC is the arithmetic mean across seeds. mean_auc, worst_auc, std_auc, "
    "ci_low, and ci_high are summarize_fold_aucs on that fold map. worst_auc is the worst "
    "seed-averaged fold, not the minimum of the per-seed worsts. mean_auc equals the mean "
    "of the per-seed mean AUCs. std and the bootstrap interval are over folds."
)

_SCORE_KEYS = ("aggregate", "summary", "overall", "loso")


def aggregate_seeds(per_seed: dict) -> dict:
    """Та же сводка по семенам, что у этапов 0, 1 и 4."""
    if not per_seed:
        raise ValueError("no seeds")
    summary = af_common.summarize_fold_aucs(
        af_common.mean_fold_aucs(item["fold_aucs"] for item in per_seed.values())
    )
    seed_means = {str(seed): float(item["mean_auc"]) for seed, item in per_seed.items()}
    return {
        "mean_auc": summary["mean_auc"],
        "worst_auc": summary["worst_auc"],
        "std_auc": summary["std_auc"],
        "ci_low": summary["ci_low"],
        "ci_high": summary["ci_high"],
        "fold_aucs": summary["fold_aucs"],
        "seed_mean_aucs": seed_means,
        "note": AGGREGATE_NOTE,
    }


def _score_pair(block) -> dict | None:
    if isinstance(block, dict) and "mean_auc" in block and "worst_auc" in block:
        return {"mean_auc": float(block["mean_auc"]), "worst_auc": float(block["worst_auc"])}
    return None


def extract_scores(metrics: dict) -> dict | None:
    """mean_auc и worst_auc из блока aggregate, иначе из самого payload."""
    if not isinstance(metrics, dict):
        return None
    for key in _SCORE_KEYS:
        found = _score_pair(metrics.get(key))
        if found is not None:
            return found
    return _score_pair(metrics)


def load_auc_block(path: Path) -> dict | None:
    path = Path(path)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return extract_scores(payload)


def mean_fold_profile(per_seed: dict) -> dict[str, float]:
    return af_common.mean_fold_aucs(summary["fold_aucs"] for summary in per_seed.values())


def compare_profiles(reference: dict[str, float], candidate: dict[str, float]) -> dict:
    better = []
    worse = []
    deltas = {}
    for key, ref_auc in reference.items():
        if key not in candidate:
            continue
        delta = float(candidate[key]) - float(ref_auc)
        deltas[key] = delta
        if delta > 0.0:
            better.append(key)
        elif delta < 0.0:
            worse.append(key)
    return {
        "reference_fold_mean_auc": reference,
        "candidate_fold_mean_auc": {key: candidate[key] for key in deltas},
        "deltas": deltas,
        "stage3_better_folds": better,
        "stage3_worse_folds": worse,
        "profile_differs": bool(better) and bool(worse),
    }
