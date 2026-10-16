"""Агрегация семян через общую сводку AUC по фолдам. Вторая реализация метрик не нужна."""

from __future__ import annotations

import numpy as np

from af_common import mean_fold_aucs, summarize_fold_aucs


def layer_family(checkpoint: str) -> str:
    lowered = str(checkpoint).lower()
    if "wavlm" in lowered:
        return "wavlm"
    if "xls-r" in lowered or "xlsr" in lowered:
        return "xlsr"
    raise ValueError(checkpoint)


def layer_rank(row: dict) -> tuple:
    """Сначала выше среднее, затем худший фолд, затем WavLM, затем более ранний слой."""
    simple = 1 if layer_family(row["checkpoint"]) == "wavlm" else 0
    return (float(row["mean_auc"]), float(row["worst_auc"]), simple, -int(row["layer"]))


def summarize_seeds(fold_aucs_by_seed: dict[int, dict[str, float]]) -> dict:
    if not fold_aucs_by_seed:
        raise ValueError("no seeds")
    per_seed = {
        str(seed): summarize_fold_aucs(fold_aucs)
        for seed, fold_aucs in fold_aucs_by_seed.items()
    }
    aggregate = summarize_fold_aucs(mean_fold_aucs(fold_aucs_by_seed.values()))
    seed_means = [float(per_seed[str(seed)]["mean_auc"]) for seed in fold_aucs_by_seed]
    if len(seed_means) > 1:
        seed_std = float(np.std(np.asarray(seed_means, dtype=np.float64), ddof=1))
    else:
        seed_std = 0.0
    return {
        "per_seed": per_seed,
        "aggregate": aggregate,
        "seed_mean_aucs": seed_means,
        "seed_std_of_mean_auc": seed_std,
    }
