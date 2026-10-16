"""Этап 1: моменты MFCC/LFCC, логистическая регрессия, LightGBM.

Каждый классификатор получает N_OPTUNA_TRIALS на общем кепстральном кэше.
Файлы сдачи не пишутся. Длительность, тишина, громкость и частота дискретизации не признаки.
"""

from __future__ import annotations

import os
import time
from multiprocessing import freeze_support
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np

import af_common
from af_common import (
    STAGE0_CLASS_WEIGHT_NONE,
    STAGE0_CLASS_WEIGHT_POLICIES,
    class_weight_arg,
    mean_fold_aucs,
)
from jsonutil import to_jsonable, wait_until_ready, write_json
from stage1.features import (
    AGGREGATION_CHOICES,
    CEPSTRAL_SOURCES,
    HOP_LENGTH_CHOICES,
    MOMENT_GROUPS,
    N_FFT_CHOICES,
    N_LFCC_CHOICES,
    N_MELS_CHOICES,
    N_MFCC_CHOICES,
    assemble,
    cache_signature,
    extract_paths,
    load_cache,
    save_cache,
    suggest_frontend,
)

# Границы поиска Optuna. Solver, итерации и потоки LightGBM фиксированы.
LOGREG_C_LOW = 1e-3
LOGREG_C_HIGH = 1e2
CLASS_WEIGHT_NONE = STAGE0_CLASS_WEIGHT_NONE
CLASS_WEIGHT_POLICIES = STAGE0_CLASS_WEIGHT_POLICIES
LOGREG_SOLVER = "lbfgs"
LOGREG_MAX_ITER = 1000
LOGREG_TOL = 1e-4

LGBM_N_ESTIMATORS_LOW = 80
LGBM_N_ESTIMATORS_HIGH = 300
LGBM_LEARNING_RATE_LOW = 0.02
LGBM_LEARNING_RATE_HIGH = 0.2
LGBM_NUM_LEAVES_LOW = 15
LGBM_NUM_LEAVES_HIGH = 63
LGBM_MIN_CHILD_SAMPLES_LOW = 20
LGBM_MIN_CHILD_SAMPLES_HIGH = 200
LGBM_SUBSAMPLE_LOW = 0.7
LGBM_SUBSAMPLE_HIGH = 1.0
LGBM_COLSAMPLE_BYTREE_LOW = 0.7
LGBM_COLSAMPLE_BYTREE_HIGH = 1.0
LGBM_REG_LAMBDA_LOW = 1e-3
LGBM_REG_LAMBDA_HIGH = 10.0
LIGHTGBM_OBJECTIVE = "binary"
LIGHTGBM_N_JOBS = -1
LIGHTGBM_SUBSAMPLE_FREQ = 1
LIGHTGBM_VERBOSITY = -1
LIGHTGBM_DETERMINISTIC = True
LIGHTGBM_FORCE_ROW_WISE = True

MODEL_NAMES = ("LogReg", "LightGBM")
LOGREG_PARAM_NAMES = ("C", "class_weight")
LIGHTGBM_PARAM_NAMES = (
    "n_estimators",
    "learning_rate",
    "num_leaves",
    "min_child_samples",
    "subsample",
    "colsample_bytree",
    "reg_lambda",
    "class_weight",
)
FORBIDDEN_FEATURES = ("duration", "silence_ratio", "loudness", "sample_rate")
WAIT_POLL_SECONDS = 15

CROSS_SEED_AGGREGATION = (
    "iter_loso_folds is called once and reused for every trial and seed; "
    "fold identity does not depend on the seed. "
    "For each seed, summarize_fold_aucs is called on that seed's fold-id to AUC map. "
    "Across seeds, each fold id's AUC is the arithmetic mean of the per-seed AUCs. "
    "winner_summary is summarize_fold_aucs on that averaged map. "
    "The Optuna objective is winner_summary mean_auc, equal to the mean over seeds of mean LOSO AUC."
)


def wait_for_training_verdict() -> dict:
    return wait_until_ready(af_common.STAGE0_VERDICT_PATH, WAIT_POLL_SECONDS)


def resolve_manifest(raw: str) -> Path:
    path = Path(raw)
    if not path.is_absolute():
        path = af_common.PROJECT_ROOT / path
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def resolve_audio(raw: str) -> str:
    candidate = Path(raw)
    if candidate.is_file():
        return str(candidate.resolve())
    rooted = af_common.PROJECT_ROOT / raw
    if rooted.is_file():
        return str(rooted.resolve())
    raise FileNotFoundError(raw)


def assert_no_shortcut_features(names: list[str]) -> None:
    banned = [name for name in names if name in FORBIDDEN_FEATURES]
    if banned:
        raise RuntimeError(f"shortcut features are not allowed: {banned}")


def effective_frontend(params: dict) -> dict:
    source = params["cepstral_source"]
    return {
        "n_fft": int(params["n_fft"]),
        "hop_length": int(params["hop_length"]),
        "n_mels": int(params["n_mels"]),
        "cepstral_source": source,
        "n_mfcc": None if source == "lfcc" else int(params["n_mfcc"]),
        "n_lfcc": None if source == "mfcc" else int(params["n_lfcc"]),
        "aggregation": params["aggregation"],
        "moments": list(MOMENT_GROUPS[params["aggregation"]]),
    }


def model_params(params: dict, model_name: str) -> dict:
    names = LOGREG_PARAM_NAMES if model_name == "LogReg" else LIGHTGBM_PARAM_NAMES
    return {name: params[name] for name in names}


def suggest_model(trial, model_name: str) -> dict:
    if model_name == "LogReg":
        return {
            "C": trial.suggest_float("C", LOGREG_C_LOW, LOGREG_C_HIGH, log=True),
            "class_weight": trial.suggest_categorical("class_weight", list(CLASS_WEIGHT_POLICIES)),
        }
    if model_name == "LightGBM":
        return {
            "n_estimators": trial.suggest_int(
                "n_estimators", LGBM_N_ESTIMATORS_LOW, LGBM_N_ESTIMATORS_HIGH
            ),
            "learning_rate": trial.suggest_float(
                "learning_rate", LGBM_LEARNING_RATE_LOW, LGBM_LEARNING_RATE_HIGH, log=True
            ),
            "num_leaves": trial.suggest_int("num_leaves", LGBM_NUM_LEAVES_LOW, LGBM_NUM_LEAVES_HIGH),
            "min_child_samples": trial.suggest_int(
                "min_child_samples", LGBM_MIN_CHILD_SAMPLES_LOW, LGBM_MIN_CHILD_SAMPLES_HIGH
            ),
            "subsample": trial.suggest_float("subsample", LGBM_SUBSAMPLE_LOW, LGBM_SUBSAMPLE_HIGH),
            "colsample_bytree": trial.suggest_float(
                "colsample_bytree", LGBM_COLSAMPLE_BYTREE_LOW, LGBM_COLSAMPLE_BYTREE_HIGH
            ),
            "reg_lambda": trial.suggest_float(
                "reg_lambda", LGBM_REG_LAMBDA_LOW, LGBM_REG_LAMBDA_HIGH, log=True
            ),
            "class_weight": trial.suggest_categorical("class_weight", list(CLASS_WEIGHT_POLICIES)),
        }
    raise ValueError(model_name)


def build_model(model_name: str, params: dict, seed: int):
    weight = class_weight_arg(params["class_weight"])
    if model_name == "LogReg":
        from sklearn.linear_model import LogisticRegression

        return LogisticRegression(
            C=float(params["C"]),
            solver=LOGREG_SOLVER,
            max_iter=LOGREG_MAX_ITER,
            tol=LOGREG_TOL,
            class_weight=weight,
            random_state=int(seed),
        )
    if model_name == "LightGBM":
        from lightgbm import LGBMClassifier

        return LGBMClassifier(
            objective=LIGHTGBM_OBJECTIVE,
            n_estimators=int(params["n_estimators"]),
            learning_rate=float(params["learning_rate"]),
            num_leaves=int(params["num_leaves"]),
            min_child_samples=int(params["min_child_samples"]),
            subsample=float(params["subsample"]),
            subsample_freq=LIGHTGBM_SUBSAMPLE_FREQ,
            colsample_bytree=float(params["colsample_bytree"]),
            reg_lambda=float(params["reg_lambda"]),
            class_weight=weight,
            random_state=int(seed),
            n_jobs=LIGHTGBM_N_JOBS,
            verbosity=LIGHTGBM_VERBOSITY,
            deterministic=LIGHTGBM_DETERMINISTIC,
            force_row_wise=LIGHTGBM_FORCE_ROW_WISE,
        )
    raise ValueError(model_name)


def spoof_scores(model, features: np.ndarray, label_spoof: int) -> np.ndarray:
    classes = [int(label) for label in model.classes_]
    if label_spoof not in classes:
        raise RuntimeError(f"model classes {classes} do not include spoof label {label_spoof}")
    scores = model.predict_proba(features)[:, classes.index(label_spoof)]
    return np.asarray(scores, dtype=np.float64)


def evaluate_seed(matrix: np.ndarray, labels: np.ndarray, folds, model_name: str, params: dict, seed: int) -> dict:
    af_common.set_global_seed(seed)
    fold_aucs: dict[str, float] = {}
    for fold in folds:
        train_index = fold.train_index
        val_index = fold.val_index
        x_train = matrix[train_index]
        x_val = matrix[val_index]
        y_train = labels[train_index]
        y_val = labels[val_index]
        if model_name == "LogReg":
            from sklearn.preprocessing import StandardScaler

            scaler = StandardScaler()
            x_train = scaler.fit_transform(np.asarray(x_train, dtype=np.float64))
            x_val = scaler.transform(np.asarray(x_val, dtype=np.float64))
        model = build_model(model_name, params, seed)
        model.fit(x_train, y_train)
        scores = spoof_scores(model, x_val, af_common.LABEL_SPOOF)
        fold_aucs[str(fold.fold_id)] = af_common.binary_auc(y_val, scores)
    summary = af_common.summarize_fold_aucs(fold_aucs)
    return {"seed": int(seed), "fold_aucs": fold_aucs, "summary": summary}


def average_fold_aucs(per_seed: list[dict]) -> dict[str, float]:
    return mean_fold_aucs(item["fold_aucs"] for item in per_seed)


def make_objective(model_name: str, cache: dict, labels: np.ndarray, folds):
    assembled: dict[tuple, np.ndarray] = {}

    def objective(trial) -> float:
        frontend = suggest_frontend(trial)
        params = suggest_model(trial, model_name)
        cache_key = (
            frontend["n_fft"],
            frontend["hop_length"],
            frontend["n_mels"],
            frontend["cepstral_source"],
            frontend["n_mfcc"],
            frontend["n_lfcc"],
            frontend["aggregation"],
        )
        if cache_key not in assembled:
            matrix, names = assemble(cache, frontend)
            assert_no_shortcut_features(names)
            if matrix.shape[0] != labels.shape[0]:
                raise RuntimeError("feature matrix rows do not match the LOSO table")
            assembled[cache_key] = matrix
        matrix = assembled[cache_key]
        per_seed = [
            evaluate_seed(matrix, labels, folds, model_name, params, int(seed))
            for seed in af_common.SEEDS
        ]
        averaged = average_fold_aucs(per_seed)
        summary = af_common.summarize_fold_aucs(averaged)
        seed_mean = float(np.mean([item["summary"]["mean_auc"] for item in per_seed]))
        if abs(seed_mean - float(summary["mean_auc"])) > float(af_common.AUC_MATCH_ATOL):
            raise RuntimeError("cross-seed mean does not match the mean of per-seed mean AUCs")
        trial.set_user_attr("summary", to_jsonable(summary))
        trial.set_user_attr("per_seed", to_jsonable(per_seed))
        print(
            f"{model_name} trial {trial.number} mean_auc={summary['mean_auc']:.4f} "
            f"worst_auc={summary['worst_auc']:.4f}",
            flush=True,
        )
        return float(summary["mean_auc"])

    return objective


def tune_model(model_name: str, cache: dict, labels: np.ndarray, folds) -> dict:
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=int(af_common.SEEDS[0]))
    study = optuna.create_study(direction="maximize", sampler=sampler)
    started = time.perf_counter()
    study.optimize(
        make_objective(model_name, cache, labels, folds),
        n_trials=int(af_common.N_OPTUNA_TRIALS),
        n_jobs=1,
        catch=(),
        show_progress_bar=False,
    )
    elapsed = time.perf_counter() - started
    best = study.best_trial
    frontend = effective_frontend(best.params)
    _matrix, names = assemble(cache, suggest_frontend_from_params(best.params))
    assert_no_shortcut_features(names)
    trials = []
    for trial in study.trials:
        if trial.value is None:
            continue
        summary = trial.user_attrs.get("summary", {})
        trials.append(
            {
                "number": int(trial.number),
                "mean_auc": float(trial.value),
                "worst_auc": summary.get("worst_auc"),
                "params": to_jsonable(trial.params),
            }
        )
    return {
        "model": model_name,
        "best_params": to_jsonable(best.params),
        "effective_frontend": frontend,
        "model_params": to_jsonable(model_params(best.params, model_name)),
        "feature_names": names,
        "n_features": len(names),
        "summary": best.user_attrs["summary"],
        "per_seed": best.user_attrs["per_seed"],
        "per_seed_mean_auc": {
            str(item["seed"]): item["summary"]["mean_auc"] for item in best.user_attrs["per_seed"]
        },
        "trials": trials,
        "n_trials": int(af_common.N_OPTUNA_TRIALS),
        "elapsed_seconds": elapsed,
    }


def suggest_frontend_from_params(params: dict) -> dict:
    return {
        "n_fft": int(params["n_fft"]),
        "hop_length": int(params["hop_length"]),
        "n_mels": int(params["n_mels"]),
        "cepstral_source": params["cepstral_source"],
        "n_mfcc": int(params["n_mfcc"]),
        "n_lfcc": int(params["n_lfcc"]),
        "aggregation": params["aggregation"],
    }


def search_bounds() -> dict:
    return {
        "n_fft": list(N_FFT_CHOICES),
        "hop_length": list(HOP_LENGTH_CHOICES),
        "n_mels": list(N_MELS_CHOICES),
        "n_mfcc": list(N_MFCC_CHOICES),
        "n_lfcc": list(N_LFCC_CHOICES),
        "cepstral_source": list(CEPSTRAL_SOURCES),
        "aggregation": list(AGGREGATION_CHOICES),
        "logreg_C": [LOGREG_C_LOW, LOGREG_C_HIGH],
        "class_weight": list(CLASS_WEIGHT_POLICIES),
        "logreg_solver": LOGREG_SOLVER,
        "logreg_max_iter": LOGREG_MAX_ITER,
        "lightgbm_n_estimators": [LGBM_N_ESTIMATORS_LOW, LGBM_N_ESTIMATORS_HIGH],
        "lightgbm_learning_rate": [LGBM_LEARNING_RATE_LOW, LGBM_LEARNING_RATE_HIGH],
        "lightgbm_num_leaves": [LGBM_NUM_LEAVES_LOW, LGBM_NUM_LEAVES_HIGH],
        "lightgbm_min_child_samples": [LGBM_MIN_CHILD_SAMPLES_LOW, LGBM_MIN_CHILD_SAMPLES_HIGH],
        "lightgbm_subsample": [LGBM_SUBSAMPLE_LOW, LGBM_SUBSAMPLE_HIGH],
        "lightgbm_colsample_bytree": [LGBM_COLSAMPLE_BYTREE_LOW, LGBM_COLSAMPLE_BYTREE_HIGH],
        "lightgbm_reg_lambda": [LGBM_REG_LAMBDA_LOW, LGBM_REG_LAMBDA_HIGH],
    }


def load_or_extract(paths: list[str]) -> dict:
    signature = cache_signature(paths)
    cached = load_cache(af_common.STAGE1_CACHE_PATH, signature, paths)
    if cached is not None:
        print(f"loaded feature cache {af_common.STAGE1_CACHE_PATH}", flush=True)
        return cached
    print(f"extracting cepstral moments for {len(paths)} files", flush=True)
    cache = extract_paths(paths)
    save_cache(cache, af_common.STAGE1_CACHE_PATH, signature)
    print(f"saved feature cache {af_common.STAGE1_CACHE_PATH}", flush=True)
    return cache


def main() -> None:
    verdict = wait_for_training_verdict()
    manifest = resolve_manifest(verdict["data_manifest"])
    table = af_common.load_loso_table(manifest)
    labels = table[af_common.LABEL_COLUMN].to_numpy(dtype=np.int64)
    paths = [resolve_audio(raw) for raw in table[af_common.PATH_COLUMN].tolist()]
    folds = af_common.iter_loso_folds(table)
    print(
        f"rows={len(table)} folds={[fold.fold_id for fold in folds]} "
        f"spoof={int(np.sum(labels == af_common.LABEL_SPOOF))} "
        f"bonafide={int(np.sum(labels != af_common.LABEL_SPOOF))}",
        flush=True,
    )
    cache = load_or_extract(paths)
    results = {model_name: tune_model(model_name, cache, labels, folds) for model_name in MODEL_NAMES}
    logreg = results["LogReg"]["summary"]
    lightgbm = results["LightGBM"]["summary"]
    lightgbm_accepted = bool(af_common.accept_stage(logreg, lightgbm))
    winner_name = "LightGBM" if lightgbm_accepted else "LogReg"
    winner = results[winner_name]
    dummy_previous = {"mean_auc": float("-inf"), "worst_auc": float("-inf")}
    kept_against_dummy = bool(af_common.accept_stage(dummy_previous, winner["summary"]))
    metrics = {
        "stage": 1,
        "winner": winner_name,
        "best_params": winner["best_params"],
        "effective_frontend": winner["effective_frontend"],
        "model_params": winner["model_params"],
        "feature_names": winner["feature_names"],
        "n_features": winner["n_features"],
        "aggregation": CROSS_SEED_AGGREGATION,
        "winner_summary": winner["summary"],
        "mean_auc": winner["summary"]["mean_auc"],
        "worst_auc": winner["summary"]["worst_auc"],
        "std_auc": winner["summary"]["std_auc"],
        "ci_low": winner["summary"]["ci_low"],
        "ci_high": winner["summary"]["ci_high"],
        "fold_aucs": winner["summary"]["fold_aucs"],
        "per_seed": winner["per_seed"],
        "per_seed_mean_auc": winner["per_seed_mean_auc"],
        "selection": {
            "rule": (
                "LogReg is the simpler incumbent. LightGBM replaces it only when "
                "accept_stage(LogReg summary, LightGBM summary) is true: mean AUC gain "
                "is at least MIN_MEAN_AUC_GAIN and worst_auc does not fall. "
                "A tie or an insufficient gain keeps LogReg. "
                "accept_stage against a dummy previous mean_auc=-inf records that this "
                "within-stage winner is kept as the reference baseline."
            ),
            "lightgbm_replaces_logreg": lightgbm_accepted,
            "mean_gain_lightgbm_minus_logreg": float(lightgbm["mean_auc"]) - float(logreg["mean_auc"]),
            "min_mean_auc_gain": float(af_common.MIN_MEAN_AUC_GAIN),
            "kept_against_dummy_previous": kept_against_dummy,
        },
        "models": results,
        "seeds": [int(seed) for seed in af_common.SEEDS],
        "n_optuna_trials_per_model": int(af_common.N_OPTUNA_TRIALS),
        "search_bounds": search_bounds(),
        "forbidden_features": list(FORBIDDEN_FEATURES),
        "data_manifest": verdict["data_manifest"],
        "data_manifest_resolved": str(manifest),
        "used_stage0_manifest": True,
        "ready_for_training": True,
        "leak": verdict.get("leak"),
        "frontend_sample_rate": int(cache["sample_rate"]),
        "n_rows": len(table),
    }
    write_json(af_common.STAGE1_METRICS_PATH, metrics)
    print(
        f"winner={winner_name} mean_auc={metrics['mean_auc']:.4f} "
        f"worst_auc={metrics['worst_auc']:.4f} wrote {af_common.STAGE1_METRICS_PATH}",
        flush=True,
    )


if __name__ == "__main__":
    freeze_support()
    main()
