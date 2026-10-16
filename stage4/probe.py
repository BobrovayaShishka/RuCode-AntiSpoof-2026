"""Этап 4b. Логистическая регрессия на кэшированных mean-pooled слоях. Без GPU."""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse

import numpy as np
import optuna
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from af_common import (
    LABEL_COLUMN,
    LABEL_SPOOF,
    MIN_MEAN_AUC_GAIN,
    N_OPTUNA_TRIALS,
    SEEDS,
    STAGE0_CLASS_WEIGHT_POLICIES,
    STAGE0_LOGREG_C_HIGH,
    STAGE0_LOGREG_C_LOW,
    STAGE0_LOGREG_MAX_ITER,
    STAGE0_LOGREG_TOL,
    binary_auc,
    class_weight_arg,
    iter_loso_folds,
    set_global_seed,
    summarize_fold_aucs,
)
from jsonutil import read_json, write_json
from stage4.cache_io import cache_dir_from_pointer, cache_is_complete, read_layer
from stage4.constants import (
    LAYER_PROBE_PATH,
    MODEL_CHECKPOINTS,
    PROBE_LOGREG_SOLVER,
    PROBE_REFERENCE,
    PROBE_STATE_PATH,
    SCALE_POOLED_FEATURES,
    WAVLM_CHECKPOINT,
)
from stage4.data import load_training_table, table_fingerprint
from stage4.summaries import layer_family, layer_rank, summarize_seeds


def _class_weight(name: str):
    return class_weight_arg(name)


def _fit_auc(features, labels, train_index, val_index, params: dict, seed: int) -> float:
    set_global_seed(seed)
    x_train = features[train_index]
    x_val = features[val_index]
    if SCALE_POOLED_FEATURES:
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_val = scaler.transform(x_val)
    model = LogisticRegression(
        C=float(params["C"]),
        class_weight=_class_weight(params["class_weight"]),
        solver=PROBE_LOGREG_SOLVER,
        max_iter=STAGE0_LOGREG_MAX_ITER,
        tol=STAGE0_LOGREG_TOL,
        random_state=int(seed),
    )
    model.fit(x_train, labels[train_index])
    spoof_at = int(np.flatnonzero(model.classes_ == LABEL_SPOOF)[0])
    scores = model.predict_proba(x_val)[:, spoof_at]
    return binary_auc(labels[val_index], scores)


def _loso_aucs(features, labels, folds, params: dict, seed: int) -> dict[str, float]:
    return {
        fold.fold_id: _fit_auc(features, labels, fold.train_index, fold.val_index, params, seed)
        for fold in folds
    }


def choose_layer(rows: list[dict]) -> tuple[dict, str]:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(layer_family(row["checkpoint"]), []).append(row)
    wavlm = max(grouped["wavlm"], key=layer_rank) if "wavlm" in grouped else None
    xlsr = max(grouped["xlsr"], key=layer_rank) if "xlsr" in grouped else None
    if wavlm is not None and xlsr is not None:
        within = wavlm["mean_auc"] >= xlsr["mean_auc"] - MIN_MEAN_AUC_GAIN
        worst_ok = wavlm["worst_auc"] >= xlsr["worst_auc"]
        if within and worst_ok:
            return wavlm, (
                "WavLM base kept: its mean AUC is within MIN_MEAN_AUC_GAIN of XLS-R "
                "and its worst fold is not worse"
            )
    chosen = max(rows, key=layer_rank)
    return chosen, "highest mean AUC, then worst fold, then simpler model, then earlier layer"


def _load_models(fingerprint: str) -> list[dict]:
    loaded = []
    for checkpoint in MODEL_CHECKPOINTS:
        found = cache_dir_from_pointer(checkpoint, fingerprint)
        if found is None:
            raise FileNotFoundError(f"no feature cache for {checkpoint}")
        directory, meta = found
        if not cache_is_complete(meta):
            raise RuntimeError(f"incomplete cache for {checkpoint}")
        loaded.append({"checkpoint": checkpoint, "directory": directory, "meta": meta})
    return loaded


def _reference(models: list[dict]) -> tuple[dict, int]:
    preferred = [item for item in models if item["checkpoint"] == WAVLM_CHECKPOINT]
    host = preferred[0] if preferred else models[0]
    layer = int(host["meta"]["n_layers"]) - 1
    return host, layer


def _evaluate_matrix(features, labels, folds, params: dict) -> dict:
    by_seed = {
        seed: _loso_aucs(features, labels, folds, params, seed) for seed in SEEDS
    }
    summary = summarize_seeds(by_seed)
    return summary


def run_probe(manifest: str) -> None:
    table = load_training_table(manifest)
    fingerprint = table_fingerprint(table)
    if LAYER_PROBE_PATH.is_file():
        existing = read_json(LAYER_PROBE_PATH)
        if existing.get("fingerprint") == fingerprint and existing.get("chosen"):
            print("layer probe already complete for this manifest", flush=True)
            return

    labels = table[LABEL_COLUMN].to_numpy(dtype=np.int64)
    folds = iter_loso_folds(table)
    models = _load_models(fingerprint)
    print(
        f"probe folds={[fold.fold_id for fold in folds]} rows={len(table)}",
        flush=True,
    )

    state = {"fingerprint": fingerprint, "layers": {}}
    if PROBE_STATE_PATH.is_file():
        previous = read_json(PROBE_STATE_PATH)
        if previous.get("fingerprint") == fingerprint:
            state = previous
    state.setdefault("layers", {})

    if "best_params" not in state:
        reference_model, reference_layer = _reference(models)
        print(
            f"optuna reference {reference_model['checkpoint']} layer {reference_layer}",
            flush=True,
        )
        features = read_layer(
            reference_model["directory"],
            reference_layer,
            int(reference_model["meta"]["n_rows"]),
            int(reference_model["meta"]["hidden"]),
        )

        def objective(trial: optuna.Trial) -> float:
            params = {
                "C": trial.suggest_float(
                    "C", STAGE0_LOGREG_C_LOW, STAGE0_LOGREG_C_HIGH, log=True
                ),
                "class_weight": trial.suggest_categorical(
                    "class_weight", list(STAGE0_CLASS_WEIGHT_POLICIES)
                ),
            }
            means = []
            for seed in SEEDS:
                fold_aucs = _loso_aucs(features, labels, folds, params, seed)
                means.append(summarize_fold_aucs(fold_aucs)["mean_auc"])
            value = float(np.mean(means))
            print(f"trial {trial.number} mean_auc={value:.4f} params={params}", flush=True)
            return value

        sampler = optuna.samplers.TPESampler(seed=SEEDS[0])
        study = optuna.create_study(direction="maximize", sampler=sampler)
        study.optimize(objective, n_trials=N_OPTUNA_TRIALS, show_progress_bar=False)
        state["best_params"] = {
            "C": float(study.best_params["C"]),
            "class_weight": study.best_params["class_weight"],
            "solver": PROBE_LOGREG_SOLVER,
        }
        state["optuna_best_value"] = float(study.best_value)
        state["reference"] = {
            "checkpoint": reference_model["checkpoint"],
            "layer": reference_layer,
            "which": PROBE_REFERENCE,
        }
        state["n_trials"] = N_OPTUNA_TRIALS
        write_json(PROBE_STATE_PATH, state)
        del features

    params = state["best_params"]
    rows = []
    model_payload = {}
    for item in models:
        checkpoint = item["checkpoint"]
        meta = item["meta"]
        n_layers = int(meta["n_layers"])
        layers_out = {}
        for layer in range(n_layers):
            key = f"{checkpoint}::{layer}"
            if key in state["layers"]:
                record = state["layers"][key]
            else:
                print(f"grid {checkpoint} layer {layer}/{n_layers - 1}", flush=True)
                features = read_layer(
                    item["directory"], layer, int(meta["n_rows"]), int(meta["hidden"])
                )
                summary = _evaluate_matrix(features, labels, folds, params)
                del features
                record = {
                    "checkpoint": checkpoint,
                    "layer": layer,
                    "per_seed": summary["per_seed"],
                    "mean_auc": summary["aggregate"]["mean_auc"],
                    "worst_auc": summary["aggregate"]["worst_auc"],
                    "std_auc": summary["aggregate"]["std_auc"],
                    "ci_low": summary["aggregate"]["ci_low"],
                    "ci_high": summary["aggregate"]["ci_high"],
                    "fold_aucs": summary["aggregate"]["fold_aucs"],
                    "seed_mean_aucs": summary["seed_mean_aucs"],
                    "seed_std_of_mean_auc": summary["seed_std_of_mean_auc"],
                }
                state["layers"][key] = record
                write_json(PROBE_STATE_PATH, state)
                print(
                    f"  mean={record['mean_auc']:.4f} worst={record['worst_auc']:.4f}",
                    flush=True,
                )
            layers_out[str(layer)] = record
            rows.append(record)
        model_payload[checkpoint] = {
            "n_layers": n_layers,
            "hidden": int(meta["hidden"]),
            "max_audio_seconds": meta.get("max_audio_seconds"),
            "n_cropped": meta.get("n_cropped", 0),
            "vram_fallback": meta.get("vram_fallback"),
            "layers": layers_out,
        }

    ranked = sorted(rows, key=layer_rank, reverse=True)
    print("top layers", flush=True)
    for row in ranked[:8]:
        print(
            f"  {row['checkpoint']} layer {row['layer']} "
            f"mean={row['mean_auc']:.4f} worst={row['worst_auc']:.4f}",
            flush=True,
        )
    chosen, reason = choose_layer(rows)
    payload = {
        "fingerprint": fingerprint,
        "manifest": manifest,
        "optuna": {
            "n_trials": state.get("n_trials", N_OPTUNA_TRIALS),
            "reference": state.get("reference"),
            "best_params": params,
            "best_value": state.get("optuna_best_value"),
            "solver": PROBE_LOGREG_SOLVER,
            "seeds": list(SEEDS),
        },
        "models": model_payload,
        "chosen": {
            "checkpoint": chosen["checkpoint"],
            "layer": int(chosen["layer"]),
            "mean_auc": chosen["mean_auc"],
            "worst_auc": chosen["worst_auc"],
            "std_auc": chosen["std_auc"],
            "ci_low": chosen["ci_low"],
            "ci_high": chosen["ci_high"],
            "reason": reason,
        },
    }
    write_json(LAYER_PROBE_PATH, payload)
    print(
        f"chosen {chosen['checkpoint']} layer {chosen['layer']} "
        f"mean={chosen['mean_auc']:.4f} worst={chosen['worst_auc']:.4f}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()
    run_probe(args.manifest)


if __name__ == "__main__":
    main()
