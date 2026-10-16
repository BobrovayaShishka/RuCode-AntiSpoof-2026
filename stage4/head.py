"""Этап 4: линейная голова на кэшированном слое, выбранном пробником."""

from __future__ import annotations

import argparse

from stage4.gpu_env import configure_process_env

configure_process_env()

import numpy as np
import optuna
import torch
from sklearn.preprocessing import StandardScaler

from af_common import (
    LABEL_COLUMN,
    N_OPTUNA_TRIALS,
    SEEDS,
    binary_auc,
    gpu_lock,
    iter_loso_folds,
    set_global_seed,
    summarize_fold_aucs,
)
from jsonutil import read_json, write_json
from stage4.cache_io import cache_dir_from_pointer, cache_is_complete, read_layer
from stage4.constants import (
    HEAD_BATCH_SIZES,
    HEAD_EPOCHS_HIGH,
    HEAD_EPOCHS_LOW,
    HEAD_LR_HIGH,
    HEAD_LR_LOW,
    HEAD_OPTIMIZER,
    HEAD_OPTUNA_PATH,
    HEAD_POS_WEIGHT_HIGH,
    HEAD_POS_WEIGHT_LOW,
    HEAD_RESULT_PATH,
    HEAD_WEIGHT_DECAY_HIGH,
    HEAD_WEIGHT_DECAY_LOW,
    LAYER_PROBE_PATH,
)
from stage4.data import load_training_table, table_fingerprint
from stage4.summaries import summarize_seeds


def fit_linear(features, labels, train_index, params: dict, seed: int, device: torch.device):
    set_global_seed(seed)
    scaler = StandardScaler()
    x_train = scaler.fit(features[train_index], labels[train_index]).transform(features[train_index])
    x_train = np.asarray(x_train, dtype=np.float32)
    y_train = labels[train_index].astype(np.float32)
    model = torch.nn.Linear(x_train.shape[1], 1).to(device)
    if HEAD_OPTIMIZER != "adamw":
        raise ValueError(HEAD_OPTIMIZER)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(params["lr"]),
        weight_decay=float(params["weight_decay"]),
    )
    pos_weight = torch.tensor([float(params["pos_weight"])], dtype=torch.float32, device=device)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    x_tensor = torch.from_numpy(x_train).to(device)
    y_tensor = torch.from_numpy(y_train).to(device)
    batch_size = int(params["batch_size"])
    epochs = int(params["epochs"])
    count = x_tensor.shape[0]
    model.train()
    for _epoch in range(epochs):
        order = torch.randperm(count)
        for start in range(0, count, batch_size):
            batch = order[start : start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            logits = model(x_tensor[batch]).reshape(-1)
            loss = loss_fn(logits, y_tensor[batch])
            loss.backward()
            optimizer.step()
    model.eval()
    return model, scaler


@torch.inference_mode()
def predict_linear(model, scaler, features, device: torch.device) -> np.ndarray:
    model.eval()
    transformed = np.asarray(scaler.transform(features), dtype=np.float32)
    logits = model(torch.from_numpy(transformed).to(device)).reshape(-1)
    scores = torch.sigmoid(logits).float().cpu().numpy()
    if not np.isfinite(scores).all():
        raise RuntimeError("non-finite linear-head scores")
    return scores


def _suggest(trial: optuna.Trial) -> dict:
    return {
        "lr": trial.suggest_float("lr", HEAD_LR_LOW, HEAD_LR_HIGH, log=True),
        "weight_decay": trial.suggest_float(
            "weight_decay", HEAD_WEIGHT_DECAY_LOW, HEAD_WEIGHT_DECAY_HIGH, log=True
        ),
        "epochs": trial.suggest_int("epochs", HEAD_EPOCHS_LOW, HEAD_EPOCHS_HIGH),
        "batch_size": trial.suggest_categorical("batch_size", list(HEAD_BATCH_SIZES)),
        "pos_weight": trial.suggest_float(
            "pos_weight", HEAD_POS_WEIGHT_LOW, HEAD_POS_WEIGHT_HIGH, log=True
        ),
    }


def _loso(features, labels, folds, params: dict, seed: int, device: torch.device) -> dict[str, float]:
    fold_aucs = {}
    for fold in folds:
        model, scaler = fit_linear(features, labels, fold.train_index, params, seed, device)
        scores = predict_linear(model, scaler, features[fold.val_index], device)
        fold_aucs[fold.fold_id] = binary_auc(labels[fold.val_index], scores)
        del model
    return fold_aucs


def _chosen_features(table_fingerprint_value: str):
    if not LAYER_PROBE_PATH.is_file():
        raise FileNotFoundError(LAYER_PROBE_PATH)
    probe = read_json(LAYER_PROBE_PATH)
    if probe.get("fingerprint") != table_fingerprint_value:
        raise RuntimeError("layer probe fingerprint does not match the training manifest")
    chosen = probe["chosen"]
    checkpoint = chosen["checkpoint"]
    layer = int(chosen["layer"])
    found = cache_dir_from_pointer(checkpoint, table_fingerprint_value)
    if found is None:
        raise FileNotFoundError(checkpoint)
    directory, meta = found
    if not cache_is_complete(meta):
        raise RuntimeError("feature cache is incomplete")
    features = read_layer(directory, layer, int(meta["n_rows"]), int(meta["hidden"]))
    return probe, features


def run_head(manifest: str) -> None:
    table = load_training_table(manifest)
    fingerprint = table_fingerprint(table)
    if HEAD_RESULT_PATH.is_file():
        existing = read_json(HEAD_RESULT_PATH)
        if existing.get("fingerprint") == fingerprint and existing.get("aggregate"):
            print("linear head result already complete", flush=True)
            return

    probe, features = _chosen_features(fingerprint)
    labels = table[LABEL_COLUMN].to_numpy(dtype=np.int64)
    folds = iter_loso_folds(table)
    chosen = probe["chosen"]
    optuna_state = None
    if HEAD_OPTUNA_PATH.is_file():
        previous = read_json(HEAD_OPTUNA_PATH)
        if previous.get("fingerprint") == fingerprint and int(previous.get("layer", -1)) == int(chosen["layer"]):
            optuna_state = previous

    with gpu_lock():
        if not torch.cuda.is_available():
            raise SystemExit("CUDA is required to train the linear head")
        device = torch.device("cuda")
        print(f"gpu lock held for linear head on {torch.cuda.get_device_name(0)}", flush=True)
        if optuna_state is None:
            def objective(trial: optuna.Trial) -> float:
                params = _suggest(trial)
                fold_aucs = _loso(features, labels, folds, params, SEEDS[0], device)
                value = float(summarize_fold_aucs(fold_aucs)["mean_auc"])
                print(f"head trial {trial.number} mean_auc={value:.4f} params={params}", flush=True)
                return value

            sampler = optuna.samplers.TPESampler(seed=SEEDS[0])
            study = optuna.create_study(direction="maximize", sampler=sampler)
            study.optimize(objective, n_trials=N_OPTUNA_TRIALS, show_progress_bar=False)
            optuna_state = {
                "fingerprint": fingerprint,
                "checkpoint": chosen["checkpoint"],
                "layer": int(chosen["layer"]),
                "best_params": study.best_params,
                "best_value": float(study.best_value),
                "search_seed": SEEDS[0],
                "n_trials": N_OPTUNA_TRIALS,
            }
            write_json(HEAD_OPTUNA_PATH, optuna_state)

        params = optuna_state["best_params"]
        by_seed = {}
        for seed in SEEDS:
            print(f"retrain seed {seed}", flush=True)
            by_seed[seed] = _loso(features, labels, folds, params, seed, device)
        summary = summarize_seeds(by_seed)
        torch.cuda.empty_cache()

    payload = {
        "fingerprint": fingerprint,
        "manifest": manifest,
        "checkpoint": chosen["checkpoint"],
        "layer": int(chosen["layer"]),
        "best_params": params,
        "optuna_best_value": optuna_state.get("best_value"),
        "search_seed": SEEDS[0],
        "n_trials": N_OPTUNA_TRIALS,
        "per_seed": summary["per_seed"],
        "aggregate": summary["aggregate"],
        "seed_mean_aucs": summary["seed_mean_aucs"],
        "seed_std_of_mean_auc": summary["seed_std_of_mean_auc"],
        "probe_chosen": chosen,
    }
    write_json(HEAD_RESULT_PATH, payload)
    aggregate = summary["aggregate"]
    print(
        f"head mean={aggregate['mean_auc']:.4f} worst={aggregate['worst_auc']:.4f} "
        f"std={aggregate['std_auc']:.4f} ci=[{aggregate['ci_low']:.4f}, {aggregate['ci_high']:.4f}]",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()
    run_head(args.manifest)


if __name__ == "__main__":
    main()
