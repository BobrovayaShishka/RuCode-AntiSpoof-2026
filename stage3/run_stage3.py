"""Этап 3: CNN по сырой волне. Запускается после появления метрик этапа 2."""

from __future__ import annotations

import json
import sys

import torch
from torch import nn

import af_common
from jsonutil import wait_until_ready, write_json
from stage2.bounds import (
    BATCH_SIZE_CHOICES,
    EARLY_STOP_PATIENCE,
    EXCLUDED_SHORTCUT_FEATURES,
    LR_LOW,
    OOM_BATCH_SIZE_FLOOR,
    PROTOCOL_POLL_S,
    TARGET_LENGTH,
)
from stage2.datautil import (
    extract_labels,
    make_store,
    open_training_table,
    validate_folds,
)
from stage2.driver import (
    configure_cudnn,
    hold_gpu,
    is_oom,
    run_study_and_seeds,
    study_slug,
)
from stage2.report import (
    SEED_PROTOCOL_NOTE,
    aggregate_seeds,
    compare_profiles,
    extract_scores,
    mean_fold_profile,
)
from stage3.bounds import RAW_SEARCH_BOUNDS, STUDY_CODE
from stage3.frontend import build_raw_cnn, frontend_report


def _fill_stage1_if_ready() -> None:
    if not af_common.STAGE2_METRICS_PATH.is_file() or not af_common.STAGE1_METRICS_PATH.is_file():
        return
    metrics = json.loads(af_common.STAGE2_METRICS_PATH.read_text(encoding="utf-8"))
    if metrics.get("kept_vs_stage1") is not None:
        return
    previous = extract_scores(json.loads(af_common.STAGE1_METRICS_PATH.read_text(encoding="utf-8")))
    candidate = extract_scores(metrics)
    if previous is None or candidate is None:
        return
    metrics["kept_vs_stage1"] = bool(af_common.accept_stage(previous, candidate))
    metrics["stage1_metrics_present"] = True
    write_json(af_common.STAGE2_METRICS_PATH, metrics)


def _probe_batch(report_fit: dict, batch_state: dict, cudnn_log: dict) -> None:
    params = {
        "width": max(RAW_SEARCH_BOUNDS["width"]),
        "kernel": max(RAW_SEARCH_BOUNDS["kernel"]),
        "stride": min(RAW_SEARCH_BOUNDS["stride"]),
        "n_blocks": max(RAW_SEARCH_BOUNDS["n_blocks"]),
        "dropout": 0.0,
        "lr": LR_LOW,
    }
    sizes = sorted(set(list(BATCH_SIZE_CHOICES) + [OOM_BATCH_SIZE_FLOOR]), reverse=True)
    frontend_report()
    with hold_gpu():
        configure_cudnn(cudnn_log)
        device = torch.device("cuda")
        cudnn_log["gpu"] = torch.cuda.get_device_name(0)
        chosen = None
        for batch in sizes:
            model = None
            optimizer = None
            wave = None
            target = None
            try:
                model = build_raw_cnn(params).to(device)
                optimizer = torch.optim.AdamW(model.parameters(), lr=LR_LOW)
                wave = torch.randn(int(batch), TARGET_LENGTH, device=device)
                target = torch.zeros(int(batch), device=device)
                with torch.amp.autocast("cuda"):
                    loss = nn.BCEWithLogitsLoss()(model(wave), target)
                loss.backward()
                optimizer.step()
                chosen = int(batch)
                break
            except Exception as exc:
                if not is_oom(exc):
                    raise
                torch.cuda.empty_cache()
            finally:
                if model is not None:
                    model.cpu()
                del model, optimizer, wave, target
                torch.cuda.empty_cache()
    if chosen is None:
        raise RuntimeError("raw CNN does not fit even at the named batch floor")
    report_fit["max_batch"] = chosen
    if chosen < max(BATCH_SIZE_CHOICES):
        batch_state["cap"] = chosen
        batch_state["events"].append({"probe_cap": chosen})


def suggest_stage3(trial) -> dict:
    return {
        "lr": trial.suggest_float("lr", RAW_SEARCH_BOUNDS["lr"][0], RAW_SEARCH_BOUNDS["lr"][1], log=True),
        "weight_decay": trial.suggest_float(
            "weight_decay", RAW_SEARCH_BOUNDS["weight_decay"][0], RAW_SEARCH_BOUNDS["weight_decay"][1], log=True
        ),
        "dropout": trial.suggest_float("dropout", RAW_SEARCH_BOUNDS["dropout"][0], RAW_SEARCH_BOUNDS["dropout"][1]),
        "batch_size": trial.suggest_categorical("batch_size", list(RAW_SEARCH_BOUNDS["batch_size"])),
        "max_epochs": trial.suggest_int(
            "max_epochs", RAW_SEARCH_BOUNDS["max_epochs"][0], RAW_SEARCH_BOUNDS["max_epochs"][1]
        ),
        "width": trial.suggest_categorical("width", list(RAW_SEARCH_BOUNDS["width"])),
        "kernel": trial.suggest_categorical("kernel", list(RAW_SEARCH_BOUNDS["kernel"])),
        "stride": trial.suggest_categorical("stride", list(RAW_SEARCH_BOUNDS["stride"])),
        "n_blocks": trial.suggest_categorical("n_blocks", list(RAW_SEARCH_BOUNDS["n_blocks"])),
    }


def main() -> int:
    print("stage 3 starting", flush=True)
    if not af_common.STAGE2_METRICS_PATH.is_file():
        raise SystemExit("stage 2 metrics are required before stage 3")
    stage2 = json.loads(af_common.STAGE2_METRICS_PATH.read_text(encoding="utf-8"))
    if stage2.get("status") != "complete":
        raise SystemExit("stage 2 did not finish")
    _fill_stage1_if_ready()
    if af_common.STAGE3_METRICS_PATH.is_file():
        existing = json.loads(af_common.STAGE3_METRICS_PATH.read_text(encoding="utf-8"))
        if existing.get("status") == "complete":
            print("stage 3 metrics already complete", flush=True)
            return 0

    verdict = wait_until_ready(af_common.STAGE0_VERDICT_PATH, PROTOCOL_POLL_S)
    table, source = open_training_table(verdict)
    labels = extract_labels(table)
    folds = list(af_common.iter_loso_folds(table))
    validate_folds(folds, len(table))
    store = make_store(table)
    store.prepare(af_common.STAGE3_DIR)
    print(f"stage3 rows={len(table)} folds={len(folds)} audio={store.source}", flush=True)

    batch_state = {"cap": max(BATCH_SIZE_CHOICES), "events": []}
    cudnn_log: dict = {}
    family_fit: dict = {}
    _probe_batch(family_fit, batch_state, cudnn_log)
    report = frontend_report()

    def materialize(params: dict):
        def model_fn():
            return build_raw_cnn(params)

        def dataset_fn(indices):
            from stage2.datautil import WaveDataset

            return WaveDataset(store, labels, indices)

        return model_fn, dataset_fn

    trained = run_study_and_seeds(
        study_name=study_slug(STUDY_CODE, {"bounds": RAW_SEARCH_BOUNDS, "frontend": report["source"]}),
        storage_path=af_common.STAGE3_DIR / "optuna.db",
        suggest=suggest_stage3,
        materialize=materialize,
        folds=folds,
        labels=labels,
        batch_state=batch_state,
        cudnn_log=cudnn_log,
        heartbeat=af_common.STAGE3_DIR / "heartbeat.txt",
        partial_path=af_common.STAGE3_DIR / "partial_seeds.json",
        log_prefix="stage3",
    )
    aggregate = aggregate_seeds(trained["per_seed"])
    stage2_scores = extract_scores(stage2)
    if stage2_scores is None:
        raise RuntimeError("stage 2 metrics have no mean_auc/worst_auc")
    kept = bool(af_common.accept_stage(stage2_scores, aggregate))
    profile = compare_profiles(mean_fold_profile(stage2["per_seed"]), mean_fold_profile(trained["per_seed"]))

    limitations = [
        "Shortcut scalars are not model inputs.",
        "No submission CSV; this stage is not treated as the final model.",
        report["note"],
        f"Audio source: {store.source}.",
    ]
    if cudnn_log.get("exception"):
        limitations.append(f"cuDNN deterministic setting raised: {cudnn_log['exception']}")
    if any(event.get("to") == OOM_BATCH_SIZE_FLOOR or event.get("probe_cap") == OOM_BATCH_SIZE_FLOOR for event in batch_state["events"]):
        limitations.append(f"Batch-size bound reached the named floor {OOM_BATCH_SIZE_FLOOR}.")

    payload = {
        "stage": 3,
        "status": "complete",
        "architecture": "raw_waveform_cnn",
        "raw_frontend": report,
        "pretrained": False,
        "best_params": trained["best_params"],
        "search_bounds": RAW_SEARCH_BOUNDS,
        "early_stop_patience": EARLY_STOP_PATIENCE,
        "seed_protocol": {
            "optuna_seed": int(af_common.SEEDS[0]),
            "final_seeds": [int(seed) for seed in af_common.SEEDS],
            "n_optuna_trials": int(af_common.N_OPTUNA_TRIALS),
            "full_loso": True,
            "note": SEED_PROTOCOL_NOTE,
        },
        "optuna_best_value": trained["optuna_best_value"],
        "trials": trained["trials"],
        "per_seed": trained["per_seed"],
        "aggregate": aggregate,
        "kept_vs_stage2": kept,
        "fold_profile_vs_stage2": profile,
        "features_excluded": list(EXCLUDED_SHORTCUT_FEATURES),
        "oom": batch_state,
        "family_fit": family_fit,
        "cudnn": cudnn_log,
        "n_folds": len(folds),
        "fold_ids": [str(fold.fold_id) for fold in folds],
        "data_manifest": verdict.get("data_manifest"),
        "table_source": source,
        "n_rows": len(table),
        "python": sys.executable,
        "limitations": limitations,
    }
    write_json(af_common.STAGE3_METRICS_PATH, payload)
    print(
        f"stage3 mean={aggregate['mean_auc']:.4f} worst={aggregate['worst_auc']:.4f} "
        f"kept_vs_stage2={kept} profile_differs={profile['profile_differs']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
