"""Этап 2: LFCC или log-mel в LCNN либо ResNet-18 без предобучения."""

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
    FAMILY_CHOICES,
    FAMILY_LCNN,
    FAMILY_RESNET18,
    FRONTEND_LOGMEL,
    HOP_CHOICES,
    LCNN_WIDTH_CHOICES,
    LR_LOW,
    N_BINS_CHOICES,
    N_FFT_CHOICES,
    OOM_BATCH_SIZE_FLOOR,
    PROTOCOL_POLL_S,
    RESNET_WIDTH_CHOICES,
    SEARCH_BOUNDS,
    STAGE1_POLL_S,
    STAGE1_WAIT_S,
    STUDY_CODE,
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
from stage2.features import SpecFrontend, ensure_spec_memmap
from stage2.models import build_spectrogram_model, spectrogram_factories
from stage2.report import SEED_PROTOCOL_NOTE, aggregate_seeds, extract_scores


def _worst_spec_shape() -> tuple[int, int, int]:
    frontend = SpecFrontend(FRONTEND_LOGMEL, max(N_BINS_CHOICES), max(N_FFT_CHOICES), min(HOP_CHOICES)).eval()
    with torch.no_grad():
        spec = frontend(torch.zeros(1, TARGET_LENGTH))
    return (1, int(spec.shape[-2]), int(spec.shape[-1]))


def _one_step(family: str, shape: tuple[int, int, int], batch: int, device: torch.device) -> None:
    model = build_spectrogram_model(
        family,
        max(LCNN_WIDTH_CHOICES),
        max(RESNET_WIDTH_CHOICES),
        dropout=0.0,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR_LOW)
    inputs = torch.randn(batch, *shape, device=device)
    target = torch.zeros(batch, device=device)
    try:
        with torch.amp.autocast("cuda"):
            loss = nn.BCEWithLogitsLoss()(model(inputs), target)
        loss.backward()
        optimizer.step()
    finally:
        model.cpu()
        del model, optimizer, inputs, target
        torch.cuda.empty_cache()


def _largest_fit(family: str, shape: tuple[int, int, int], device: torch.device) -> int | None:
    sizes = list(BATCH_SIZE_CHOICES) + [OOM_BATCH_SIZE_FLOOR]
    ordered = []
    for size in sizes:
        if size not in ordered:
            ordered.append(size)
    ordered.sort(reverse=True)
    for batch in ordered:
        try:
            _one_step(family, shape, int(batch), device)
            return int(batch)
        except Exception as exc:
            if not is_oom(exc):
                raise
            torch.cuda.empty_cache()
    return None


def probe_families(shape, batch_state: dict, cudnn_log: dict) -> tuple[list[str], dict]:
    report = {}
    families = []
    fit_batches = []
    with hold_gpu():
        configure_cudnn(cudnn_log)
        device = torch.device("cuda")
        cudnn_log["gpu"] = torch.cuda.get_device_name(0)
        for family in FAMILY_CHOICES:
            fit = _largest_fit(family, shape, device)
            report[family] = {"fits": fit is not None, "max_batch": fit}
            if fit is None:
                report[family]["reason"] = "OOM down to the named batch floor; family left out of the search"
            else:
                families.append(family)
                fit_batches.append(fit)
    if not families:
        raise RuntimeError("neither LCNN nor ResNet-18 fits in GPU memory")
    cap = min(fit_batches)
    if cap < max(BATCH_SIZE_CHOICES):
        batch_state["cap"] = cap
        batch_state["events"].append({"probe_cap": cap})
    if FAMILY_RESNET18 not in families:
        report["why_not_both"] = "ResNet-18 did not fit; Optuna searches LCNN only."
    return families, report


def suggest_stage2(trial, families: list[str]) -> dict:
    params = {
        "family": trial.suggest_categorical("family", list(families)),
        "frontend": trial.suggest_categorical("frontend", list(SEARCH_BOUNDS["frontend"])),
        "n_bins": trial.suggest_categorical("n_bins", list(SEARCH_BOUNDS["n_bins"])),
        "n_fft": trial.suggest_categorical("n_fft", list(SEARCH_BOUNDS["n_fft"])),
        "hop": trial.suggest_categorical("hop", list(SEARCH_BOUNDS["hop"])),
        "lr": trial.suggest_float("lr", SEARCH_BOUNDS["lr"][0], SEARCH_BOUNDS["lr"][1], log=True),
        "weight_decay": trial.suggest_float(
            "weight_decay", SEARCH_BOUNDS["weight_decay"][0], SEARCH_BOUNDS["weight_decay"][1], log=True
        ),
        "dropout": trial.suggest_float("dropout", SEARCH_BOUNDS["dropout"][0], SEARCH_BOUNDS["dropout"][1]),
        "batch_size": trial.suggest_categorical("batch_size", list(SEARCH_BOUNDS["batch_size"])),
        "max_epochs": trial.suggest_int("max_epochs", SEARCH_BOUNDS["max_epochs"][0], SEARCH_BOUNDS["max_epochs"][1]),
    }
    if FAMILY_LCNN in families:
        params["lcnn_width"] = trial.suggest_categorical("lcnn_width", list(LCNN_WIDTH_CHOICES))
    if FAMILY_RESNET18 in families:
        params["resnet_width"] = trial.suggest_categorical("resnet_width", list(RESNET_WIDTH_CHOICES))
    return params


def main() -> int:
    print("stage 2 starting", flush=True)
    if af_common.STAGE2_METRICS_PATH.is_file():
        existing = json.loads(af_common.STAGE2_METRICS_PATH.read_text(encoding="utf-8"))
        if existing.get("status") == "complete":
            print("stage 2 metrics already complete", flush=True)
            return 0

    verdict = wait_until_ready(af_common.STAGE0_VERDICT_PATH, PROTOCOL_POLL_S)
    table, source = open_training_table(verdict)
    print(f"stage 2 table {source} rows={len(table)}", flush=True)
    labels = extract_labels(table)
    folds = list(af_common.iter_loso_folds(table))
    validate_folds(folds, len(table))
    store = make_store(table)
    store.prepare(af_common.STAGE2_DIR)
    print(f"rows={len(table)} folds={len(folds)} audio={store.source} table={source}", flush=True)

    batch_state = {"cap": max(BATCH_SIZE_CHOICES), "events": []}
    cudnn_log: dict = {}
    shape = _worst_spec_shape()
    families, family_fit = probe_families(shape, batch_state, cudnn_log)

    cache_dir = af_common.STAGE2_DIR / "feat_cache"

    def materialize(params: dict):
        specs = ensure_spec_memmap(
            store,
            str(params["frontend"]),
            int(params["n_bins"]),
            int(params["n_fft"]),
            int(params["hop"]),
            cache_dir,
        )

        return spectrogram_factories(params, specs, labels)

    trained = run_study_and_seeds(
        study_name=study_slug(STUDY_CODE, {"families": families, "bounds": SEARCH_BOUNDS}),
        storage_path=af_common.STAGE2_DIR / "optuna.db",
        suggest=lambda trial: suggest_stage2(trial, families),
        materialize=materialize,
        folds=folds,
        labels=labels,
        batch_state=batch_state,
        cudnn_log=cudnn_log,
        heartbeat=af_common.STAGE2_DIR / "heartbeat.txt",
        partial_path=af_common.STAGE2_DIR / "partial_seeds.json",
        log_prefix="stage2",
    )
    aggregate = aggregate_seeds(trained["per_seed"])
    import time

    deadline = time.time() + STAGE1_WAIT_S
    kept = None
    stage1_present = False
    while True:
        if af_common.STAGE1_METRICS_PATH.is_file():
            previous = extract_scores(json.loads(af_common.STAGE1_METRICS_PATH.read_text(encoding="utf-8")))
            if previous is not None:
                stage1_present = True
                kept = bool(af_common.accept_stage(previous, aggregate))
                break
        if time.time() >= deadline:
            break
        time.sleep(STAGE1_POLL_S)

    limitations = [
        "Shortcut scalars (duration, silence ratio, loudness, sample rate) are not model inputs.",
        "Submissions were not written; later SSL stages may still be the accepted model.",
        f"Audio source: {store.source}.",
    ]
    if FAMILY_RESNET18 not in families:
        limitations.append(family_fit.get("why_not_both", "ResNet-18 was not searched."))
    if any(event.get("to") == OOM_BATCH_SIZE_FLOOR for event in batch_state["events"]):
        limitations.append(f"Batch-size bound was lowered to the named floor {OOM_BATCH_SIZE_FLOOR} after OOM.")
    if cudnn_log.get("exception"):
        limitations.append(f"cuDNN deterministic setting raised: {cudnn_log['exception']}")

    payload = {
        "stage": 2,
        "status": "complete",
        "architecture": trained["best_params"].get("family"),
        "pretrained": False,
        "best_params": trained["best_params"],
        "search_bounds": SEARCH_BOUNDS,
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
        "kept_vs_stage1": kept,
        "stage1_metrics_present": stage1_present,
        "features_excluded": list(EXCLUDED_SHORTCUT_FEATURES),
        "family_fit": family_fit,
        "oom": batch_state,
        "cudnn": cudnn_log,
        "n_folds": len(folds),
        "fold_ids": [str(fold.fold_id) for fold in folds],
        "data_manifest": verdict.get("data_manifest"),
        "table_source": source,
        "n_rows": len(table),
        "ready_for_training": True,
        "verdict_leak": verdict.get("leak"),
        "python": sys.executable,
        "limitations": limitations,
    }
    write_json(af_common.STAGE2_METRICS_PATH, payload)
    print(
        f"stage2 {payload['architecture']} mean={aggregate['mean_auc']:.4f} "
        f"worst={aggregate['worst_auc']:.4f} kept_vs_stage1={kept}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
