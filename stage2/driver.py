"""Optuna по полному LOSO, затем повторное обучение на каждом семени. GPU берёт блокировку."""

from __future__ import annotations

import time
from pathlib import Path

import optuna
import torch

import af_common
from jsonutil import beat, write_json
from torchutil import is_oom, release_cuda
from stage2.bounds import (
    BATCH_SIZE_CHOICES,
    LOCK_RETRY_SLEEP_S,
    OOM_BATCH_SIZE_FLOOR,
    OPTUNA_N_JOBS,
)
from stage2.loop import run_folds


def lower_batch_bound(current: int, state: dict) -> int | None:
    named = [size for size in BATCH_SIZE_CHOICES if size < current]
    if named:
        new_size = max(named)
    elif current > OOM_BATCH_SIZE_FLOOR:
        new_size = OOM_BATCH_SIZE_FLOOR
    else:
        return None
    state["cap"] = min(int(state.get("cap", current)), int(new_size))
    state.setdefault("events", []).append({"from": int(current), "to": int(new_size)})
    return int(new_size)


def configure_cudnn(cudnn_log: dict) -> None:
    if cudnn_log.get("configured"):
        return
    cudnn_log["configured"] = True
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        cudnn_log["deterministic"] = True
    except Exception as exc:
        cudnn_log["deterministic"] = False
        cudnn_log["exception"] = repr(exc)
        cudnn_log["force_nondeterministic"] = True


def make_seed_fn(cudnn_log: dict):
    def seed_fn(seed: int) -> None:
        af_common.set_global_seed(int(seed))
        if cudnn_log.get("force_nondeterministic"):
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = False

    return seed_fn


class _GpuLock:
    def __init__(self) -> None:
        self._ctx = None

    def __enter__(self):
        while True:
            ctx = af_common.gpu_lock()
            try:
                ctx.__enter__()
            except TimeoutError:
                print("gpu lock timed out, waiting to retry", flush=True)
                time.sleep(LOCK_RETRY_SLEEP_S)
                continue
            self._ctx = ctx
            return self

    def __exit__(self, exc_type, exc, tb):
        ctx = self._ctx
        self._ctx = None
        if ctx is None:
            return False
        return ctx.__exit__(exc_type, exc, tb)


def hold_gpu():
    return _GpuLock()


def _release_cuda() -> None:
    release_cuda()


def _run_locked(cudnn_log, fn):
    with hold_gpu():
        try:
            configure_cudnn(cudnn_log)
            device = torch.device("cuda")
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is required")
            return fn(device)
        finally:
            _release_cuda()


def _fold_runner(cudnn_log, model_fn, dataset_fn, folds, labels, seed, params, batch, heartbeat, prefix):
    def _go(device):
        def on_fold(fold_id, auc, elapsed=None):
            seconds = "" if elapsed is None else f" {elapsed:.0f}s"
            beat(heartbeat, f"{prefix} seed {seed} fold {fold_id} auc {auc:.4f} batch {batch}{seconds}")

        return run_folds(
            model_fn,
            dataset_fn,
            folds,
            labels,
            seed=int(seed),
            lr=float(params["lr"]),
            weight_decay=float(params["weight_decay"]),
            max_epochs=int(params["max_epochs"]),
            batch_size=int(batch),
            device=device,
            auc_fn=af_common.binary_auc,
            seed_fn=make_seed_fn(cudnn_log),
            on_fold=on_fold,
        )

    fold_aucs = _run_locked(cudnn_log, _go)
    return af_common.summarize_fold_aucs(fold_aucs)


def _call_with_oom_and_cudnn(cudnn_log, batch_state, batch, heartbeat, prefix, fn):
    det_retried = False
    current = int(batch)
    while True:
        try:
            return fn(current), current
        except Exception as exc:
            _release_cuda()
            if is_oom(exc):
                new_batch = lower_batch_bound(current, batch_state)
                if new_batch is None:
                    raise
                current = new_batch
                beat(heartbeat, f"{prefix} OOM, lowered batch bound to {current}")
                continue
            deterministic = cudnn_log.get("deterministic") and "deterministic" in str(exc).lower()
            if deterministic and not det_retried:
                cudnn_log["force_nondeterministic"] = True
                cudnn_log["deterministic"] = False
                cudnn_log["exception"] = repr(exc)
                det_retried = True
                beat(heartbeat, f"{prefix} cudnn deterministic failed, recorded and disabled")
                continue
            raise


def run_study_and_seeds(
    *,
    study_name: str,
    storage_path: Path,
    suggest,
    materialize,
    folds,
    labels,
    batch_state: dict,
    cudnn_log: dict,
    heartbeat: Path,
    partial_path: Path,
    log_prefix: str,
) -> dict:
    storage_path.parent.mkdir(parents=True, exist_ok=True)
    storage = "sqlite:///" + storage_path.resolve().as_posix()
    sampler = optuna.samplers.TPESampler(seed=int(af_common.SEEDS[0]))
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",
        sampler=sampler,
        load_if_exists=True,
    )

    def objective(trial: optuna.Trial) -> float:
        params = suggest(trial)
        beat(heartbeat, f"{log_prefix} trial {trial.number} start {params}")
        model_fn, dataset_fn = materialize(params)
        start_batch = min(int(params["batch_size"]), int(batch_state["cap"]))

        def attempt(batch: int):
            summary = _fold_runner(
                cudnn_log,
                model_fn,
                dataset_fn,
                folds,
                labels,
                int(af_common.SEEDS[0]),
                params,
                batch,
                heartbeat,
                f"{log_prefix} trial {trial.number}",
            )
            trial.set_user_attr("effective_batch_size", int(batch))
            trial.set_user_attr("fold_aucs", summary["fold_aucs"])
            trial.set_user_attr("worst_auc", float(summary["worst_auc"]))
            beat(heartbeat, f"{log_prefix} trial {trial.number} mean_auc {summary['mean_auc']:.4f}")
            return float(summary["mean_auc"])

        value, _batch = _call_with_oom_and_cudnn(
            cudnn_log, batch_state, start_batch, heartbeat, f"{log_prefix} trial {trial.number}", attempt
        )
        return value

    complete = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
    remaining = int(af_common.N_OPTUNA_TRIALS) - len(complete)
    if remaining > 0:
        study.optimize(objective, n_trials=remaining, n_jobs=OPTUNA_N_JOBS, catch=())
    complete = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
    if len(complete) != int(af_common.N_OPTUNA_TRIALS):
        raise RuntimeError(f"{log_prefix} completed {len(complete)} trials, expected {af_common.N_OPTUNA_TRIALS}")

    best = study.best_trial
    params = dict(best.params)
    params["batch_size"] = int(best.user_attrs.get("effective_batch_size", params["batch_size"]))
    params["batch_size"] = min(int(params["batch_size"]), int(batch_state["cap"]))
    model_fn, dataset_fn = materialize(params)

    saved = {}
    if partial_path.is_file():
        loaded = json_load(partial_path)
        if loaded.get("params") == params:
            saved = dict(loaded.get("per_seed", {}))

    per_seed = dict(saved)
    for seed in af_common.SEEDS:
        key = str(int(seed))
        if key in per_seed:
            beat(heartbeat, f"{log_prefix} reuse final seed {key}")
            continue
        start_batch = min(int(params["batch_size"]), int(batch_state["cap"]))

        def attempt(batch: int, seed=seed):
            return _fold_runner(
                cudnn_log,
                model_fn,
                dataset_fn,
                folds,
                labels,
                int(seed),
                params,
                batch,
                heartbeat,
                f"{log_prefix} final",
            )

        summary, used_batch = _call_with_oom_and_cudnn(
            cudnn_log, batch_state, start_batch, heartbeat, f"{log_prefix} final seed {key}", attempt
        )
        params["batch_size"] = int(used_batch)
        per_seed[key] = summary
        write_json(partial_path, {"params": params, "per_seed": per_seed})

    trials = [
        {
            "number": trial.number,
            "state": trial.state.name,
            "value": None if trial.value is None else float(trial.value),
            "params": trial.params,
            "user_attrs": trial.user_attrs,
        }
        for trial in study.trials
    ]
    return {
        "best_params": params,
        "optuna_best_value": float(best.value),
        "per_seed": per_seed,
        "trials": trials,
        "batch_state": batch_state,
        "cudnn": cudnn_log,
    }


def json_load(path: Path) -> dict:
    import json

    return json.loads(path.read_text(encoding="utf-8"))


def study_slug(prefix: str, payload: dict) -> str:
    import hashlib
    import json

    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:10]
    return f"{prefix}_{digest}"
