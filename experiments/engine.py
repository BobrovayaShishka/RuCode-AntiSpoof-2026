"""Обучение LOSO, Optuna, приёмка и сдача для этапов 5-9.

Порядок ячеек: pip, опциональный Drive, prepare_session, preview_stage, run_stage.
Финальный фит учится на всех размеченных строках без early stopping.
Фолды LOSO сохраняют patience этапа 2. Решение: сводки stage2.report.aggregate_seeds.
"""

from __future__ import annotations

import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import af_common
from experiments.augment import WaveAugmentor
from experiments.config import (
    AM_MARGIN_HIGH,
    AM_MARGIN_LOW,
    AM_SCALE_HIGH,
    AM_SCALE_LOW,
    ATTENTION_HIDDEN_HIGH,
    ATTENTION_HIDDEN_LOW,
    ATTENTION_HIDDEN_STEP,
    BASELINE_BATCH_SIZE,
    BASELINE_HOP,
    BASELINE_LCNN_WIDTH,
    BASELINE_N_BINS,
    BASELINE_N_FFT,
    CHANNEL_DIM,
    CHECKPOINT_DIRNAME,
    CODEC_APPLY_P_HIGH,
    CODEC_APPLY_P_LOW,
    CODEC_CHOICES,
    CONTINUATION_CODE_VERSION,
    CONTINUATION_DIRNAME,
    CONV_IN_CHANNELS,
    DEVICE_INDEX,
    DISPLAY_INDEX_OFFSET,
    DISPLAY_METRIC_FORMAT,
    DISPLAY_SECONDS_FORMAT,
    DROP_LAST,
    EARLY_STOP_PATIENCE,
    FEAT_CACHE_DIRNAME,
    FIRST_STAGE,
    FLATTEN_ALL,
    FRONTEND_LFCC,
    GRAPH_HIDDEN_HIGH,
    GRAPH_HIDDEN_LOW,
    GRAPH_HIDDEN_STEP,
    GRAPH_LAYERS_HIGH,
    GRAPH_LAYERS_LOW,
    HEARTBEAT_TEMPLATE,
    LAST_STAGE,
    LOSS_AM,
    LOSS_OC,
    MIN_MEAN_AUC_GAIN,
    N_LABELED_ROWS,
    N_OPTUNA_TRIALS,
    NUM_WORKERS,
    OC_MARGIN_BONAFIDE_CAP,
    OC_MARGIN_GAP_HIGH,
    OC_MARGIN_GAP_LOW,
    OC_MARGIN_SPOOF_HIGH,
    OC_MARGIN_SPOOF_LOW,
    OC_SCALE_HIGH,
    OC_SCALE_LOW,
    OPTIMIZER_NAME,
    OPTUNA_SEED_INDEX,
    PIN_MEMORY_ON_CUDA,
    POOL_AASIST,
    POOL_ASP,
    RAWBOOST_APPLY_P_HIGH,
    RAWBOOST_APPLY_P_LOW,
    RAWBOOST_IMPULSE_RATE_HIGH,
    RAWBOOST_IMPULSE_RATE_LOW,
    RAWBOOST_N_TAPS_CHOICES,
    RAWBOOST_STRENGTH_HIGH,
    RAWBOOST_STRENGTH_LOW,
    RAWBOOST_VARIANTS,
    ROLE_NEW,
    ROLE_REFERENCE,
    SEEDS,
    SMOKE_BATCH,
    SMOKE_FRAMES,
    SPEC_COMPUTE_BATCH,
    SPEC_RANK_WITHOUT_CHANNEL,
    STAGE2_DIRNAME,
    STALE_STEP,
    START_INDEX,
    STATUS_COMPLETE,
    STATUS_RUNNING,
    TRIAL_BUDGET_NOTE,
    TWO,
)
from experiments.decisions import (
    choose_winner,
    config_name,
    ensure_baseline,
    preview_text,
    require_prior_stage,
    save_decisions,
)
from experiments.models import build_model
from jsonutil import beat, to_jsonable
from stage2.bounds import OPTUNA_N_JOBS
from stage2.driver import _run_locked, is_oom, lower_batch_bound
from stage2.loop import fit_epochs, positive_weight
from torchutil import release_cuda
from stage2.report import aggregate_seeds, mean_fold_profile

KIND_SPEC = "spec"
KIND_WAVE = "wave"


class RunContext:
    def __init__(self, labels, folds, specs, store, device: torch.device) -> None:
        self.labels = labels
        self.folds = folds
        self.specs = specs
        self.store = store
        self.device = device
        self.batch_cap = int(BASELINE_BATCH_SIZE)
        self.oom_state = {"cap": int(BASELINE_BATCH_SIZE), "events": []}
        self.frontend = None


def continuation_dir() -> Path:
    return Path(af_common.OUTPUT_DIR) / CONTINUATION_DIRNAME


def prepare_session(repo_root: str | None = None, data_dir: str | None = None, require_cuda: bool = True) -> torch.device:
    """Зафиксировать семя, направить аудио на data_dir в Colab и вернуть устройство CUDA."""
    root = Path(repo_root) if repo_root else af_common.PROJECT_ROOT
    if Path(root).resolve() != Path(af_common.PROJECT_ROOT).resolve():
        raise RuntimeError(f"imported af_common lives in {af_common.PROJECT_ROOT}, not {root}")
    if str(root) not in sys.path:
        sys.path.insert(START_INDEX, str(root))
    af_common.use_data_dir(data_dir)
    print(f"data {af_common.DATA_DIR}", flush=True)
    print(f"protocol {af_common.TRAIN_PROTOCOL_PATH}", flush=True)
    af_common.set_global_seed(int(SEEDS[OPTUNA_SEED_INDEX]))
    if not torch.cuda.is_available():
        if require_cuda:
            raise RuntimeError(
                "Stages 5-9 need a CUDA GPU. In Colab use Runtime → Change runtime type → GPU, then rerun from the top."
            )
        return torch.device("cpu")
    print(f"GPU {torch.cuda.get_device_name(DEVICE_INDEX)}", flush=True)
    return torch.device("cuda", DEVICE_INDEX)


def _emit(text: str) -> None:
    """Печать и на консолях не UTF-8 (Windows cp1252), и в Colab."""
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    safe = text.encode(encoding, errors="replace").decode(encoding, errors="replace")
    print(safe, flush=True)


def preview_stage(stage: int) -> str:
    text = preview_text(int(stage))
    _emit(text)
    return text


def run_stage(stage: int, device: torch.device | None = None) -> dict:
    """Запустить этап под блокировкой GPU. Завершённый этап не переобучается."""
    if int(stage) < FIRST_STAGE or int(stage) > LAST_STAGE:
        raise ValueError(f"stage must be {FIRST_STAGE}..{LAST_STAGE}")
    if device is not None and device.type != "cuda":
        raise RuntimeError("run_stage trains on CUDA")

    def _go(locked_device: torch.device) -> dict:
        return _run_stage_body(int(stage), locked_device)

    return _run_locked({}, _go)


def _run_stage_body(stage: int, device: torch.device) -> dict:
    payload = ensure_baseline()
    require_prior_stage(payload, stage)
    existing = payload["stages"].get(str(stage))
    if isinstance(existing, dict) and existing.get("status") == STATUS_COMPLETE:
        if existing.get("accepted_new") and not existing.get("submission", {}).get("written"):
            _finish_submission(payload, stage, existing, device)
        print(existing.get("summary_ru", "stage already complete"), flush=True)
        return existing

    winner = payload["current_winner"]
    reference_hparams = deepcopy(winner["hparams"])
    reference_name = str(winner["name"])
    record = existing if isinstance(existing, dict) else {}
    if record.get("code_version") != CONTINUATION_CODE_VERSION or record.get("reference_name") != reference_name:
        record = {
            "status": STATUS_RUNNING,
            "code_version": CONTINUATION_CODE_VERSION,
            "question": STAGE_QUESTIONS[stage],
            "reference_name": reference_name,
            "reference_recomputed": stage == FIRST_STAGE,
            "n_trials": int(N_OPTUNA_TRIALS),
            "trial_budget_note": TRIAL_BUDGET_NOTE,
            "candidates": [],
        }
    payload["stages"][str(stage)] = record
    save_decisions(payload)

    context = load_context(device)
    by_name = _index_candidates(record["candidates"])

    if reference_name not in by_name:
        if stage == FIRST_STAGE:
            print(f"stage {stage} remeasuring reference {reference_name}", flush=True)
            reference_metrics = evaluate_hparams(reference_hparams, context, stage, reference_name)
        else:
            reference_metrics = deepcopy(winner["metrics"])
            _require_metric_keys(reference_metrics)
        by_name[reference_name] = {
            "name": reference_name,
            "role": ROLE_REFERENCE,
            "n_trials": START_INDEX,
            "hparams": reference_hparams,
            "metrics": reference_metrics,
            "trials": None,
        }
        record["candidates"] = list(by_name.values())
        save_decisions(payload)
    reference_arm = by_name[reference_name]

    for arm_name, suggest in stage_arms(stage):
        if arm_name in by_name:
            continue
        print(f"stage {stage} tuning {arm_name} for {int(N_OPTUNA_TRIALS)} trials", flush=True)
        tuned, metrics, trials = tune_arm(reference_hparams, suggest, context, stage, arm_name)
        by_name[arm_name] = {
            "name": config_name(tuned),
            "arm": arm_name,
            "role": ROLE_NEW,
            "n_trials": int(N_OPTUNA_TRIALS),
            "hparams": tuned,
            "metrics": metrics,
            "trials": trials,
        }
        record["candidates"] = list(by_name.values())
        save_decisions(payload)

    new_arms = [item for item in record["candidates"] if item.get("role") == ROLE_NEW]
    best, reason = choose_winner(reference_arm, new_arms)
    accepted = best is not None
    if accepted:
        carried = {
            "name": best["name"],
            "stage": stage,
            "hparams": best["hparams"],
            "metrics": best["metrics"],
        }
        metrics = best["metrics"]
        winner_name = best["name"]
    else:
        carried = {
            "name": reference_name,
            "stage": winner.get("stage"),
            "hparams": reference_hparams,
            "metrics": reference_arm["metrics"],
        }
        metrics = reference_arm["metrics"]
        winner_name = reference_name
        if stage == FIRST_STAGE:
            payload["current_winner"]["metrics"] = reference_arm["metrics"]
            payload["current_winner"]["remeasured_at_stage"] = stage
    if accepted:
        payload["current_winner"] = carried
    record.update(
        {
            "status": STATUS_COMPLETE,
            "winner": winner_name,
            "accepted_new": accepted,
            "metrics": metrics,
            "reference_metrics": reference_arm["metrics"],
            "reason": reason,
            "summary_ru": _summary_ru(stage, reason, metrics),
            "hparams": carried["hparams"],
            "winner_config": carried,
            "submission": {
                "written": False,
                "paths": [],
                "reason": "pending final fit" if accepted else "rejected; previous submission kept",
            },
        }
    )
    payload["stages"][str(stage)] = record
    save_decisions(payload)
    if accepted:
        _finish_submission(payload, stage, record, device)
    _emit(record["summary_ru"])
    return record


def _finish_submission(payload: dict, stage: int, record: dict, device: torch.device) -> None:
    context = load_context(device)
    paths = write_submissions(record["winner_config"]["hparams"], context, stage)
    record["submission"] = {"written": True, "paths": paths, "reason": "accepted; refit on all labeled rows, no early stopping"}
    payload["stages"][str(stage)] = record
    save_decisions(payload)


def load_context(device: torch.device) -> RunContext:
    from stage2 import datautil
    from stage2.features import ensure_spec_memmap

    table = af_common.load_loso_table(af_common.TRAIN_PROTOCOL_PATH)
    labels = datautil.extract_labels(table)
    if int(labels.shape[START_INDEX]) != int(N_LABELED_ROWS):
        raise RuntimeError(f"expected {int(N_LABELED_ROWS)} labeled rows, found {int(labels.shape[START_INDEX])}")
    folds = af_common.iter_loso_folds(table)
    datautil.validate_folds(folds, len(labels))
    store = datautil.WaveStore(table)
    store.prepare(continuation_dir() / "waves")
    cache_dir = Path(af_common.OUTPUT_DIR) / STAGE2_DIRNAME / FEAT_CACHE_DIRNAME
    specs = ensure_spec_memmap(
        store,
        FRONTEND_LFCC,
        int(BASELINE_N_BINS),
        int(BASELINE_N_FFT),
        int(BASELINE_HOP),
        cache_dir,
    )
    context = RunContext(labels, folds, specs, store, device)
    # LFCC на CPU, чтобы признаки обучения совпали с кэшированными спектрограммами валидации.
    context.frontend = _make_frontend()
    print(f"rows={len(labels)} folds={len(folds)} audio={store.source}", flush=True)
    return context


def evaluate_hparams(hparams: dict, context: RunContext, stage: int, label: str) -> dict:
    per_seed = {}
    for seed in SEEDS:
        fold_aucs = run_loso(hparams, context, int(seed), stage, label)
        per_seed[int(seed)] = af_common.summarize_fold_aucs(fold_aucs)
    return decision_metrics(per_seed)


def tune_arm(base: dict, suggest, context: RunContext, stage: int, arm_name: str) -> tuple[dict, dict, list]:
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=int(SEEDS[OPTUNA_SEED_INDEX]))
    study = optuna.create_study(direction="maximize", sampler=sampler)

    def objective(trial: optuna.Trial) -> float:
        resolved = suggest(trial)
        trial.set_user_attr("resolved", resolved)
        hparams = merge_hparams(base, resolved)
        fold_aucs = run_loso(hparams, context, int(SEEDS[OPTUNA_SEED_INDEX]), stage, arm_name)
        summary = af_common.summarize_fold_aucs(fold_aucs)
        trial.set_user_attr("worst_auc", float(summary["worst_auc"]))
        trial.set_user_attr("fold_aucs", summary["fold_aucs"])
        beat(
            _heartbeat(stage),
            f"stage {stage} {arm_name} trial {trial.number} mean_auc {format(float(summary['mean_auc']), DISPLAY_METRIC_FORMAT)}",
        )
        return float(summary["mean_auc"])

    study.optimize(objective, n_trials=int(N_OPTUNA_TRIALS), n_jobs=int(OPTUNA_N_JOBS))
    incomplete = [trial for trial in study.trials if trial.state != optuna.trial.TrialState.COMPLETE]
    if incomplete or len(study.trials) != int(N_OPTUNA_TRIALS):
        raise RuntimeError(
            f"{arm_name} must complete exactly {int(N_OPTUNA_TRIALS)} trials, "
            f"got {len(study.trials)} with {len(incomplete)} incomplete"
        )
    resolved = study.best_trial.user_attrs["resolved"]
    hparams = merge_hparams(base, resolved)
    metrics = evaluate_hparams(hparams, context, stage, arm_name)
    trials = [
        {
            "number": int(trial.number),
            "value": float(trial.value),
            "params": trial.params,
            "resolved": trial.user_attrs.get("resolved"),
            "worst_auc": trial.user_attrs.get("worst_auc"),
            "fold_aucs": trial.user_attrs.get("fold_aucs"),
        }
        for trial in study.trials
    ]
    return hparams, metrics, trials


def run_loso(hparams: dict, context: RunContext, seed: int, stage: int, label: str) -> dict[str, float]:
    fold_aucs: dict[str, float] = {}
    for fold in context.folds:
        started = time.perf_counter()
        auc = _fit_fold_with_oom_retry(
            hparams,
            context,
            np.asarray(fold.train_index),
            np.asarray(fold.val_index),
            int(seed),
            early_stop=True,
        )
        elapsed = time.perf_counter() - started
        fold_aucs[str(fold.fold_id)] = float(auc)
        beat(
            _heartbeat(stage),
            f"stage {stage} {label} seed {int(seed)} fold {fold.fold_id} "
            f"auc {format(float(auc), DISPLAY_METRIC_FORMAT)} {format(elapsed, DISPLAY_SECONDS_FORMAT)}s",
        )
    return fold_aucs


def write_submissions(hparams: dict, context: RunContext, stage: int) -> list[str]:
    """Переобучить каждое семя на всех размеченных строках и записать CSV сдачи.

    Тест скорится после решения LOSO. Метки теста не читаются.
    """
    ids = af_common.read_submission_ids()
    test_specs = _test_specs(ids, context.device)
    train_index = np.arange(int(context.labels.shape[START_INDEX]), dtype=np.int64)
    written: list[str] = []
    checkpoint_dir = continuation_dir() / CHECKPOINT_DIRNAME / f"stage{int(stage)}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    use_amp = context.device.type == "cuda"
    for seed in SEEDS:
        state = _fit_fold_with_oom_retry(
            hparams,
            context,
            train_index,
            None,
            int(seed),
            early_stop=False,
        )
        checkpoint = checkpoint_dir / f"seed_{int(seed)}.pt"
        torch.save({"seed": int(seed), "hparams": hparams, "state_dict": state}, checkpoint)
        model = build_model(hparams).to(context.device)
        model.load_state_dict(state)
        scores = _score_tensor(model, test_specs, context.device, int(hparams["batch_size"]), use_amp)
        destination = af_common.submission_path(int(seed))
        af_common.write_submission_csv(destination, ids, scores)
        written.append(af_common.repo_relative(destination))
        model.cpu()
        del model
        print(f"wrote {destination}", flush=True)
    return written


def decision_metrics(per_seed: dict) -> dict:
    aggregate = aggregate_seeds(per_seed)
    profile = mean_fold_profile(per_seed)
    worst_id = min(profile, key=profile.get)
    worst = float(aggregate["worst_auc"])
    return {
        "mean_auc": float(aggregate["mean_auc"]),
        "worst_fold": worst,
        "worst_auc": worst,
        "std": float(aggregate["std_auc"]),
        "std_auc": float(aggregate["std_auc"]),
        "ci_low": float(aggregate["ci_low"]),
        "ci_high": float(aggregate["ci_high"]),
        "worst_fold_id": str(worst_id),
        "worst_fold_mean_auc": float(profile[worst_id]),
        "fold_aucs": {str(key): float(value) for key, value in profile.items()},
        "seed_mean_aucs": {str(key): float(value) for key, value in aggregate["seed_mean_aucs"].items()},
        "per_seed": {str(seed): to_jsonable(summary) for seed, summary in per_seed.items()},
    }


def merge_hparams(base: dict, resolved: dict) -> dict:
    merged = deepcopy(base)
    merged.update(resolved)
    if merged.get("pooling") != POOL_ASP:
        merged.pop("attention_hidden", None)
    if merged.get("pooling") != POOL_AASIST:
        merged.pop("graph_layers", None)
        merged.pop("graph_hidden", None)
    if merged.get("loss") != LOSS_AM:
        merged.pop("am_scale", None)
        merged.pop("am_margin", None)
    if merged.get("loss") != LOSS_OC:
        for key in ("oc_scale", "oc_margin_bonafide", "oc_margin_spoof", "oc_margin_gap"):
            merged.pop(key, None)
    return merged


def stage_arms(stage: int) -> list[tuple[str, object]]:
    if stage == FIRST_STAGE:
        return [(POOL_ASP, suggest_asp)]
    if stage == FIRST_STAGE + STALE_STEP:
        return [(POOL_AASIST, suggest_graph)]
    if stage == FIRST_STAGE + STALE_STEP + STALE_STEP:
        return [(LOSS_AM, suggest_am), (LOSS_OC, suggest_oc)]
    if stage == LAST_STAGE - STALE_STEP:
        return [("rawboost", suggest_rawboost)]
    if stage == LAST_STAGE:
        return [("codec", suggest_codec)]
    raise ValueError(f"no arms for stage {stage}")


def suggest_asp(trial) -> dict:
    hidden = trial.suggest_int(
        "attention_hidden",
        int(ATTENTION_HIDDEN_LOW),
        int(ATTENTION_HIDDEN_HIGH),
        step=int(ATTENTION_HIDDEN_STEP),
    )
    return {"pooling": POOL_ASP, "attention_hidden": int(hidden)}


def suggest_graph(trial) -> dict:
    return {
        "pooling": POOL_AASIST,
        "graph_layers": int(trial.suggest_int("graph_layers", int(GRAPH_LAYERS_LOW), int(GRAPH_LAYERS_HIGH))),
        "graph_hidden": int(
            trial.suggest_int(
                "graph_hidden",
                int(GRAPH_HIDDEN_LOW),
                int(GRAPH_HIDDEN_HIGH),
                step=int(GRAPH_HIDDEN_STEP),
            )
        ),
    }


def suggest_am(trial) -> dict:
    return {
        "loss": LOSS_AM,
        "am_scale": float(trial.suggest_float("am_scale", float(AM_SCALE_LOW), float(AM_SCALE_HIGH), log=True)),
        "am_margin": float(trial.suggest_float("am_margin", float(AM_MARGIN_LOW), float(AM_MARGIN_HIGH))),
    }


def suggest_oc(trial) -> dict:
    margin_spoof = float(
        trial.suggest_float("oc_margin_spoof", float(OC_MARGIN_SPOOF_LOW), float(OC_MARGIN_SPOOF_HIGH))
    )
    gap = float(trial.suggest_float("oc_margin_gap", float(OC_MARGIN_GAP_LOW), float(OC_MARGIN_GAP_HIGH)))
    margin_bonafide = min(float(OC_MARGIN_BONAFIDE_CAP), margin_spoof + gap)
    if margin_bonafide <= margin_spoof:
        raise RuntimeError("OC-Softmax margins collapsed; widen the gap bounds")
    return {
        "loss": LOSS_OC,
        "oc_scale": float(trial.suggest_float("oc_scale", float(OC_SCALE_LOW), float(OC_SCALE_HIGH), log=True)),
        "oc_margin_spoof": margin_spoof,
        "oc_margin_bonafide": margin_bonafide,
        "oc_margin_gap": gap,
    }


def suggest_rawboost(trial) -> dict:
    return {
        "rawboost": {
            "variant": str(trial.suggest_categorical("rawboost_variant", list(RAWBOOST_VARIANTS))),
            "strength": float(
                trial.suggest_float("rawboost_strength", float(RAWBOOST_STRENGTH_LOW), float(RAWBOOST_STRENGTH_HIGH))
            ),
            "n_taps": int(trial.suggest_categorical("rawboost_n_taps", list(RAWBOOST_N_TAPS_CHOICES))),
            "impulse_rate": float(
                trial.suggest_float(
                    "rawboost_impulse_rate",
                    float(RAWBOOST_IMPULSE_RATE_LOW),
                    float(RAWBOOST_IMPULSE_RATE_HIGH),
                    log=True,
                )
            ),
            "apply_probability": float(
                trial.suggest_float("rawboost_apply_p", float(RAWBOOST_APPLY_P_LOW), float(RAWBOOST_APPLY_P_HIGH))
            ),
        }
    }


def suggest_codec(trial) -> dict:
    return {
        "codec": {
            "choice": str(trial.suggest_categorical("codec_choice", list(CODEC_CHOICES))),
            "apply_probability": float(
                trial.suggest_float("codec_apply_p", float(CODEC_APPLY_P_LOW), float(CODEC_APPLY_P_HIGH))
            ),
        }
    }


STAGE_QUESTIONS = {
    5: "Даёт ли attentive statistics pooling прирост относительно mean pooling?",
    6: "Окупается ли AASIST-lite относительно победителя этапа 5?",
    7: "Лучше ли one-class loss на невидимых атаках, чем BCE и AM-Softmax? Смотреть худший LOSO-фолд.",
    8: "Даёт ли RawBoost устойчивость к артефактам других вокодеров? Смотреть худший фолд.",
    9: "Помогает ли симуляция телефонного тракта операторского звонка?",
}


def _fit_fold_with_oom_retry(hparams, context, train_index, val_index, seed: int, early_stop: bool):
    while True:
        batch = min(int(hparams["batch_size"]), int(context.batch_cap))
        try:
            return _fit(hparams, context, train_index, val_index, seed, early_stop, batch)
        except Exception as exc:
            if not is_oom(exc):
                raise
            release_cuda()
            reduced = lower_batch_bound(int(batch), context.oom_state)
            if reduced is None:
                raise
            context.batch_cap = int(reduced)
            print(f"CUDA OOM at batch {int(batch)}; retrying at {int(reduced)}", flush=True)


def _fit(hparams, context, train_index, val_index, seed: int, early_stop: bool, batch_size: int):
    af_common.set_global_seed(int(seed))
    device = context.device
    use_amp = device.type == "cuda"
    pin = bool(PIN_MEMORY_ON_CUDA and use_amp)
    augmented = _uses_waveform_aug(hparams)
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    train_loader = DataLoader(
        _dataset(context, train_index, augmented),
        batch_size=int(batch_size),
        shuffle=True,
        generator=generator,
        num_workers=int(NUM_WORKERS),
        drop_last=bool(DROP_LAST),
        pin_memory=pin,
    )
    val_loader = None
    if early_stop:
        val_loader = DataLoader(
            _dataset(context, val_index, False),
            batch_size=int(batch_size),
            shuffle=False,
            num_workers=int(NUM_WORKERS),
            drop_last=bool(DROP_LAST),
            pin_memory=pin,
        )
    model = None
    optimizer = None
    scaler = None
    try:
        if OPTIMIZER_NAME != "adamw":
            raise RuntimeError(f"unsupported optimizer {OPTIMIZER_NAME}")
        model = build_model(hparams).to(device)
        weight = positive_weight(context.labels, train_index, device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(hparams["lr"]),
            weight_decay=float(hparams["weight_decay"]),
        )
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if use_amp else None
        augmentor = WaveAugmentor(hparams) if augmented else None
        kind = KIND_WAVE if augmented else KIND_SPEC

        def prepare_batch(batch_x, batch_y):
            return _to_specs(batch_x, kind, context.frontend, augmentor), batch_y

        def compute_loss(net, batch_x, batch_y):
            return net.training_loss(batch_x, batch_y, weight)

        def on_epoch(epoch: int) -> None:
            print(
                f"full fit seed {int(seed)} epoch {epoch + int(DISPLAY_INDEX_OFFSET)}/{int(hparams['max_epochs'])}",
                flush=True,
            )

        trained = fit_epochs(
            model,
            train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            use_amp=use_amp,
            max_epochs=int(hparams["max_epochs"]),
            compute_loss=compute_loss,
            prepare_batch=prepare_batch,
            evaluate=None if not early_stop else (lambda: _eval_auc(model, val_loader, device, use_amp)),
            patience=int(EARLY_STOP_PATIENCE),
            on_epoch=None if early_stop else on_epoch,
        )
        if early_stop:
            return _eval_auc(model, val_loader, device, use_amp)
        return trained
    finally:
        if model is not None:
            model.cpu()
            del model
        del optimizer, scaler, train_loader, val_loader
        release_cuda()


@torch.no_grad()
def _eval_auc(model, loader, device, use_amp: bool) -> float:
    y_true, y_score = _collect(model, loader, device, use_amp)
    return float(af_common.binary_auc(y_true, y_score))


@torch.no_grad()
def _collect(model, loader, device, use_amp: bool) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    scores: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device, non_blocking=True)
        with torch.amp.autocast(device.type, enabled=use_amp):
            chunk = model.spoof_scores(batch_x).float()
        scores.append(chunk.cpu().numpy())
        targets.append(batch_y.reshape(FLATTEN_ALL).numpy())
    return np.concatenate(targets), np.concatenate(scores)


def _to_specs(batch_x, kind: str, frontend, augmentor):
    if kind == KIND_SPEC:
        return batch_x
    wave = batch_x
    if augmentor is not None:
        wave = augmentor(wave)
    feat = frontend(wave.cpu())
    if feat.dim() == int(SPEC_RANK_WITHOUT_CHANNEL):
        feat = feat.unsqueeze(CHANNEL_DIM)
    return feat.to(batch_x.device)


def _dataset(context: RunContext, indices, waveform: bool):
    from stage2.datautil import SpecDataset, WaveDataset

    if waveform:
        return WaveDataset(context.store, context.labels, np.asarray(indices))
    return SpecDataset(context.specs, context.labels, np.asarray(indices))


def _uses_waveform_aug(hparams: dict) -> bool:
    return bool(hparams.get("rawboost")) or bool(hparams.get("codec"))


def _make_frontend():
    """LFCC на CPU, то же размещение, что у кэша спектрограмм этапа 2."""
    from stage2.features import SpecFrontend

    frontend = SpecFrontend(FRONTEND_LFCC, int(BASELINE_N_BINS), int(BASELINE_N_FFT), int(BASELINE_HOP))
    return frontend.eval()


def _test_specs(ids: list[tuple[str, Path]], _device: torch.device) -> torch.Tensor:
    from stage2.datautil import load_wave_from_path

    frontend = _make_frontend()
    waves = [load_wave_from_path(path) for _name, path in ids]
    chunks: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(int(START_INDEX), len(waves), int(SPEC_COMPUTE_BATCH)):
            stop = min(len(waves), start + int(SPEC_COMPUTE_BATCH))
            batch = torch.from_numpy(np.stack(waves[start:stop], axis=START_INDEX))
            feat = frontend(batch)
            if feat.dim() == int(SPEC_RANK_WITHOUT_CHANNEL):
                feat = feat.unsqueeze(CHANNEL_DIM)
            chunks.append(feat.cpu())
            print(f"test specs {stop}/{len(waves)}", flush=True)
    return torch.cat(chunks, dim=START_INDEX)


def _score_tensor(model, specs: torch.Tensor, device, batch_size: int, use_amp: bool) -> np.ndarray:
    model.eval()
    chunks: list[np.ndarray] = []
    total = int(specs.shape[START_INDEX])
    with torch.no_grad():
        for start in range(int(START_INDEX), total, int(batch_size)):
            stop = min(total, start + int(batch_size))
            batch = specs[start:stop].to(device)
            with torch.amp.autocast(device.type, enabled=use_amp):
                chunks.append(model.spoof_scores(batch).float().cpu().numpy())
    if not chunks:
        raise RuntimeError("test spec tensor was empty")
    return np.concatenate(chunks)


def _index_candidates(candidates: list) -> dict:
    indexed = {}
    for item in candidates:
        key = str(item["arm"]) if item.get("arm") else str(item["name"])
        indexed[key] = item
    return indexed


def _heartbeat(stage: int) -> Path:
    return continuation_dir() / HEARTBEAT_TEMPLATE.format(stage=int(stage))


def _summary_ru(stage: int, reason: str, metrics: dict) -> str:
    worst = metrics.get("worst_fold_id", "")
    return (
        f"Этап {int(stage)}. {reason} "
        f"Итоговый mean AUC {format(float(metrics['mean_auc']), DISPLAY_METRIC_FORMAT)}, "
        f"худший фолд {worst} {format(float(metrics['worst_auc']), DISPLAY_METRIC_FORMAT)}, "
        f"std {format(float(metrics['std']), DISPLAY_METRIC_FORMAT)}, "
        f"CI [{format(float(metrics['ci_low']), DISPLAY_METRIC_FORMAT)}, "
        f"{format(float(metrics['ci_high']), DISPLAY_METRIC_FORMAT)}]."
    )


def _require_metric_keys(metrics: dict) -> None:
    missing = [key for key in ("mean_auc", "worst_auc") if key not in metrics]
    if missing:
        raise RuntimeError(f"winner metrics missing {missing}")


def self_check() -> None:
    """Проверка форм и решения на CPU. LOSO и тест не запускаются."""
    from experiments.augment import WaveAugmentor
    from experiments.config import baseline_hparams
    from experiments.decisions import accept_loso, fresh_payload

    if config_name(baseline_hparams()) != "lcnn64_lfcc_mean_bce":
        raise RuntimeError("baseline name drifted")
    chance = float(af_common.CHANCE_AUC)
    gain = float(MIN_MEAN_AUC_GAIN)
    kept = {"mean_auc": chance, "worst_auc": chance}
    better = {"mean_auc": chance + gain, "worst_auc": chance}
    flat = {"mean_auc": chance + gain / float(TWO), "worst_auc": chance}
    worse_tail = {"mean_auc": chance + gain, "worst_auc": chance - gain}
    if not accept_loso(kept, better):
        raise RuntimeError("accept_loso rejected a gain that meets the rule")
    if accept_loso(kept, flat) or accept_loso(kept, worse_tail):
        raise RuntimeError("accept_loso kept a stage that fails the rule")
    try:
        accept_loso(kept, {"mean_auc": chance + gain, "worst_auc": chance, "test_auc": chance})
    except RuntimeError:
        pass
    else:
        raise RuntimeError("accept_loso allowed a test score")

    af_common.set_global_seed(int(SEEDS[OPTUNA_SEED_INDEX]))
    specs = torch.randn(int(SMOKE_BATCH), int(CONV_IN_CHANNELS), int(BASELINE_N_BINS), int(SMOKE_FRAMES))
    labels = torch.tensor([float(af_common.LABEL_BONAFIDE), float(af_common.LABEL_SPOOF)])
    hparams = baseline_hparams()
    hparams["lcnn_width"] = int(BASELINE_LCNN_WIDTH)
    _step(hparams, specs, labels)
    asp = merge_hparams(hparams, {"pooling": POOL_ASP, "attention_hidden": int(ATTENTION_HIDDEN_LOW)})
    _step(asp, specs, labels)
    graph = merge_hparams(
        hparams,
        {"pooling": POOL_AASIST, "graph_layers": int(GRAPH_LAYERS_LOW), "graph_hidden": int(GRAPH_HIDDEN_LOW)},
    )
    _step(graph, specs, labels)

    class _FloorTrial:
        def suggest_float(self, name, low, high, log=False):
            if float(low) >= float(high):
                raise RuntimeError(f"{name} bounds are empty")
            return float(low)

    suggested_am = suggest_am(_FloorTrial())
    if suggested_am["am_scale"] != float(AM_SCALE_LOW) or suggested_am["am_margin"] != float(AM_MARGIN_LOW):
        raise RuntimeError("suggest_am did not honor the low bounds")
    am = merge_hparams(asp, {"loss": LOSS_AM, "am_scale": float(AM_SCALE_LOW), "am_margin": float(AM_MARGIN_LOW)})
    _step(am, specs, labels)
    oc = merge_hparams(
        graph,
        {
            "loss": LOSS_OC,
            "oc_scale": float(OC_SCALE_LOW),
            "oc_margin_spoof": float(OC_MARGIN_SPOOF_LOW),
            "oc_margin_bonafide": float(OC_MARGIN_SPOOF_LOW) + float(OC_MARGIN_GAP_LOW),
        },
    )
    _step(oc, specs, labels)

    from experiments.config import TARGET_LENGTH

    wave = torch.randn(int(SMOKE_BATCH), int(TARGET_LENGTH))
    augmentor = WaveAugmentor(
        {
            "rawboost": {
                "variant": RAWBOOST_VARIANTS[START_INDEX],
                "strength": float(RAWBOOST_STRENGTH_LOW),
                "n_taps": int(RAWBOOST_N_TAPS_CHOICES[START_INDEX]),
                "impulse_rate": float(RAWBOOST_IMPULSE_RATE_LOW),
                "apply_probability": float(RAWBOOST_APPLY_P_HIGH),
            },
            "codec": {"choice": CODEC_CHOICES[len(CODEC_CHOICES) - STALE_STEP], "apply_probability": float(RAWBOOST_APPLY_P_HIGH)},
        }
    )
    out = augmentor(wave)
    if tuple(out.shape) != tuple(wave.shape):
        raise RuntimeError(f"augment changed length: {tuple(out.shape)}")

    payload = fresh_payload()
    if payload["current_winner"]["name"] != payload["baseline"]["name"]:
        raise RuntimeError("fresh decisions do not start at the baseline")
    ensure_baseline()
    preview_stage(FIRST_STAGE)
    try:
        preview_stage(FIRST_STAGE + STALE_STEP)
    except RuntimeError as exc:
        if "stage 5" not in str(exc):
            raise
    else:
        if "5" not in ensure_baseline()["stages"]:
            raise RuntimeError("stage 6 ran without a stage 5 decision")
    print("self_check ok", flush=True)


def _step(hparams: dict, specs: torch.Tensor, labels: torch.Tensor) -> None:
    model = build_model(hparams)
    loss = model.training_loss(specs, labels, None)
    if not torch.isfinite(loss):
        raise RuntimeError(f"non-finite smoke loss for {config_name(hparams)}")
    loss.backward()
    scores = model.spoof_scores(specs)
    if scores.shape[START_INDEX] != int(SMOKE_BATCH):
        raise RuntimeError("smoke scores have the wrong batch")


if __name__ == "__main__":
    self_check()
