"""Этап 10. Частичное дообучение замороженного SSL-энкодера этапа 4.

Optuna выбирает last-K или LoRA. Замороженная голова: baseline, не испытание.
Отбор LOSO тест не читает. Early stopping смотрит внутренний сплит train.
"""

from __future__ import annotations

import gc
import time
from pathlib import Path

import numpy as np
import optuna
import torch
from torch import nn

import af_common
from experiments.config import INITIAL_BEST_AUC, START_INDEX
from jsonutil import beat, read_json, write_json
from torchutil import masked_mean_pool
from ssl_stages.common import (
    configure_determinism,
    decide_against,
    emit,
    format_seconds,
    frozen_ssl_baseline,
    inner_monitor_split,
    load_protocol_table,
    print_decision,
    print_metrics,
    stage_dir,
    submission_ids,
    summarize_loso,
    write_score_csv,
    write_sidecar,
)
from ssl_stages.constants import (
    DECISION_ACCEPT,
    FINETUNE_BATCH_CHOICES,
    FINETUNE_BATCH_FLOOR,
    FINETUNE_EPOCH_CHOICES,
    FINETUNE_LR_HIGH,
    FINETUNE_LR_LOW,
    FROZEN_SSL_BASELINE_NAME,
    GRAD_CLIP_NORM,
    HEARTBEAT_NAME,
    LAST_K_CHOICES,
    LORA_ALPHA_CHOICES,
    LORA_RANK_CHOICES,
    LORA_SCOPE,
    LORA_TARGET_SUFFIXES,
    N_OPTUNA_TRIALS,
    OPTUNA_SAMPLER_SEED,
    OPTUNA_STATE_NAME,
    PARTIAL_SEEDS_NAME,
    READOUT_HIDDEN_INDEX,
    RESULT_NAME,
    SEEDS,
    SSL_STAGES_CODE_VERSION,
    STAGE_PARTIAL,
    STRATEGIES,
    STRATEGY_LAST_K,
    STRATEGY_LORA,
    SUBMISSION_STAMP_NAME,
)
from ssl_stages.surgery import (
    apply_strategy,
    assert_backbone_api,
    eval_frozen_subtrees,
    feature_sample_rate,
    load_backbone,
    make_head,
    restore_trainable,
    snapshot_trainable,
    trainable_parameters,
)
from stage2.bounds import EARLY_STOP_PATIENCE, OPTIMIZER_NAME, OPTUNA_N_JOBS
from stage2.driver import is_oom
from stage4.constants import (
    AUDIO_MAX_SECONDS_CANDIDATES,
    HEAD_OPTIMIZER,
    MIN_AUDIO_SECONDS,
)
from stage4.head import _loso as frozen_head_loso
from stage4.head import fit_linear, predict_linear


def _dir() -> Path:
    path = stage_dir(STAGE_PARTIAL)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _heartbeat() -> Path:
    return _dir() / HEARTBEAT_NAME


def inherited_head() -> dict:
    """Исследование головы этапа 4. Эти ручки повторно не ищутся."""
    baseline = frozen_ssl_baseline()
    params = baseline["best_params"]
    return {
        "checkpoint": str(baseline["checkpoint"]),
        "frozen_layer": int(baseline["layer"]),
        "weight_decay": float(params["weight_decay"]),
        "pos_weight": float(params["pos_weight"]),
        "frozen_epochs": int(params["epochs"]),
        "frozen_batch_size": int(params["batch_size"]),
        "frozen_lr": float(params["lr"]),
    }


def search_space_document() -> dict:
    fixed = inherited_head()
    return {
        "n_trials": int(N_OPTUNA_TRIALS),
        "sampler_seed": int(OPTUNA_SAMPLER_SEED),
        "objective": "mean LOSO AUC at SEEDS[0]; the held-out fold is not used for early stopping",
        "strategy": list(STRATEGIES),
        "k": list(LAST_K_CHOICES),
        "k_when": STRATEGY_LAST_K,
        "lora_rank": list(LORA_RANK_CHOICES),
        "lora_alpha": list(LORA_ALPHA_CHOICES),
        "lora_when": STRATEGY_LORA,
        "lora_scope": LORA_SCOPE,
        "lora_targets": list(LORA_TARGET_SUFFIXES),
        "lr": [float(FINETUNE_LR_LOW), float(FINETUNE_LR_HIGH)],
        "lr_scale": "log",
        "epochs": list(FINETUNE_EPOCH_CHOICES),
        "batch_size": list(FINETUNE_BATCH_CHOICES),
        "batch_floor": int(FINETUNE_BATCH_FLOOR),
        "readout_hidden_index": int(READOUT_HIDDEN_INDEX),
        "grad_clip_norm": float(GRAD_CLIP_NORM),
        "early_stop_patience": int(EARLY_STOP_PATIENCE),
        "early_stop_split": "inner training synthesizer, not the LOSO holdout, not the test set",
        "inherited_from_stage4_head": {
            "weight_decay": fixed["weight_decay"],
            "pos_weight": fixed["pos_weight"],
            "checkpoint": fixed["checkpoint"],
            "frozen_layer": fixed["frozen_layer"],
        },
        "not_searched": (
            "weight_decay and pos_weight stay at the stage-4 head study. "
            "The frozen encoder is not a trial."
        ),
    }


def suggest(trial: optuna.Trial) -> dict:
    """Стратегия категориальная. K и rank/alpha LoRA условны от неё."""
    strategy = trial.suggest_categorical("strategy", list(STRATEGIES))
    params = {
        "strategy": strategy,
        "lr": trial.suggest_float("lr", float(FINETUNE_LR_LOW), float(FINETUNE_LR_HIGH), log=True),
        "epochs": trial.suggest_categorical("epochs", list(FINETUNE_EPOCH_CHOICES)),
        "batch_size": trial.suggest_categorical("batch_size", list(FINETUNE_BATCH_CHOICES)),
    }
    if strategy == STRATEGY_LAST_K:
        params["k"] = trial.suggest_categorical("k", list(LAST_K_CHOICES))
    elif strategy == STRATEGY_LORA:
        params["lora_rank"] = trial.suggest_categorical("lora_rank", list(LORA_RANK_CHOICES))
        params["lora_alpha"] = trial.suggest_categorical("lora_alpha", list(LORA_ALPHA_CHOICES))
    else:
        raise ValueError(strategy)
    return params


def _forward_logits(model, head, input_values, attention_mask):
    # Индекс -1: последний блок. Полный стек скрытых состояний в память не берём.
    if int(READOUT_HIDDEN_INDEX) != -1:
        raise RuntimeError("stage 10 reads the last hidden state only")
    outputs = model(
        input_values,
        attention_mask=attention_mask,
        output_hidden_states=False,
        return_dict=True,
    )
    hidden = outputs.last_hidden_state
    frame_mask = model._get_feature_vector_attention_mask(hidden.shape[1], attention_mask)
    pooled = masked_mean_pool(hidden, frame_mask)
    return head(pooled).reshape(-1)


def _load_batch(paths, indices, sample_rate: int, max_samples: int, min_samples: int):
    from stage4.audio import load_wave

    waves = []
    for index in indices:
        wave, _cropped = load_wave(Path(paths[int(index)]), sample_rate, max_samples, min_samples)
        waves.append(wave)
    return waves


def _collate(extractor, waves, labels, device: torch.device):
    inputs = extractor(
        waves,
        sampling_rate=int(extractor.sampling_rate),
        return_tensors="pt",
        padding=True,
        return_attention_mask=True,
    )
    input_values = inputs["input_values"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    target = torch.as_tensor(labels, dtype=torch.float32, device=device)
    return input_values, attention_mask, target


def _train_epoch(model, head, optimizer, scaler, criterion, extractor, paths, labels, indices, generator, batch_size, max_samples, min_samples, device):
    head.train()
    eval_frozen_subtrees(model)
    order = torch.randperm(int(indices.shape[0]), generator=generator).tolist()
    sample_rate = int(extractor.sampling_rate)
    use_amp = device.type == "cuda"
    for start in range(0, len(order), int(batch_size)):
        chosen = indices[order[start : start + int(batch_size)]]
        waves = _load_batch(paths, chosen, sample_rate, max_samples, min_samples)
        input_values, attention_mask, target = _collate(extractor, waves, labels[chosen], device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device.type, enabled=use_amp):
            loss = criterion(_forward_logits(model, head, input_values, attention_mask), target)
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite loss")
        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable_parameters(model, head), float(GRAD_CLIP_NORM))
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_parameters(model, head), float(GRAD_CLIP_NORM))
            optimizer.step()


@torch.inference_mode()
def _score_indices(model, head, extractor, paths, indices, max_samples, min_samples, batch_size, device) -> np.ndarray:
    model.eval()
    head.eval()
    sample_rate = int(extractor.sampling_rate)
    chunks = []
    indices = np.asarray(indices, dtype=np.int64)
    for start in range(0, int(indices.shape[0]), int(batch_size)):
        chosen = indices[start : start + int(batch_size)]
        waves = _load_batch(paths, chosen, sample_rate, max_samples, min_samples)
        input_values, attention_mask, _target = _collate(
            extractor, waves, np.zeros(int(chosen.shape[0]), dtype=np.float32), device
        )
        logits = _forward_logits(model, head, input_values, attention_mask)
        chunks.append(torch.sigmoid(logits.float()).cpu().numpy())
    scores = np.concatenate(chunks)
    if not np.isfinite(scores).all():
        raise RuntimeError("non-finite fine-tune scores")
    return scores


def _build(checkpoint: str, params: dict, seed: int, device: torch.device):
    af_common.set_global_seed(int(seed))
    extractor, model = load_backbone(checkpoint, device)
    af_common.set_global_seed(int(seed))
    note = apply_strategy(model, params)
    eval_frozen_subtrees(model)
    hidden = int(model.config.hidden_size)
    head = make_head(hidden, device)
    if OPTIMIZER_NAME != "adamw" or HEAD_OPTIMIZER != "adamw":
        raise RuntimeError(f"optimizer must be adamw, got {OPTIMIZER_NAME} / {HEAD_OPTIMIZER}")
    optimizer = torch.optim.AdamW(
        trainable_parameters(model, head),
        lr=float(params["lr"]),
        weight_decay=float(params["weight_decay"]),
    )
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    pos_weight = torch.tensor([float(params["pos_weight"])], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    return extractor, model, head, optimizer, scaler, criterion, note


def _release(*objects) -> None:
    """Освободить память GPU для переданных модулей и оптимизаторов.

    del в цикле снимает только имя. Кортеж аргументов и список у вызывающего
    держали бы объекты после gc.collect. Список очищается здесь.
    """
    pending: list = []
    for item in objects:
        if isinstance(item, list):
            pending.extend(item)
            item.clear()
        else:
            pending.append(item)
    del objects
    pending.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def train_fold_scores(
    table,
    paths,
    labels,
    fold,
    params: dict,
    seed: int,
    device: torch.device,
    max_samples: int,
    min_samples: int,
    *,
    early_stop: bool,
) -> np.ndarray:
    """Учить синтезаторы train LOSO. В конце скорить только fold.val_index.

    При early_stop монитор: inner_monitor_split, без holdout LOSO.
    Фиты сдачи передают early_stop=False.
    """
    extractor = model = head = optimizer = scaler = criterion = None
    try:
        extractor, model, head, optimizer, scaler, criterion, note = _build(
            str(params["checkpoint"]), params, seed, device
        )
        batch_size = int(params["batch_size"])
        generator = torch.Generator()
        generator.manual_seed(int(seed))
        if early_stop:
            inner_train, inner_val = inner_monitor_split(table, fold.train_index, str(fold.fold_id))
            best_auc = float(INITIAL_BEST_AUC)
            best_state = None
            stale = 0
            for _epoch in range(int(params["epochs"])):
                _train_epoch(
                    model, head, optimizer, scaler, criterion, extractor, paths, labels,
                    inner_train, generator, batch_size, max_samples, min_samples, device,
                )
                monitor_scores = _score_indices(
                    model, head, extractor, paths, inner_val, max_samples, min_samples, batch_size, device
                )
                monitor_auc = af_common.binary_auc(labels[inner_val], monitor_scores)
                if monitor_auc > best_auc:
                    best_auc = monitor_auc
                    best_state = snapshot_trainable(model, head)
                    stale = 0
                else:
                    stale += 1
                    if stale >= int(EARLY_STOP_PATIENCE):
                        break
            if best_state is None:
                raise RuntimeError("inner early stopping saved no weights")
            restore_trainable(model, head, best_state)
            emit(f"  {note}; best inner-monitor AUC {best_auc:.4f} on {fold.fold_id}")
        else:
            train_rows = np.asarray(fold.train_index)
            for _epoch in range(int(params["epochs"])):
                _train_epoch(
                    model, head, optimizer, scaler, criterion, extractor, paths, labels,
                    train_rows, generator, batch_size, max_samples, min_samples, device,
                )
        if int(np.asarray(fold.val_index).size) == 0:
            return np.zeros(0, dtype=np.float64)
        return _score_indices(
            model, head, extractor, paths, fold.val_index, max_samples, min_samples, batch_size, device
        )
    finally:
        batch = [extractor, model, head, optimizer, scaler, criterion]
        extractor = model = head = optimizer = scaler = criterion = None
        _release(batch)


def _audio_limits(sample_rate: int, cap_seconds: float) -> tuple[int, int]:
    max_samples = int(round(float(cap_seconds) * int(sample_rate)))
    min_samples = max(1, int(round(float(MIN_AUDIO_SECONDS) * int(sample_rate))))
    return max_samples, min_samples


def _cap_schedule(start_seconds: float) -> tuple[float, ...]:
    chosen = [float(cap) for cap in AUDIO_MAX_SECONDS_CANDIDATES if float(cap) <= float(start_seconds)]
    if not chosen:
        return tuple(float(cap) for cap in AUDIO_MAX_SECONDS_CANDIDATES)
    return tuple(chosen)


def _batch_schedule(requested: int) -> tuple[int, ...]:
    sizes = [int(size) for size in FINETUNE_BATCH_CHOICES if int(size) <= int(requested)]
    if int(FINETUNE_BATCH_FLOOR) not in sizes:
        sizes.append(int(FINETUNE_BATCH_FLOOR))
    ordered = []
    for size in sorted(sizes, reverse=True):
        if size not in ordered:
            ordered.append(size)
    return tuple(ordered)


def run_loso_seed(table, folds, labels, params: dict, seed: int, device: torch.device, cap_seconds: float) -> dict[str, float]:
    paths = table[af_common.PATH_COLUMN].tolist()
    sample_rate = feature_sample_rate(str(params["checkpoint"]))
    max_samples, min_samples = _audio_limits(sample_rate, cap_seconds)
    fold_aucs = {}
    for fold in folds:
        started = time.perf_counter()
        scores = train_fold_scores(
            table, paths, labels, fold, params, seed, device, max_samples, min_samples, early_stop=True
        )
        fold_aucs[str(fold.fold_id)] = af_common.binary_auc(labels[fold.val_index], scores)
        elapsed = time.perf_counter() - started
        beat(
            _heartbeat(),
            f"stage10 seed {seed} fold {fold.fold_id} auc {fold_aucs[str(fold.fold_id)]:.4f} {elapsed:.0f}s",
        )
    return fold_aucs


def _attempt_loso(table, folds, labels, params, seed, device, cap_seconds: float) -> dict[str, float]:
    schedule_caps = _cap_schedule(cap_seconds)
    schedule_batches = _batch_schedule(int(params["batch_size"]))
    last_error: Exception | None = None
    for cap in schedule_caps:
        for batch in schedule_batches:
            trial_params = dict(params)
            trial_params["batch_size"] = int(batch)
            trial_params["max_audio_seconds"] = float(cap)
            try:
                fold_aucs = run_loso_seed(table, folds, labels, trial_params, seed, device, cap)
                params["batch_size"] = int(batch)
                params["max_audio_seconds"] = float(cap)
                return fold_aucs
            except Exception as exc:
                _release()
                if not is_oom(exc):
                    raise
                last_error = exc
                beat(_heartbeat(), f"stage10 OOM cap {cap} batch {batch}")
    raise RuntimeError("partial fine-tune does not fit in GPU memory") from last_error


def _with_inherited(params: dict) -> dict:
    merged = dict(params)
    fixed = inherited_head()
    merged["checkpoint"] = fixed["checkpoint"]
    merged["weight_decay"] = fixed["weight_decay"]
    merged["pos_weight"] = fixed["pos_weight"]
    merged["readout_hidden_index"] = int(READOUT_HIDDEN_INDEX)
    return merged


def preview_stage() -> str:
    baseline = frozen_ssl_baseline()
    metrics = baseline["metrics"]
    text = (
        f"Этап {int(STAGE_PARTIAL)}. Эталон: {baseline['name']} "
        f"({baseline['checkpoint']}, замороженный слой {baseline['layer']}). "
        f"LOSO mean AUC {format(float(metrics['mean_auc']), '.4f')}, "
        f"худший фолд {metrics['worst_fold_id']} {format(float(metrics['worst_auc']), '.4f')}. "
        "Решение только по LOSO. Тест не участвует."
    )
    emit(text)
    emit(f"Пространство поиска: {search_space_document()}")
    return text


def run_optuna(device: torch.device) -> dict:
    """Одно исследование, N_OPTUNA_TRIALS, семя OPTUNA_SAMPLER_SEED. Продолжение из sqlite."""
    state_path = _dir() / OPTUNA_STATE_NAME
    if state_path.is_file():
        existing = read_json(state_path)
        if (
            int(existing.get("code_version", -1)) == int(SSL_STAGES_CODE_VERSION)
            and int(existing.get("n_trials", 0)) == int(N_OPTUNA_TRIALS)
            and existing.get("best_params")
        ):
            emit("stage 10 optuna already complete")
            return existing

    table, folds = load_protocol_table()
    labels = table[af_common.LABEL_COLUMN].to_numpy(dtype=np.int64)
    fixed = inherited_head()
    start_cap = float(AUDIO_MAX_SECONDS_CANDIDATES[START_INDEX])
    storage_path = _dir() / "optuna.db"
    storage = "sqlite:///" + storage_path.resolve().as_posix()
    sampler = optuna.samplers.TPESampler(seed=int(OPTUNA_SAMPLER_SEED))
    study = optuna.create_study(
        study_name="stage10_partial_finetune",
        storage=storage,
        direction="maximize",
        sampler=sampler,
        load_if_exists=True,
    )

    def objective(trial: optuna.Trial) -> float:
        params = _with_inherited(suggest(trial))
        beat(_heartbeat(), f"stage10 trial {trial.number} start { {k: params[k] for k in params if k != 'pos_weight'} }")
        fold_aucs = _attempt_loso(table, folds, labels, params, int(SEEDS[START_INDEX]), device, start_cap)
        summary = af_common.summarize_fold_aucs(fold_aucs)
        trial.set_user_attr("effective_batch_size", int(params["batch_size"]))
        trial.set_user_attr("max_audio_seconds", float(params["max_audio_seconds"]))
        trial.set_user_attr("fold_aucs", summary["fold_aucs"])
        beat(_heartbeat(), f"stage10 trial {trial.number} mean_auc {summary['mean_auc']:.4f}")
        return float(summary["mean_auc"])

    with af_common.gpu_lock():
        if device.type != "cuda":
            raise RuntimeError("CUDA is required")
        configure_determinism()
        n_layers = assert_backbone_api(fixed["checkpoint"], device)
        emit(f"encoder layers {n_layers}; frozen baseline is not a trial")
        complete = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
        remaining = int(N_OPTUNA_TRIALS) - len(complete)
        if remaining > 0:
            study.optimize(objective, n_trials=remaining, n_jobs=int(OPTUNA_N_JOBS))
        complete = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
        if len(complete) != int(N_OPTUNA_TRIALS):
            raise RuntimeError(f"stage 10 completed {len(complete)} trials, expected {N_OPTUNA_TRIALS}")

    best = study.best_trial
    params = _with_inherited(dict(best.params))
    params["batch_size"] = int(best.user_attrs.get("effective_batch_size", params["batch_size"]))
    params["max_audio_seconds"] = float(best.user_attrs["max_audio_seconds"])
    payload = {
        "code_version": int(SSL_STAGES_CODE_VERSION),
        "n_trials": int(N_OPTUNA_TRIALS),
        "sampler_seed": int(OPTUNA_SAMPLER_SEED),
        "best_value": float(study.best_value),
        "best_params": params,
        "search_space": search_space_document(),
        "trials": [
            {
                "number": int(trial.number),
                "value": float(trial.value),
                "params": trial.params,
                "effective_batch_size": trial.user_attrs.get("effective_batch_size"),
                "max_audio_seconds": trial.user_attrs.get("max_audio_seconds"),
            }
            for trial in complete
        ],
    }
    write_json(state_path, payload)
    emit(f"stage 10 selected {params} optuna mean_auc {study.best_value:.4f}")
    return payload


def _time_frozen_head(table, labels, folds, device: torch.device) -> float:
    """Переобучить сохранённую замороженную голову на кэше. Это временной baseline."""
    from stage4.cache_io import cache_dir_from_pointer, cache_is_complete, read_layer
    from stage4.data import table_fingerprint

    baseline = frozen_ssl_baseline()
    fingerprint = table_fingerprint(table)
    found = cache_dir_from_pointer(baseline["checkpoint"], fingerprint)
    if found is None:
        raise FileNotFoundError(baseline["checkpoint"])
    directory, meta = found
    if not cache_is_complete(meta):
        raise RuntimeError("stage-4 feature cache is incomplete")
    features = read_layer(directory, int(baseline["layer"]), int(meta["n_rows"]), int(meta["hidden"]))
    started = time.perf_counter()
    for seed in SEEDS:
        frozen_head_loso(features, labels, folds, baseline["best_params"], int(seed), device)
    return float(time.perf_counter() - started)


def evaluate_seeds(device: torch.device) -> dict:
    """Переобучить выбранное испытание на каждом семени протокола. Отбор остаётся на LOSO."""
    state = read_json(_dir() / OPTUNA_STATE_NAME)
    params = state["best_params"]
    partial_path = _dir() / PARTIAL_SEEDS_NAME
    saved = {}
    frozen_seconds = None
    finetune_seconds = 0.0
    if partial_path.is_file():
        previous = read_json(partial_path)
        if previous.get("params") == params and int(previous.get("code_version", -1)) == int(SSL_STAGES_CODE_VERSION):
            saved = dict(previous.get("per_seed_fold_aucs", {}))
            frozen_seconds = previous.get("frozen_head_seconds")
            finetune_seconds = float(previous.get("finetune_seconds") or 0.0)

    table, folds = load_protocol_table()
    labels = table[af_common.LABEL_COLUMN].to_numpy(dtype=np.int64)
    per_seed = dict(saved)

    with af_common.gpu_lock():
        if device.type != "cuda":
            raise RuntimeError("CUDA is required")
        configure_determinism()
        if frozen_seconds is None:
            frozen_seconds = _time_frozen_head(table, labels, folds, device)
            emit(f"frozen head, 3 seeds: {format_seconds(frozen_seconds)} s")
        for seed in SEEDS:
            key = str(int(seed))
            if key in per_seed:
                emit(f"reuse stage 10 seed {key}")
                continue
            started = time.perf_counter()
            fold_aucs = run_loso_seed(
                table, folds, labels, params, int(seed), device, float(params["max_audio_seconds"])
            )
            finetune_seconds += float(time.perf_counter() - started)
            per_seed[key] = fold_aucs
            write_json(
                partial_path,
                {
                    "code_version": int(SSL_STAGES_CODE_VERSION),
                    "params": params,
                    "per_seed_fold_aucs": per_seed,
                    "finetune_seconds": finetune_seconds,
                    "frozen_head_seconds": frozen_seconds,
                },
            )

    by_seed = {int(seed): per_seed[str(int(seed))] for seed in SEEDS}
    summary = summarize_loso(by_seed)
    payload = {
        "code_version": int(SSL_STAGES_CODE_VERSION),
        "params": params,
        "per_seed_fold_aucs": per_seed,
        "finetune_seconds": finetune_seconds,
        "frozen_head_seconds": frozen_seconds,
        "summary": summary,
    }
    write_json(partial_path, payload)
    print_metrics("частичное дообучение", summary["view"])
    return payload


def decide() -> dict:
    seeds = read_json(_dir() / PARTIAL_SEEDS_NAME)
    view = seeds["summary"]["view"] if "summary" in seeds else summarize_loso(
        {int(seed): seeds["per_seed_fold_aucs"][str(int(seed))] for seed in SEEDS}
    )["view"]
    baseline = frozen_ssl_baseline()
    report = decide_against(baseline["name"], "partial_finetune", baseline["metrics"], view)
    finetune_seconds = float(seeds["finetune_seconds"])
    frozen_seconds = float(seeds["frozen_head_seconds"])
    ratio = finetune_seconds / frozen_seconds if frozen_seconds > 0.0 else None
    report["train_seconds"] = {
        "partial_finetune_three_seeds": finetune_seconds,
        "frozen_head_three_seeds": frozen_seconds,
        "ratio_finetune_over_frozen": ratio,
    }
    print_decision(report)
    emit(
        "Время 3 семян, частичное дообучение: "
        + format_seconds(finetune_seconds)
        + " с. Замороженная голова: "
        + format_seconds(frozen_seconds)
        + " с."
    )
    if ratio is not None:
        emit(f"Отношение времени (дообучение / замороженная голова): {ratio:.1f}")
    if report["decision"] == DECISION_ACCEPT:
        emit("Правило LOSO этап принимает. Отношение времени — справка; отдельного порога по времени в протоколе нет.")
    else:
        emit("Дополнительное время не окупается: этап отклонён правилом LOSO. Остаётся замороженный SSL.")
    payload = {
        "stage": int(STAGE_PARTIAL),
        "code_version": int(SSL_STAGES_CODE_VERSION),
        "status": "complete",
        "baseline_name": FROZEN_SSL_BASELINE_NAME,
        "baseline_metrics": baseline["metrics"],
        "candidate_metrics": view,
        "best_params": seeds["params"],
        "search_space": search_space_document(),
        "optuna_best_value": read_json(_dir() / OPTUNA_STATE_NAME).get("best_value"),
        "decision": report,
        "test_used_for_selection": False,
    }
    write_json(_dir() / RESULT_NAME, payload)
    return payload


def _write_frozen_submissions(device: torch.device) -> list[str]:
    """Отклонённый этап: замороженная голова этапа 4, переобучение на всех размеченных строках."""
    from stage4.cache_io import cache_dir_from_pointer, cache_is_complete, read_layer
    from stage4.data import table_fingerprint

    baseline = frozen_ssl_baseline()
    table, _folds = load_protocol_table()
    fingerprint = table_fingerprint(table)
    found = cache_dir_from_pointer(baseline["checkpoint"], fingerprint)
    if found is None:
        raise FileNotFoundError(baseline["checkpoint"])
    directory, meta = found
    if not cache_is_complete(meta):
        raise RuntimeError("stage-4 feature cache is incomplete")
    from stage4.submit import _encode_test, _read_test_cache

    features = read_layer(directory, int(baseline["layer"]), int(meta["n_rows"]), int(meta["hidden"]))
    labels = table[af_common.LABEL_COLUMN].to_numpy(dtype=np.int64)
    ids = submission_ids()
    names = [name for name, _path in ids]
    test_features = _read_test_cache(baseline["checkpoint"], int(baseline["layer"]), names)
    if test_features is None:
        test_features = _encode_test(
            baseline["checkpoint"],
            int(baseline["layer"]),
            float(meta["max_audio_seconds"]),
            ids,
            device,
        )
    train_index = np.arange(len(labels))
    written = []
    for seed in SEEDS:
        model, scaler = fit_linear(features, labels, train_index, baseline["best_params"], int(seed), device)
        scores = predict_linear(model, scaler, test_features, device)
        destination = af_common.submission_path(int(seed))
        write_score_csv(destination, ids, scores)
        write_sidecar(
            int(seed),
            {
                "stage": int(STAGE_PARTIAL),
                "decision": "REJECT",
                "system": FROZEN_SSL_BASELINE_NAME,
                "checkpoint": baseline["checkpoint"],
                "layer": int(baseline["layer"]),
                "note": "stage 10 rejected; submission is the frozen SSL baseline. No test metric was computed.",
            },
        )
        written.append(str(destination))
        emit(f"wrote {destination}")
        del model
    return written


def _score_test_adapted(params: dict, seed: int, device: torch.device, ids) -> np.ndarray:
    """Учить все размеченные строки выбранное число эпох. Early stopping выключен: holdout нет.

    Тестовые волны скорятся после фита. Их метки не загружаются и эпохи не меняют.
    """
    table, _folds = load_protocol_table()
    labels = table[af_common.LABEL_COLUMN].to_numpy(dtype=np.int64)
    paths = table[af_common.PATH_COLUMN].tolist()
    rate = feature_sample_rate(str(params["checkpoint"]))
    max_samples, min_samples = _audio_limits(rate, float(params["max_audio_seconds"]))
    train_rows = np.arange(len(labels), dtype=np.int64)
    extractor = model = head = optimizer = scaler = criterion = None
    try:
        extractor, model, head, optimizer, scaler, criterion, _note = _build(
            str(params["checkpoint"]), params, seed, device
        )
        generator = torch.Generator()
        generator.manual_seed(int(seed))
        for _epoch in range(int(params["epochs"])):
            _train_epoch(
                model, head, optimizer, scaler, criterion, extractor, paths, labels,
                train_rows, generator, int(params["batch_size"]), max_samples, min_samples, device,
            )
            emit(f"submission seed {seed} epoch {_epoch + 1}/{int(params['epochs'])}")
        test_paths = [path for _name, path in ids]
        waves_index = np.arange(len(test_paths), dtype=np.int64)
        return _score_indices(
            model, head, extractor, test_paths, waves_index,
            max_samples, min_samples, int(params["batch_size"]), device,
        )
    finally:
        batch = [extractor, model, head, optimizer, scaler, criterion]
        extractor = model = head = optimizer = scaler = criterion = None
        _release(batch)


def write_submission(device: torch.device) -> dict:
    """Записать submission_seed_{seed}.csv только для принятой системы.

    Если этап 10 отклонён, файл: замороженный SSL baseline.
    Тестовый AUC, порог и выбор модели здесь не считаются. Файл решения уже должен быть.
    """
    result_path = _dir() / RESULT_NAME
    if not result_path.is_file():
        raise RuntimeError("call decide() before write_submission(); the test set cannot choose the model")
    result = read_json(result_path)
    decision = result["decision"]["decision"]
    written = []
    with af_common.gpu_lock():
        if device.type != "cuda":
            raise RuntimeError("CUDA is required")
        configure_determinism()
        if decision == DECISION_ACCEPT:
            emit("Этап 10 принят. Файл сдачи: частичное дообучение. Тест только как файл сдачи, без метрик.")
            ids = submission_ids()
            params = result["best_params"]
            for seed in SEEDS:
                scores = _score_test_adapted(params, int(seed), device, ids)
                destination = af_common.submission_path(int(seed))
                write_score_csv(destination, ids, scores)
                write_sidecar(
                    int(seed),
                    {
                        "stage": int(STAGE_PARTIAL),
                        "decision": DECISION_ACCEPT,
                        "system": "partial_finetune",
                        "params": params,
                        "note": "No test metric, threshold, or early stopping used the competition test set.",
                    },
                )
                written.append(str(destination))
                emit(f"wrote {destination}")
        else:
            emit("Этап 10 отклонён. Файл сдачи: замороженный SSL этапа 4, не новая модель.")
            written = _write_frozen_submissions(device)
    stamp = {"stage": int(STAGE_PARTIAL), "decision": decision, "paths": written, "seeds": [int(seed) for seed in SEEDS]}
    write_json(_dir() / SUBMISSION_STAMP_NAME, stamp)
    return stamp
