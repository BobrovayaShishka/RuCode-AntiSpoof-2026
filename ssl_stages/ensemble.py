"""Этап 11. Равное среднее рангов XLS-R, WavLM и принятой модели этапа 2.

Вес смеси не подбирается. Optuna не запускается: свободного веса нет.
Сравнение с лучшей из трёх одиночных моделей на общих фолдах LOSO.
Тест соревнования модель и смесь не выбирает.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import af_common
from jsonutil import beat, read_json, write_json
from ssl_stages.common import (
    baseline_stage_banner,
    configure_determinism,
    decide_against,
    emit,
    equal_rank_average,
    fold_win_report,
    load_protocol_table,
    print_decision,
    print_fold_wins,
    print_metrics,
    stage2_record,
    stage_dir,
    submission_ids,
    summarize_loso,
    write_score_csv,
    write_sidecar,
)
from ssl_stages.constants import (
    BASELINE_STAGE_NAME,
    DECISION_ACCEPT,
    ENSEMBLE_NAME,
    FUSION_NAME,
    FUSION_WEIGHTS_FIT,
    HEARTBEAT_NAME,
    N_OPTUNA_TRIALS,
    OOF_DIRNAME,
    OPTUNA_SAMPLER_SEED,
    PARTIAL_SEEDS_NAME,
    RESULT_NAME,
    SEEDS,
    SSL_STAGES_CODE_VERSION,
    STAGE_ENSEMBLE,
    SUBMISSION_STAMP_NAME,
    WAVLM_MODEL_ID,
    WAVLM_SINGLE_NAME,
    XLSR_MODEL_ID,
    XLSR_SINGLE_NAME,
)
from stage2.bounds import NUM_WORKERS, PIN_MEMORY_ON_CUDA
from stage2.loop import collect_scores, fit_fold, fit_full
from stage2.models import spectrogram_factories
from stage4.constants import HEAD_RESULT_PATH, LAYER_PROBE_PATH
from stage4.head import fit_linear, predict_linear
from stage4.summaries import layer_rank


def _dir() -> Path:
    path = stage_dir(STAGE_ENSEMBLE)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _heartbeat() -> Path:
    return _dir() / HEARTBEAT_NAME


def _oof_dir() -> Path:
    path = _dir() / OOF_DIRNAME
    path.mkdir(parents=True, exist_ok=True)
    return path


def note_no_optuna() -> str:
    if FUSION_WEIGHTS_FIT:
        raise RuntimeError("stage 11 does not fit fusion weights")
    text = (
        f"Optuna не запускается. Смесь — {FUSION_NAME}: равные веса рангов, "
        f"свободного веса нет. Неизрасходованный бюджет {int(N_OPTUNA_TRIALS)} испытаний, "
        f"семья сэмплера было бы {int(OPTUNA_SAMPLER_SEED)}. "
        "Подбор весов на тесте не делается."
    )
    emit(text)
    return text


def _wavlm_layer(probe: dict) -> dict:
    layers = probe["models"][WAVLM_MODEL_ID]["layers"]
    return max(layers.values(), key=layer_rank)


def _ssl_specs() -> list[dict]:
    """XLS-R берёт слой головы этапа 4. WavLM берёт лучший слой пробника.

    Отдельного исследования головы у WavLM нет: гиперпараметры этапа 4
    переиспользуются, второй бюджет испытаний не выдаётся.
    """
    if not HEAD_RESULT_PATH.is_file():
        raise FileNotFoundError(HEAD_RESULT_PATH)
    if not LAYER_PROBE_PATH.is_file():
        raise FileNotFoundError(LAYER_PROBE_PATH)
    head = read_json(HEAD_RESULT_PATH)
    probe = read_json(LAYER_PROBE_PATH)
    if str(head["checkpoint"]) != XLSR_MODEL_ID:
        raise RuntimeError(f"stage-4 head checkpoint {head['checkpoint']} is not {XLSR_MODEL_ID}")
    wavlm = _wavlm_layer(probe)
    return [
        {
            "name": XLSR_SINGLE_NAME,
            "checkpoint": str(head["checkpoint"]),
            "layer": int(head["layer"]),
            "head_params": head["best_params"],
            "head_params_source": "outputs/stage4/head_result.json",
        },
        {
            "name": WAVLM_SINGLE_NAME,
            "checkpoint": WAVLM_MODEL_ID,
            "layer": int(wavlm["layer"]),
            "head_params": head["best_params"],
            "head_params_source": "transferred from the stage-4 XLS-R head study; no extra Optuna budget",
        },
    ]


def _load_cached_features(table, checkpoint: str, layer: int):
    from stage4.cache_io import cache_dir_from_pointer, cache_is_complete, read_layer
    from stage4.data import table_fingerprint

    fingerprint = table_fingerprint(table)
    found = cache_dir_from_pointer(checkpoint, fingerprint)
    if found is None:
        raise FileNotFoundError(f"no stage-4 feature cache for {checkpoint}")
    directory, meta = found
    if not cache_is_complete(meta):
        raise RuntimeError(f"incomplete feature cache for {checkpoint}")
    features = read_layer(directory, int(layer), int(meta["n_rows"]), int(meta["hidden"]))
    return features, meta, fingerprint


def _ssl_oof(features, labels, folds, params: dict, seed: int, device: torch.device):
    scores = np.full(int(labels.shape[0]), np.nan, dtype=np.float64)
    fold_aucs = {}
    for fold in folds:
        model, scaler = fit_linear(features, labels, fold.train_index, params, int(seed), device)
        predicted = predict_linear(model, scaler, features[fold.val_index], device)
        scores[fold.val_index] = predicted
        fold_aucs[str(fold.fold_id)] = af_common.binary_auc(labels[fold.val_index], predicted)
        del model
    if not np.isfinite(scores).all():
        raise RuntimeError("SSL out-of-fold scores left a row unscored")
    return scores, fold_aucs


def _stage2_bundle(table, labels):
    from stage2.datautil import make_store
    from stage2.features import ensure_spec_memmap

    record = stage2_record()
    params = dict(record["best_params"])
    store = make_store(table)
    store.prepare(af_common.STAGE2_DIR)
    specs = ensure_spec_memmap(
        store,
        str(params["frontend"]),
        int(params["n_bins"]),
        int(params["n_fft"]),
        int(params["hop"]),
        af_common.STAGE2_DIR / "feat_cache",
    )

    model_fn, dataset_fn = spectrogram_factories(params, specs, labels)
    return params, model_fn, dataset_fn, specs


def _stage2_fold_scores(model_fn, dataset_fn, train_index, val_index, labels, seed: int, params: dict, device):
    """Валидационные скоры из stage2.loop.fit_fold.

    Early stopping смотрит AUC этого фолда, как при обучении этапа 2.
    Этап 11 его не перенастраивает. Тест соревнования не участвует.
    """
    return fit_fold(
        model_fn,
        dataset_fn,
        np.asarray(train_index),
        np.asarray(val_index),
        labels,
        seed=int(seed),
        lr=float(params["lr"]),
        weight_decay=float(params["weight_decay"]),
        max_epochs=int(params["max_epochs"]),
        batch_size=int(params["batch_size"]),
        device=device,
        auc_fn=af_common.binary_auc,
        seed_fn=af_common.set_global_seed,
        return_scores=True,
    )


def _stage2_oof(model_fn, dataset_fn, folds, labels, seed: int, params: dict, device):
    scores = np.full(int(labels.shape[0]), np.nan, dtype=np.float64)
    fold_aucs = {}
    for fold in folds:
        started = time.perf_counter()
        predicted = _stage2_fold_scores(
            model_fn, dataset_fn, fold.train_index, fold.val_index, labels, seed, params, device
        )
        if int(predicted.shape[0]) != int(np.asarray(fold.val_index).shape[0]):
            raise RuntimeError("stage-2 score count does not match the fold")
        scores[fold.val_index] = predicted
        fold_aucs[str(fold.fold_id)] = af_common.binary_auc(labels[fold.val_index], predicted)
        beat(
            _heartbeat(),
            f"stage11 stage2 seed {seed} fold {fold.fold_id} auc {fold_aucs[str(fold.fold_id)]:.4f} "
            f"{time.perf_counter() - started:.0f}s",
        )
    if not np.isfinite(scores).all():
        raise RuntimeError("stage-2 out-of-fold scores left a row unscored")
    return scores, fold_aucs


def _cache_path(name: str, seed: int) -> Path:
    return _oof_dir() / f"{name}_seed{int(seed)}.npy"


def _stamp_path(name: str) -> Path:
    return _oof_dir() / f"{name}.json"


def _load_cached(name: str, stamp: dict, n_rows: int):
    path = _stamp_path(name)
    if not path.is_file():
        return None
    existing = read_json(path)
    if existing.get("stamp") != stamp or int(existing.get("code_version", -1)) != int(SSL_STAGES_CODE_VERSION):
        return None
    loaded = {}
    for seed in SEEDS:
        scores_path = _cache_path(name, int(seed))
        if not scores_path.is_file():
            return None
        scores = np.load(scores_path)
        if int(scores.shape[0]) != int(n_rows):
            return None
        loaded[int(seed)] = scores
    return loaded


def _store_cached(name: str, stamp: dict, by_seed: dict[int, np.ndarray], fold_aucs: dict) -> None:
    for seed, scores in by_seed.items():
        np.save(_cache_path(name, int(seed)), np.asarray(scores, dtype=np.float64))
    write_json(
        _stamp_path(name),
        {
            "code_version": int(SSL_STAGES_CODE_VERSION),
            "stamp": stamp,
            "fold_aucs": {str(seed): fold_aucs[int(seed)] for seed in fold_aucs},
        },
    )


def preview_stage() -> str:
    specs = _ssl_specs()
    record = stage2_record()
    saved = record["aggregate"]
    text = (
        f"Этап {int(STAGE_ENSEMBLE)}. Одиночные модели: "
        f"{XLSR_MODEL_ID} слой {specs[0]['layer']}, "
        f"{WAVLM_MODEL_ID} слой {specs[1]['layer']}, "
        f"{BASELINE_STAGE_NAME}. {baseline_stage_banner()} "
        f"Сохранённый stage-2 mean AUC {float(saved['mean_auc']):.4f}. "
        "Для решения все три модели считаются заново одной агрегацией: "
        "худший фолд — минимум AUC после усреднения по семенам."
    )
    emit(text)
    note_no_optuna()
    return text


def collect_single_models(device: torch.device) -> dict:
    """Out-of-fold скоры трёх одиночных моделей. Те же фолды LOSO, три семени."""
    table, folds = load_protocol_table()
    labels = table[af_common.LABEL_COLUMN].to_numpy(dtype=np.int64)
    n_rows = int(labels.shape[0])
    specs = _ssl_specs()
    stage2_params, model_fn, dataset_fn, _specs = _stage2_bundle(table, labels)
    collected = {}

    with af_common.gpu_lock():
        if device.type != "cuda":
            raise RuntimeError("CUDA is required")
        configure_determinism()
        for spec in specs:
            stamp = {
                "checkpoint": spec["checkpoint"],
                "layer": int(spec["layer"]),
                "head_params": spec["head_params"],
                "n_rows": n_rows,
            }
            cached = _load_cached(spec["name"], stamp, n_rows)
            if cached is None:
                features, _meta, _fingerprint = _load_cached_features(table, spec["checkpoint"], int(spec["layer"]))
                cached = {}
                fold_aucs = {}
                for seed in SEEDS:
                    emit(f"{spec['name']} seed {seed}")
                    scores, aucs = _ssl_oof(features, labels, folds, spec["head_params"], int(seed), device)
                    cached[int(seed)] = scores
                    fold_aucs[int(seed)] = aucs
                _store_cached(spec["name"], stamp, cached, fold_aucs)
                del features
            else:
                emit(f"reuse {spec['name']} out-of-fold scores")
                fold_aucs = {
                    int(seed): read_json(_stamp_path(spec["name"]))["fold_aucs"][str(int(seed))]
                    for seed in SEEDS
                }
            summary = summarize_loso(fold_aucs)
            print_metrics(spec["name"], summary["view"])
            collected[spec["name"]] = {"scores": cached, "summary": summary, "spec": spec}

        stamp = {"params": stage2_params, "n_rows": n_rows, "replay": "stage2.loop.fit_fold"}
        cached = _load_cached(BASELINE_STAGE_NAME, stamp, n_rows)
        if cached is None:
            cached = {}
            fold_aucs = {}
            for seed in SEEDS:
                emit(f"{BASELINE_STAGE_NAME} seed {seed}")
                scores, aucs = _stage2_oof(model_fn, dataset_fn, folds, labels, int(seed), stage2_params, device)
                cached[int(seed)] = scores
                fold_aucs[int(seed)] = aucs
            _store_cached(BASELINE_STAGE_NAME, stamp, cached, fold_aucs)
        else:
            emit(f"reuse {BASELINE_STAGE_NAME} out-of-fold scores")
            fold_aucs = {
                int(seed): read_json(_stamp_path(BASELINE_STAGE_NAME))["fold_aucs"][str(int(seed))]
                for seed in SEEDS
            }
        summary = summarize_loso(fold_aucs)
        print_metrics(BASELINE_STAGE_NAME, summary["view"])
        collected[BASELINE_STAGE_NAME] = {
            "scores": cached,
            "summary": summary,
            "spec": {"name": BASELINE_STAGE_NAME, "head_params": stage2_params},
        }

    payload = {
        "code_version": int(SSL_STAGES_CODE_VERSION),
        "models": {
            name: {"view": item["summary"]["view"], "spec": item["spec"]}
            for name, item in collected.items()
        }
    }
    write_json(_dir() / PARTIAL_SEEDS_NAME, payload)
    return payload


def _scores_from_disk() -> dict[str, dict[int, np.ndarray]]:
    names = (XLSR_SINGLE_NAME, WAVLM_SINGLE_NAME, BASELINE_STAGE_NAME)
    loaded = {}
    for name in names:
        by_seed = {}
        for seed in SEEDS:
            path = _cache_path(name, int(seed))
            if not path.is_file():
                raise FileNotFoundError(f"missing out-of-fold scores {path}; run collect_single_models first")
            by_seed[int(seed)] = np.load(path)
        loaded[name] = by_seed
    return loaded


def evaluate_ensemble() -> dict:
    """Равное среднее рангов внутри каждого валидационного фолда LOSO. Вес не подбирается."""
    note_no_optuna()
    table, folds = load_protocol_table()
    labels = table[af_common.LABEL_COLUMN].to_numpy(dtype=np.int64)
    scores = _scores_from_disk()
    names = (XLSR_SINGLE_NAME, WAVLM_SINGLE_NAME, BASELINE_STAGE_NAME)
    per_model = {}
    ensemble_by_seed = {}
    for name in names:
        fold_aucs = {}
        for seed in SEEDS:
            seed_folds = {}
            for fold in folds:
                column = scores[name][int(seed)][fold.val_index]
                seed_folds[str(fold.fold_id)] = af_common.binary_auc(labels[fold.val_index], column)
            fold_aucs[int(seed)] = seed_folds
        per_model[name] = summarize_loso(fold_aucs)
        print_metrics(name, per_model[name]["view"])

    for seed in SEEDS:
        seed_folds = {}
        for fold in folds:
            columns = [scores[name][int(seed)][fold.val_index] for name in names]
            fused = equal_rank_average(columns)
            seed_folds[str(fold.fold_id)] = af_common.binary_auc(labels[fold.val_index], fused)
        ensemble_by_seed[int(seed)] = seed_folds
    ensemble = summarize_loso(ensemble_by_seed)
    print_metrics(ENSEMBLE_NAME, ensemble["view"])

    best_name = max(
        names,
        key=lambda name: (
            float(per_model[name]["view"]["mean_auc"]),
            float(per_model[name]["view"]["worst_auc"]),
        ),
    )
    wins = fold_win_report(ensemble["view"]["fold_aucs"], per_model[best_name]["view"]["fold_aucs"])
    print_fold_wins(wins, best_name)
    payload = {
        "code_version": int(SSL_STAGES_CODE_VERSION),
        "fusion": FUSION_NAME,
        "fusion_weights_fit": False,
        "models": {name: per_model[name]["view"] for name in names},
        "ensemble": ensemble["view"],
        "best_single": best_name,
        "fold_wins": wins,
        "test_used_for_selection": False,
    }
    write_json(_dir() / "ensemble.json", payload)
    return payload


def decide() -> dict:
    payload = read_json(_dir() / "ensemble.json")
    best_name = str(payload["best_single"])
    report = decide_against(best_name, ENSEMBLE_NAME, payload["models"][best_name], payload["ensemble"])
    print_decision(report)
    print_fold_wins(payload["fold_wins"], best_name)
    if report["decision"] == DECISION_ACCEPT:
        emit("Ансамбль принят. Файл сдачи: равное среднее рангов трёх моделей.")
    else:
        emit(f"Ансамбль отклонён. Файл сдачи: лучшая одиночная модель {best_name}.")
    result = {
        "stage": int(STAGE_ENSEMBLE),
        "code_version": int(SSL_STAGES_CODE_VERSION),
        "status": "complete",
        "baseline_stage": BASELINE_STAGE_NAME,
        "best_single": best_name,
        "models": payload["models"],
        "ensemble": payload["ensemble"],
        "fold_wins": payload["fold_wins"],
        "decision": report,
        "fusion": FUSION_NAME,
        "fusion_weights_fit": False,
        "optuna_trials_used": 0,
        "optuna_budget_not_spent": int(N_OPTUNA_TRIALS),
        "test_used_for_selection": False,
    }
    write_json(_dir() / RESULT_NAME, result)
    return result


def _ssl_test_scores(spec: dict, seed: int, device, features, test_features) -> np.ndarray:
    labels_table, _folds = load_protocol_table()
    labels = labels_table[af_common.LABEL_COLUMN].to_numpy(dtype=np.int64)
    train_index = np.arange(len(labels))
    model, scaler = fit_linear(features, labels, train_index, spec["head_params"], int(seed), device)
    scores = predict_linear(model, scaler, test_features, device)
    del model
    return scores


def _stage2_test_scores(seed: int, device, ids, model_fn, dataset_fn, labels) -> np.ndarray:
    from stage2.features import SpecFrontend
    from stage2.submit import _test_specs

    params = stage2_record()["best_params"]
    frontend = SpecFrontend(
        str(params["frontend"]),
        int(params["n_bins"]),
        int(params["n_fft"]),
        int(params["hop"]),
    )
    test_specs = _test_specs(frontend, ids, device)
    frontend.cpu()
    test_data = torch.from_numpy(np.ascontiguousarray(test_specs, dtype=np.float32)).unsqueeze(1)
    test_loader = DataLoader(
        torch.utils.data.TensorDataset(test_data),
        batch_size=int(params["batch_size"]),
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=bool(PIN_MEMORY_ON_CUDA),
    )
    state = fit_full(
        model_fn,
        dataset_fn,
        np.arange(len(labels), dtype=np.int64),
        labels,
        seed=int(seed),
        lr=float(params["lr"]),
        weight_decay=float(params["weight_decay"]),
        max_epochs=int(params["max_epochs"]),
        batch_size=int(params["batch_size"]),
        device=device,
        seed_fn=af_common.set_global_seed,
    )
    model = model_fn().to(device)
    model.load_state_dict(state)
    scores = collect_scores(model, test_loader, device, device.type == "cuda")
    model.cpu()
    return scores


def write_submission(device: torch.device) -> dict:
    """Записать по одному файлу сдачи на семя для принятой системы.

    Отклонённый ансамбль: лучшая одиночная модель. Принятый: равное среднее
    рангов внутри файла теста, без меток и без подбора веса.
    """
    result_path = _dir() / RESULT_NAME
    if not result_path.is_file():
        raise RuntimeError("call decide() before write_submission(); the test set cannot choose the model")
    result = read_json(result_path)
    accepted = result["decision"]["decision"] == DECISION_ACCEPT
    best_name = str(result["best_single"])
    ids = submission_ids()
    names = [name for name, _path in ids]
    table, _folds = load_protocol_table()
    labels = table[af_common.LABEL_COLUMN].to_numpy(dtype=np.int64)
    specs = {spec["name"]: spec for spec in _ssl_specs()}
    written = []

    with af_common.gpu_lock():
        if device.type != "cuda":
            raise RuntimeError("CUDA is required")
        configure_determinism()
        from stage4.submit import _encode_test, _read_test_cache

        needed = set(specs) if accepted else ({best_name} & set(specs))
        features = {}
        test_features = {}
        for name in needed:
            spec = specs[name]
            train_features, meta, _fingerprint = _load_cached_features(table, spec["checkpoint"], int(spec["layer"]))
            cached_test = _read_test_cache(spec["checkpoint"], int(spec["layer"]), names)
            if cached_test is None:
                cached_test = _encode_test(
                    spec["checkpoint"], int(spec["layer"]), float(meta["max_audio_seconds"]), ids, device
                )
            features[name] = train_features
            test_features[name] = cached_test

        stage2_ready = None
        if accepted or best_name == BASELINE_STAGE_NAME:
            stage2_ready = _stage2_bundle(table, labels)

        for seed in SEEDS:
            columns = {}
            if XLSR_SINGLE_NAME in needed:
                columns[XLSR_SINGLE_NAME] = _ssl_test_scores(
                    specs[XLSR_SINGLE_NAME], int(seed), device,
                    features[XLSR_SINGLE_NAME], test_features[XLSR_SINGLE_NAME],
                )
            if WAVLM_SINGLE_NAME in needed:
                columns[WAVLM_SINGLE_NAME] = _ssl_test_scores(
                    specs[WAVLM_SINGLE_NAME], int(seed), device,
                    features[WAVLM_SINGLE_NAME], test_features[WAVLM_SINGLE_NAME],
                )
            if accepted or best_name == BASELINE_STAGE_NAME:
                params, model_fn, dataset_fn, _spec_array = stage2_ready
                del params
                columns[BASELINE_STAGE_NAME] = _stage2_test_scores(
                    int(seed), device, ids, model_fn, dataset_fn, labels
                )
            if accepted:
                scores = equal_rank_average(
                    [columns[XLSR_SINGLE_NAME], columns[WAVLM_SINGLE_NAME], columns[BASELINE_STAGE_NAME]]
                )
                system = ENSEMBLE_NAME
                decision = DECISION_ACCEPT
            else:
                scores = columns[best_name]
                system = best_name
                decision = "REJECT"
            destination = af_common.submission_path(int(seed))
            write_score_csv(destination, ids, scores)
            write_sidecar(
                int(seed),
                {
                    "stage": int(STAGE_ENSEMBLE),
                    "decision": decision,
                    "system": system,
                    "fusion": FUSION_NAME if accepted else None,
                    "fusion_weights_fit": False,
                    "best_single": best_name,
                    "note": "No test metric and no blend weight was fit on the competition test set.",
                },
            )
            written.append(str(destination))
            emit(f"wrote {destination} system={system}")

    stamp = {
        "stage": int(STAGE_ENSEMBLE),
        "decision": DECISION_ACCEPT if accepted else "REJECT",
        "system": ENSEMBLE_NAME if accepted else best_name,
        "paths": written,
        "seeds": [int(seed) for seed in SEEDS],
    }
    write_json(_dir() / SUBMISSION_STAMP_NAME, stamp)
    return stamp
