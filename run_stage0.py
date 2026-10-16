"""Этап 0: логистическая регрессия на дешёвых акустических признаках.

Диагностика. Модель сдачи не обучается, публичный тест не читается.
Границы поиска в af_common.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import af_common
from af_common import (
    AUC_MATCH_ATOL,
    BONAFIDE_SYSTEM_ID,
    CHANCE_AUC,
    LABEL_COLUMN,
    LABEL_SPOOF,
    MIN_MEAN_AUC_GAIN,
    N_OPTUNA_TRIALS,
    SEEDS,
    SHORTCUT_LEAK_AUC,
    STAGE0_AMPLITUDE_DB_FACTOR,
    STAGE0_CLASS_WEIGHT_POLICIES,
    STAGE0_DIR,
    STAGE0_FEATURE_NAMES,
    STAGE0_FRAME_SECONDS,
    STAGE0_LOGREG_C_HIGH,
    STAGE0_LOGREG_C_LOW,
    STAGE0_LOGREG_MAX_ITER,
    STAGE0_LOGREG_SOLVERS,
    STAGE0_LOGREG_TOL,
    STAGE0_LOUDNESS_EPS,
    STAGE0_METRICS_PATH,
    STAGE0_OBJECTIVE_ATOL,
    STAGE0_OPTUNA_SAMPLER_SEED,
    STAGE0_SCALE_FEATURES,
    STAGE0_SILENCE_THRESHOLD_HIGH,
    STAGE0_SILENCE_THRESHOLD_LOW,
    STAGE0_VERDICT_PATH,
    SYNTHESIZER_COLUMN,
    TRIMMED_CACHE_DIR,
    UTTERANCE_COLUMN,
    Fold,
    binary_auc,
    class_weight_arg,
    is_shortcut_leak,
    iter_loso_folds,
    load_loso_table,
    mean_fold_aucs,
    repo_relative,
    resample_audio,
    set_global_seed,
    summarize_fold_aucs,
)
from jsonutil import to_jsonable, write_json

CACHE_NATIVE_PATH = STAGE0_DIR / "acoustic_cache_native.npz"
CANONICAL_SAMPLE_RATE_COLUMN = "canonical_sample_rate"
SAMPLE_RATE_FEATURE = "sample_rate"
METADATA_FEATURES = frozenset((SAMPLE_RATE_FEATURE,))


def _check_auc_matches_sklearn() -> None:
    labels = np.array(
        [af_common.LABEL_BONAFIDE, af_common.LABEL_BONAFIDE, LABEL_SPOOF, LABEL_SPOOF]
    )
    scores = np.array([0.1, 0.4, 0.35, 0.8])
    got = binary_auc(labels, scores)
    reference = float(roc_auc_score(labels, scores))
    if abs(got - reference) > AUC_MATCH_ATOL:
        raise RuntimeError(f"binary_auc {got} != sklearn {reference}")
    tied_labels = np.array(
        [af_common.LABEL_BONAFIDE, LABEL_SPOOF, af_common.LABEL_BONAFIDE, LABEL_SPOOF, LABEL_SPOOF]
    )
    tied_scores = np.array([0.2, 0.2, 0.5, 0.5, 0.9])
    got_ties = binary_auc(tied_labels, tied_scores)
    reference_ties = float(roc_auc_score(tied_labels, tied_scores))
    if abs(got_ties - reference_ties) > AUC_MATCH_ATOL:
        raise RuntimeError(f"tie AUC {got_ties} != sklearn {reference_ties}")


def _check_accept_stage() -> None:
    previous = {"mean_auc": CHANCE_AUC, "worst_auc": CHANCE_AUC}
    kept = {
        "mean_auc": CHANCE_AUC + MIN_MEAN_AUC_GAIN,
        "worst_auc": CHANCE_AUC,
    }
    worse_tail = {
        "mean_auc": CHANCE_AUC + MIN_MEAN_AUC_GAIN,
        "worst_auc": CHANCE_AUC - MIN_MEAN_AUC_GAIN,
    }
    flat_mean = {
        "mean_auc": CHANCE_AUC,
        "worst_auc": CHANCE_AUC + MIN_MEAN_AUC_GAIN,
    }
    if not af_common.accept_stage(previous, kept):
        raise RuntimeError("accept_stage rejected a gain that meets the rule")
    if af_common.accept_stage(previous, worse_tail):
        raise RuntimeError("accept_stage kept a stage whose worst fold fell")
    if af_common.accept_stage(previous, flat_mean):
        raise RuntimeError("accept_stage kept a stage without the required mean gain")


def _analyze_audio(audio: np.ndarray, sample_rate: int) -> tuple[float, float, int, np.ndarray]:
    sample_rate = int(sample_rate)
    n_samples = int(audio.shape[0])
    if sample_rate <= 0 or n_samples == 0:
        frame = np.zeros(1, dtype=np.float32)
        loudness = float(STAGE0_AMPLITUDE_DB_FACTOR * np.log10(STAGE0_LOUDNESS_EPS))
        return 0.0, loudness, sample_rate, frame
    power = float(np.dot(audio, audio) / n_samples)
    rms = float(np.sqrt(power))
    loudness = float(STAGE0_AMPLITUDE_DB_FACTOR * np.log10(rms + STAGE0_LOUDNESS_EPS))
    duration = float(n_samples) / float(sample_rate)
    frame_length = max(int(round(sample_rate * STAGE0_FRAME_SECONDS)), 1)
    n_frames = n_samples // frame_length
    if n_frames < 1:
        frame_rms = np.array([rms], dtype=np.float32)
    else:
        used = audio[: n_frames * frame_length]
        frames = used.reshape(n_frames, frame_length)
        frame_rms = np.sqrt(np.mean(frames * frames, axis=1)).astype(np.float32, copy=False)
    return duration, loudness, sample_rate, frame_rms


def _read_one(path: str, target_sample_rate: int | None) -> tuple[float, float, int, np.ndarray]:
    import soundfile as sf

    audio, sample_rate = sf.read(path, always_2d=False, dtype="float32")
    sample_rate = int(sample_rate)
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1, dtype=np.float32)
    if target_sample_rate is not None and sample_rate != int(target_sample_rate) and audio.size:
        audio = resample_audio(audio, sample_rate, int(target_sample_rate))
        sample_rate = int(target_sample_rate)
    return _analyze_audio(audio, sample_rate)


def _trim_edges(audio: np.ndarray, sample_rate: int, threshold: float) -> tuple[np.ndarray, int, int]:
    """Обрезать края, где RMS ниже выбранного порога.

    Сетка кадров совпадает с признаком тишины этапа 0.
    Если ни один кадр не проходит порог, файл оставляют целиком.
    """
    n_samples = int(audio.shape[0])
    frame_length = max(int(round(int(sample_rate) * STAGE0_FRAME_SECONDS)), 1)
    n_frames = n_samples // frame_length
    if n_samples == 0 or n_frames < 1:
        return audio, 0, n_samples
    frames = audio[: n_frames * frame_length].reshape(n_frames, frame_length)
    frame_rms = np.sqrt(np.mean(frames * frames, axis=1))
    voiced = np.flatnonzero(frame_rms >= threshold)
    if voiced.size == 0:
        return audio, 0, n_samples
    start = int(voiced[0]) * frame_length
    end = (int(voiced[-1]) + 1) * frame_length
    if int(voiced[-1]) == n_frames - 1:
        end = n_samples
    if end <= start:
        return audio, 0, n_samples
    return audio[start:end], start, end


def _read_batch(payload: tuple[list[str], int | None]) -> list[tuple[float, float, int, np.ndarray]]:
    paths, target_sample_rate = payload
    rows = []
    for path in paths:
        try:
            rows.append(_read_one(path, target_sample_rate))
        except Exception as exc:
            raise RuntimeError(f"failed to read {path}") from exc
    return rows


class AcousticCache:
    def __init__(
        self,
        duration: np.ndarray,
        loudness: np.ndarray,
        sample_rate: np.ndarray,
        frame_offsets: np.ndarray,
        frame_rms: np.ndarray,
    ) -> None:
        self.duration = duration
        self.loudness = loudness
        self.sample_rate = sample_rate
        self.frame_offsets = frame_offsets
        self.frame_rms = frame_rms

    def silence_ratio(self, threshold: float) -> np.ndarray:
        n_rows = int(self.duration.shape[0])
        silence = np.empty(n_rows, dtype=np.float64)
        for index in range(n_rows):
            start = int(self.frame_offsets[index])
            stop = int(self.frame_offsets[index + 1])
            silence[index] = float(np.mean(self.frame_rms[start:stop] < threshold))
        return silence

    def matrix(self, threshold: float, sample_rate: np.ndarray | None = None) -> np.ndarray:
        rate = self.sample_rate if sample_rate is None else sample_rate
        columns = {
            "duration": self.duration,
            "silence_ratio": self.silence_ratio(threshold),
            "loudness": self.loudness,
            "sample_rate": np.asarray(rate, dtype=np.float64),
        }
        return np.column_stack([columns[name] for name in STAGE0_FEATURE_NAMES])


def _batches(paths: list[str], workers: int) -> list[list[str]]:
    workers = max(int(workers), 1)
    if len(paths) == 0:
        return []
    batch_size = (len(paths) + workers - 1) // workers
    return [paths[start : start + batch_size] for start in range(0, len(paths), batch_size)]


def _build_cache(paths: list[str], target_sample_rate: int | None) -> AcousticCache:
    workers = os.cpu_count() or 1
    batches = _batches(paths, workers)
    duration = np.empty(len(paths), dtype=np.float64)
    loudness = np.empty(len(paths), dtype=np.float64)
    sample_rate = np.empty(len(paths), dtype=np.int64)
    frames: list[np.ndarray | None] = [None] * len(paths)
    cursor = 0
    payloads = [(batch, target_sample_rate) for batch in batches]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for batch, rows in zip(batches, pool.map(_read_batch, payloads)):
            print(f"read {cursor + len(batch)}/{len(paths)}", flush=True)
            for row in rows:
                duration[cursor], loudness[cursor], sample_rate[cursor], frames[cursor] = row
                cursor += 1
    if cursor != len(paths):
        raise RuntimeError("feature read did not cover every labeled file")
    offsets = np.zeros(len(paths) + 1, dtype=np.int64)
    pieces: list[np.ndarray] = []
    for index, frame in enumerate(frames):
        if frame is None or frame.size == 0:
            raise RuntimeError(f"missing frames for row {index}")
        pieces.append(frame)
        offsets[index + 1] = offsets[index] + int(frame.size)
    flat = np.concatenate(pieces).astype(np.float32, copy=False)
    return AcousticCache(duration, loudness, sample_rate, offsets, flat)


def _cache_path(target_sample_rate: int | None) -> Path:
    if target_sample_rate is None:
        return CACHE_NATIVE_PATH
    return STAGE0_DIR / f"acoustic_cache_sr{int(target_sample_rate)}.npz"


def _save_cache(path: Path, cache: AcousticCache, utterance_ids: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        duration=cache.duration,
        loudness=cache.loudness,
        sample_rate=cache.sample_rate,
        frame_offsets=cache.frame_offsets,
        frame_rms=cache.frame_rms,
        utterance_id=np.asarray(utterance_ids),
        frame_seconds=np.array([STAGE0_FRAME_SECONDS], dtype=np.float64),
    )


def _load_cache(path: Path, utterance_ids: list[str]) -> AcousticCache | None:
    if not path.is_file():
        return None
    with np.load(path, allow_pickle=False) as payload:
        if float(payload["frame_seconds"][0]) != float(STAGE0_FRAME_SECONDS):
            return None
        cached_ids = payload["utterance_id"].astype(str).tolist()
        if cached_ids != utterance_ids:
            return None
        return AcousticCache(
            duration=np.asarray(payload["duration"], dtype=np.float64),
            loudness=np.asarray(payload["loudness"], dtype=np.float64),
            sample_rate=np.asarray(payload["sample_rate"], dtype=np.int64),
            frame_offsets=np.asarray(payload["frame_offsets"], dtype=np.int64),
            frame_rms=np.asarray(payload["frame_rms"], dtype=np.float32),
        )


def load_or_build_cache(
    table: pd.DataFrame, target_sample_rate: int | None
) -> AcousticCache:
    utterance_ids = table[UTTERANCE_COLUMN].astype(str).tolist()
    path = _cache_path(target_sample_rate)
    cached = _load_cache(path, utterance_ids)
    if cached is not None:
        print(f"loaded cache {path.name}", flush=True)
        return cached
    print(f"building cache target_sr={target_sample_rate}", flush=True)
    cache = _build_cache(table[af_common.PATH_COLUMN].astype(str).tolist(), target_sample_rate)
    _save_cache(path, cache, utterance_ids)
    return cache


def _build_logreg(params: dict, seed: int) -> Pipeline:
    steps = []
    if STAGE0_SCALE_FEATURES:
        steps.append(("scale", StandardScaler()))
    steps.append(
        (
            "clf",
            LogisticRegression(
                C=float(params["C"]),
                class_weight=class_weight_arg(str(params["class_weight"])),
                solver=str(params["solver"]),
                max_iter=STAGE0_LOGREG_MAX_ITER,
                tol=STAGE0_LOGREG_TOL,
                random_state=int(seed),
            ),
        )
    )
    return Pipeline(steps)


def _positive_scores(model: Pipeline, features: np.ndarray) -> np.ndarray:
    classifier = model.named_steps["clf"]
    positive_at = np.flatnonzero(classifier.classes_ == LABEL_SPOOF)
    if positive_at.size != 1:
        raise RuntimeError("logistic regression did not learn the spoof class")
    return model.predict_proba(features)[:, int(positive_at[0])]


def _fit_seeds(
    features: np.ndarray,
    labels: np.ndarray,
    folds: list[Fold],
    params: dict,
) -> dict[str, dict[str, float]]:
    per_seed: dict[str, dict[str, float]] = {}
    for seed in SEEDS:
        set_global_seed(seed)
        fold_aucs: dict[str, float] = {}
        for fold in folds:
            model = _build_logreg(params, seed)
            model.fit(features[fold.train_index], labels[fold.train_index])
            scores = _positive_scores(model, features[fold.val_index])
            fold_aucs[fold.fold_id] = binary_auc(labels[fold.val_index], scores)
        per_seed[str(seed)] = fold_aucs
    return per_seed


def _seed_averaged_fold_aucs(per_seed: dict[str, dict[str, float]]) -> dict[str, float]:
    return mean_fold_aucs(per_seed[str(seed)] for seed in SEEDS)


def _objective_value(per_seed: dict[str, dict[str, float]]) -> float:
    seed_means = [
        summarize_fold_aucs(per_seed[str(seed)])["mean_auc"] for seed in SEEDS
    ]
    return float(np.mean(seed_means))


def _suggest(trial: optuna.Trial) -> dict:
    return {
        "silence_threshold": trial.suggest_float(
            "silence_threshold",
            STAGE0_SILENCE_THRESHOLD_LOW,
            STAGE0_SILENCE_THRESHOLD_HIGH,
            log=True,
        ),
        "C": trial.suggest_float(
            "C",
            STAGE0_LOGREG_C_LOW,
            STAGE0_LOGREG_C_HIGH,
            log=True,
        ),
        "class_weight": trial.suggest_categorical(
            "class_weight", list(STAGE0_CLASS_WEIGHT_POLICIES)
        ),
        "solver": trial.suggest_categorical("solver", list(STAGE0_LOGREG_SOLVERS)),
    }


def run_search(
    cache: AcousticCache,
    labels: np.ndarray,
    folds: list[Fold],
    sample_rate: np.ndarray | None = None,
) -> dict:
    def objective(trial: optuna.Trial) -> float:
        params = _suggest(trial)
        features = cache.matrix(float(params["silence_threshold"]), sample_rate)
        per_seed = _fit_seeds(features, labels, folds, params)
        value = _objective_value(per_seed)
        trial.set_user_attr("worst_auc", summarize_fold_aucs(_seed_averaged_fold_aucs(per_seed))["worst_auc"])
        return value

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=STAGE0_OPTUNA_SAMPLER_SEED),
    )
    def _on_trial(study, trial) -> None:
        print(
            f"trial {trial.number} mean_auc={trial.value} params={trial.params}",
            flush=True,
        )

    study.optimize(
        objective,
        n_trials=N_OPTUNA_TRIALS,
        show_progress_bar=False,
        callbacks=[_on_trial],
    )
    best = dict(study.best_params)
    features = cache.matrix(float(best["silence_threshold"]), sample_rate)
    per_seed = _fit_seeds(features, labels, folds, best)
    averaged = _seed_averaged_fold_aucs(per_seed)
    summary = summarize_fold_aucs(averaged)
    objective_gap = abs(summary["mean_auc"] - _objective_value(per_seed))
    if objective_gap > STAGE0_OBJECTIVE_ATOL:
        raise RuntimeError("seed-averaged mean AUC does not match the Optuna objective")
    if abs(summary["mean_auc"] - float(study.best_value)) > AUC_MATCH_ATOL:
        raise RuntimeError(
            f"recomputed mean AUC {summary['mean_auc']} != best trial {study.best_value}"
        )
    univariate = {}
    for name, column in zip(STAGE0_FEATURE_NAMES, features.T):
        univariate[name] = _univariate_loso(labels, column, folds)
    trials = [
        {
            "number": int(trial.number),
            "value": None if trial.value is None else float(trial.value),
            "params": {key: to_jsonable(val) for key, val in trial.params.items()},
            "state": trial.state.name,
        }
        for trial in study.trials
    ]
    return {
        **summary,
        "per_seed": {
            seed: {fold_id: float(auc) for fold_id, auc in fold_aucs.items()}
            for seed, fold_aucs in per_seed.items()
        },
        "best_params": {key: to_jsonable(val) for key, val in best.items()},
        "feature_mean_auc": {
            name: float(payload["mean_auc"]) for name, payload in univariate.items()
        },
        "feature_summaries": univariate,
        "trials": trials,
        "best_trial_value": float(study.best_value),
    }


def _univariate_loso(
    labels: np.ndarray, values: np.ndarray, folds: list[Fold]
) -> dict:
    fold_aucs: dict[str, float] = {}
    for fold in folds:
        train_values = values[fold.train_index]
        val_values = values[fold.val_index]
        if float(np.nanstd(train_values)) == 0.0 or float(np.nanstd(val_values)) == 0.0:
            fold_aucs[fold.fold_id] = CHANCE_AUC
            continue
        scores = val_values
        if binary_auc(labels[fold.train_index], train_values) < CHANCE_AUC:
            scores = -val_values
        fold_aucs[fold.fold_id] = binary_auc(labels[fold.val_index], scores)
    return summarize_fold_aucs(fold_aucs)


def _mode_sample_rate(sample_rate: np.ndarray) -> int:
    values, counts = np.unique(sample_rate, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _carriers(feature_mean_auc: dict[str, float]) -> list[str]:
    ranked = sorted(feature_mean_auc, key=feature_mean_auc.get, reverse=True)
    return [name for name in ranked if feature_mean_auc[name] > SHORTCUT_LEAK_AUC]


def _write_cleaned_manifest(table: pd.DataFrame, canonical_sample_rate: int) -> Path:
    manifest = table.copy()
    manifest[CANONICAL_SAMPLE_RATE_COLUMN] = int(canonical_sample_rate)
    path = STAGE0_DIR / "cleaned_manifest.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(path, index=False)
    reloaded = load_loso_table(path)
    if len(reloaded) != len(table):
        raise RuntimeError("cleaned manifest row count changed")
    if reloaded[UTTERANCE_COLUMN].tolist() != table[UTTERANCE_COLUMN].tolist():
        raise RuntimeError("cleaned manifest reordered or dropped utterances")
    return path


def _worst_fold(summary: dict) -> str:
    fold_aucs = summary["fold_aucs"]
    return min(fold_aucs, key=fold_aucs.get)


def _format_feature_aucs(feature_mean_auc: dict[str, float]) -> str:
    parts = [f"{name}={feature_mean_auc[name]:.4f}" for name in STAGE0_FEATURE_NAMES]
    return ", ".join(parts)


def main() -> None:
    _check_auc_matches_sklearn()
    _check_accept_stage()
    table = load_loso_table(af_common.TRAIN_PROTOCOL_PATH)
    folds = iter_loso_folds(table)
    labels = table[LABEL_COLUMN].to_numpy(dtype=np.int64)
    print(
        f"rows={len(table)} folds={len(folds)} synthesizers={[fold.fold_id for fold in folds]}",
        flush=True,
    )
    native = load_or_build_cache(table, None)
    unique_rates = sorted(int(value) for value in np.unique(native.sample_rate))
    print(f"sample_rates={unique_rates}", flush=True)
    screening = run_search(native, labels, folds)
    print(
        f"screen mean={screening['mean_auc']:.4f} worst={screening['worst_auc']:.4f} "
        f"features={screening['feature_mean_auc']}",
        flush=True,
    )

    final = screening
    manifest_path = af_common.TRAIN_PROTOCOL_PATH
    cleaning_notes: list[str] = []
    carriers = _carriers(screening["feature_mean_auc"])
    metadata_carriers = [name for name in carriers if name in METADATA_FEATURES]
    content_carriers = [name for name in carriers if name not in METADATA_FEATURES]

    if metadata_carriers:
        canonical = _mode_sample_rate(native.sample_rate)
        print(f"metadata shortcut via {metadata_carriers}; forcing sr={canonical}", flush=True)
        forced_rate = np.full(len(table), canonical, dtype=np.float64)
        if unique_rates == [canonical]:
            final = run_search(native, labels, folds, sample_rate=forced_rate)
            cleaning_notes.append(
                f"Native sample rate is already a single value ({canonical} Hz). "
                "Re-ran stage 0 with the sample-rate feature replaced by that constant "
                "so the column cannot carry a shortcut."
            )
        else:
            resampled = load_or_build_cache(table, canonical)
            final = run_search(resampled, labels, folds)
            cleaning_notes.append(
                f"Sample rate varied across files ({unique_rates}). "
                f"Re-read every train file through a resampler locked to the modal rate "
                f"{canonical} Hz and re-ran stage 0 on those waveforms. "
                "Original audio files were not modified."
            )
        manifest_path = _write_cleaned_manifest(table, canonical)
        cleaning_notes.append(
            f"Derived manifest records {CANONICAL_SAMPLE_RATE_COLUMN}={canonical}. "
            "Later stages must resample to that rate before training."
        )
        content_carriers = _carriers(final["feature_mean_auc"])
        content_carriers = [name for name in content_carriers if name not in METADATA_FEATURES]

    if len(unique_rates) == 1 and not metadata_carriers:
        cleaning_notes.append(
            f"Observed sample rate is {unique_rates[0]} Hz on every labeled file. "
            "No resample filter and no derived manifest were written."
        )

    if is_shortcut_leak(final["mean_auc"]) and content_carriers:
        cleaning_notes.append(
            "The remaining gap is carried by audio-content features "
            f"({', '.join(content_carriers)}): duration, silence ratio, and loudness "
            "are measured from the waveform, not from a metadata field that can be dropped. "
            "Original audio was kept."
        )
    elif is_shortcut_leak(final["mean_auc"]):
        cleaning_notes.append(
            "Mean AUC stays above the shortcut threshold after the metadata filter, "
            "but no single content feature exceeds that threshold on its own."
        )
    elif not metadata_carriers:
        cleaning_notes.append(
            "Mean AUC is not above the shortcut threshold. "
            "No metadata filter was required. "
            "data_manifest is the original labeled train protocol."
        )
    else:
        cleaning_notes.append(
            "After removing the sample-rate shortcut, mean AUC is not above the shortcut threshold."
        )

    if not metadata_carriers:
        manifest_path = af_common.TRAIN_PROTOCOL_PATH

    worst_fold = _worst_fold(final)
    notes = " ".join(
        [
            (
                f"Synthesizer column is '{SYNTHESIZER_COLUMN}', the 4th whitespace field of "
                f"{repo_relative(af_common.TRAIN_PROTOCOL_PATH)} (ASVspoof attack id). "
                f"Bonafide sentinel is '{BONAFIDE_SYSTEM_ID}', which is not a fold. "
                f"{len(folds)} LOSO folds ({', '.join(fold.fold_id for fold in folds)}), "
                "sorted by synthesizer id. Bonafide rows are split into contiguous even chunks "
                "in protocol order so each validation fold has both classes and no row is in "
                "both train and validation."
            ),
            (
                "Stage 0 fits logistic regression on duration, silence ratio, loudness, and "
                f"sample rate. Optuna ({N_OPTUNA_TRIALS} trials) maximizes the mean over seeds "
                f"{list(SEEDS)} of the mean LOSO AUC. "
                f"Final mean AUC {final['mean_auc']:.4f}, worst fold {worst_fold} "
                f"{final['worst_auc']:.4f}, std {final['std_auc']:.4f}, "
                f"bootstrap CI [{final['ci_low']:.4f}, {final['ci_high']:.4f}] "
                f"at confidence {af_common.BOOTSTRAP_CONFIDENCE}. "
                f"Shortcut threshold is {SHORTCUT_LEAK_AUC}. "
                f"Univariate LOSO mean AUC: {_format_feature_aucs(final['feature_mean_auc'])}. "
                f"Best params: {json.dumps(final['best_params'])}."
            ),
            *cleaning_notes,
            "The public test split was not used. This stage is a gate, not the submission model.",
        ]
    )

    metrics = {
        "mean_auc": final["mean_auc"],
        "worst_auc": final["worst_auc"],
        "std_auc": final["std_auc"],
        "ci_low": final["ci_low"],
        "ci_high": final["ci_high"],
        "fold_aucs": final["fold_aucs"],
        "per_seed": final["per_seed"],
        "best_params": final["best_params"],
        "feature_mean_auc": final["feature_mean_auc"],
        "trials": final["trials"],
        "n_folds": len(folds),
        "synthesizer_column": SYNTHESIZER_COLUMN,
        "sample_rates": unique_rates,
    }
    if metadata_carriers:
        metrics["screening"] = {
            "mean_auc": screening["mean_auc"],
            "worst_auc": screening["worst_auc"],
            "std_auc": screening["std_auc"],
            "ci_low": screening["ci_low"],
            "ci_high": screening["ci_high"],
            "fold_aucs": screening["fold_aucs"],
            "per_seed": screening["per_seed"],
            "best_params": screening["best_params"],
            "feature_mean_auc": screening["feature_mean_auc"],
        }
    verdict = {
        "leak": bool(is_shortcut_leak(final["mean_auc"])),
        "mean_auc": float(final["mean_auc"]),
        "worst_auc": float(final["worst_auc"]),
        "synthesizer_column": SYNTHESIZER_COLUMN,
        "notes": notes,
        "data_manifest": repo_relative(manifest_path),
    }
    write_json(STAGE0_METRICS_PATH, metrics)
    write_json(STAGE0_VERDICT_PATH, verdict)
    print(f"verdict leak={verdict['leak']} manifest={verdict['data_manifest']}", flush=True)


TRIMMED_WAV_DIR = TRIMMED_CACHE_DIR / "wav"
TRIMMED_FEATURE_CACHE = TRIMMED_CACHE_DIR / "acoustic_cache.npz"
TRIMMED_MANIFEST_PATH = TRIMMED_CACHE_DIR / "manifest.csv"
TRIMMED_AUDIO_SUFFIX = ".wav"


def _load_mono(path: str) -> tuple[np.ndarray, int]:
    import soundfile as sf

    audio, sample_rate = sf.read(path, always_2d=False, dtype="float32")
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1, dtype=np.float32)
    return audio, int(sample_rate)


def _trim_one(payload: tuple[str, str, float]) -> tuple[float, float, int, np.ndarray, int, int, int]:
    source, destination, threshold = payload
    audio, sample_rate = _load_mono(source)
    trimmed, start, end = _trim_edges(audio, sample_rate, threshold)
    import soundfile as sf

    sf.write(destination, trimmed, sample_rate, subtype="FLOAT")
    duration, loudness, rate, frame_rms = _analyze_audio(trimmed, sample_rate)
    return duration, loudness, rate, frame_rms, int(start), int(end), int(audio.shape[0])


def _trim_batch(
    payload: tuple[list[tuple[str, str]], float],
) -> list[tuple[float, float, int, np.ndarray, int, int, int]]:
    pairs, threshold = payload
    rows = []
    for source, destination in pairs:
        try:
            rows.append(_trim_one((source, destination, threshold)))
        except Exception as exc:
            raise RuntimeError(f"failed to trim {source}") from exc
    return rows


def _chosen_trim_threshold() -> float:
    payload = json.loads(STAGE0_METRICS_PATH.read_text(encoding="utf-8"))
    return float(payload["best_params"]["silence_threshold"])


def _load_trimmed_feature_cache(
    path: Path, utterance_ids: list[str], threshold: float
) -> tuple[AcousticCache, np.ndarray, np.ndarray, np.ndarray] | None:
    if not path.is_file():
        return None
    with np.load(path, allow_pickle=False) as payload:
        if float(payload["frame_seconds"][0]) != float(STAGE0_FRAME_SECONDS):
            return None
        if float(payload["trim_threshold"][0]) != float(threshold):
            return None
        cached_ids = payload["utterance_id"].astype(str).tolist()
        if cached_ids != utterance_ids:
            return None
        cache = AcousticCache(
            duration=np.asarray(payload["duration"], dtype=np.float64),
            loudness=np.asarray(payload["loudness"], dtype=np.float64),
            sample_rate=np.asarray(payload["sample_rate"], dtype=np.int64),
            frame_offsets=np.asarray(payload["frame_offsets"], dtype=np.int64),
            frame_rms=np.asarray(payload["frame_rms"], dtype=np.float32),
        )
        return (
            cache,
            np.asarray(payload["trim_start"], dtype=np.int64),
            np.asarray(payload["trim_end"], dtype=np.int64),
            np.asarray(payload["source_samples"], dtype=np.int64),
        )


def _save_trimmed_feature_cache(
    path: Path,
    cache: AcousticCache,
    utterance_ids: list[str],
    threshold: float,
    trim_start: np.ndarray,
    trim_end: np.ndarray,
    source_samples: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        duration=cache.duration,
        loudness=cache.loudness,
        sample_rate=cache.sample_rate,
        frame_offsets=cache.frame_offsets,
        frame_rms=cache.frame_rms,
        utterance_id=np.asarray(utterance_ids),
        frame_seconds=np.array([STAGE0_FRAME_SECONDS], dtype=np.float64),
        trim_threshold=np.array([threshold], dtype=np.float64),
        trim_start=trim_start,
        trim_end=trim_end,
        source_samples=source_samples,
    )


def _build_trimmed_cache(
    table: pd.DataFrame, threshold: float
) -> tuple[AcousticCache, np.ndarray, np.ndarray, np.ndarray]:
    utterance_ids = table[UTTERANCE_COLUMN].astype(str).tolist()
    cached = _load_trimmed_feature_cache(TRIMMED_FEATURE_CACHE, utterance_ids, threshold)
    wav_ready = TRIMMED_WAV_DIR.is_dir() and len(list(TRIMMED_WAV_DIR.glob(f"*{TRIMMED_AUDIO_SUFFIX}"))) == len(table)
    if cached is not None and wav_ready:
        print(f"loaded trimmed cache threshold={threshold}", flush=True)
        return cached
    TRIMMED_WAV_DIR.mkdir(parents=True, exist_ok=True)
    sources = table[af_common.PATH_COLUMN].astype(str).tolist()
    destinations = [
        str(TRIMMED_WAV_DIR / f"{utterance_id}{TRIMMED_AUDIO_SUFFIX}")
        for utterance_id in utterance_ids
    ]
    workers = os.cpu_count() or 1
    pair_batches = _batches(list(zip(sources, destinations)), workers)
    duration = np.empty(len(table), dtype=np.float64)
    loudness = np.empty(len(table), dtype=np.float64)
    sample_rate = np.empty(len(table), dtype=np.int64)
    trim_start = np.empty(len(table), dtype=np.int64)
    trim_end = np.empty(len(table), dtype=np.int64)
    source_samples = np.empty(len(table), dtype=np.int64)
    frames: list[np.ndarray | None] = [None] * len(table)
    cursor = 0
    payloads = [(batch, threshold) for batch in pair_batches]
    print(f"trimming edges threshold={threshold}", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for batch, rows in zip(pair_batches, pool.map(_trim_batch, payloads)):
            print(f"trimmed {cursor + len(batch)}/{len(table)}", flush=True)
            for row in rows:
                duration[cursor], loudness[cursor], sample_rate[cursor], frames[cursor] = row[:4]
                trim_start[cursor], trim_end[cursor], source_samples[cursor] = row[4:]
                cursor += 1
    if cursor != len(table):
        raise RuntimeError("trim did not cover every labeled file")
    offsets = np.zeros(len(table) + 1, dtype=np.int64)
    pieces: list[np.ndarray] = []
    for index, frame in enumerate(frames):
        if frame is None or frame.size == 0:
            raise RuntimeError(f"missing trimmed frames for row {index}")
        pieces.append(frame)
        offsets[index + 1] = offsets[index] + int(frame.size)
    cache = AcousticCache(
        duration,
        loudness,
        sample_rate,
        offsets,
        np.concatenate(pieces).astype(np.float32, copy=False),
    )
    _save_trimmed_feature_cache(
        TRIMMED_FEATURE_CACHE,
        cache,
        utterance_ids,
        threshold,
        trim_start,
        trim_end,
        source_samples,
    )
    return cache, trim_start, trim_end, source_samples


def _write_trimmed_manifest(
    table: pd.DataFrame,
    threshold: float,
    trim_start: np.ndarray,
    trim_end: np.ndarray,
) -> Path:
    manifest = table.copy()
    utterance_ids = manifest[UTTERANCE_COLUMN].astype(str)
    manifest[af_common.FILE_COLUMN] = utterance_ids.map(lambda utterance: f"{utterance}{TRIMMED_AUDIO_SUFFIX}")
    manifest[af_common.PATH_COLUMN] = utterance_ids.map(
        lambda utterance: str(TRIMMED_WAV_DIR / f"{utterance}{TRIMMED_AUDIO_SUFFIX}")
    )
    manifest["trim_start"] = trim_start
    manifest["trim_end"] = trim_end
    manifest["trim_threshold"] = float(threshold)
    TRIMMED_MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(TRIMMED_MANIFEST_PATH, index=False)
    reloaded = load_loso_table(TRIMMED_MANIFEST_PATH)
    if len(reloaded) != len(table):
        raise RuntimeError("trimmed manifest row count changed")
    if reloaded[UTTERANCE_COLUMN].tolist() != table[UTTERANCE_COLUMN].tolist():
        raise RuntimeError("trimmed manifest reordered utterances")
    return TRIMMED_MANIFEST_PATH


def main_trim_recheck() -> None:
    """Повторить этап 0 после обрезки краёв уже выбранным порогом."""
    _check_auc_matches_sklearn()
    threshold = _chosen_trim_threshold()
    untrimmed = json.loads(STAGE0_METRICS_PATH.read_text(encoding="utf-8"))
    table = load_loso_table(af_common.TRAIN_PROTOCOL_PATH)
    folds = iter_loso_folds(table)
    labels = table[LABEL_COLUMN].to_numpy(dtype=np.int64)
    cache, trim_start, trim_end, source_samples = _build_trimmed_cache(table, threshold)
    kept = np.maximum(trim_end - trim_start, 0)
    removed_fraction = 1.0 - float(np.sum(kept)) / float(np.sum(source_samples))
    print(
        f"trim threshold={threshold} removed_fraction={removed_fraction:.4f}",
        flush=True,
    )
    trimmed = run_search(cache, labels, folds)
    print(
        f"trimmed mean={trimmed['mean_auc']:.4f} worst={trimmed['worst_auc']:.4f}",
        flush=True,
    )
    manifest_path = _write_trimmed_manifest(table, threshold, trim_start, trim_end)
    leak = bool(is_shortcut_leak(trimmed["mean_auc"]))
    if leak:
        data_manifest = af_common.TRAIN_PROTOCOL_PATH
        acoustic_shortcut = True
        manifest_sentence = (
            "Leading and trailing silence were not a removable pad: the cue remains inside "
            "the speech. data_manifest stays the original labeled protocol. "
            "Original audio under Data/ was not modified."
        )
    else:
        data_manifest = manifest_path
        acoustic_shortcut = False
        manifest_sentence = (
            "Edge silence carried the shortcut. data_manifest now points at the trimmed cache. "
            "Original audio under Data/ was not modified."
        )
    worst_fold = _worst_fold(trimmed)
    notes = " ".join(
        [
            (
                f"Synthesizer column is '{SYNTHESIZER_COLUMN}'. "
                f"{len(folds)} LOSO folds, same af_common splits as the untrimmed check."
            ),
            (
                f"Re-check trimmed leading and trailing frames with RMS below the silence "
                f"threshold already chosen in the untrimmed search, {threshold}. "
                f"No new trim threshold was searched. "
                f"Fraction of samples removed: {removed_fraction:.4f}. "
                f"Untrimmed mean AUC {float(untrimmed['mean_auc']):.4f}, "
                f"worst {float(untrimmed['worst_auc']):.4f}. "
                f"Trimmed mean AUC {trimmed['mean_auc']:.4f}, worst fold {worst_fold} "
                f"{trimmed['worst_auc']:.4f}, std {trimmed['std_auc']:.4f}, "
                f"bootstrap CI [{trimmed['ci_low']:.4f}, {trimmed['ci_high']:.4f}]. "
                f"Shortcut threshold is {SHORTCUT_LEAK_AUC}. "
                f"Univariate LOSO mean AUC on the trimmed signal: "
                f"{_format_feature_aucs(trimmed['feature_mean_auc'])}. "
                f"Best params: {json.dumps(trimmed['best_params'])}."
            ),
            manifest_sentence,
            "The public test split was not used. ready_for_training is set only after this re-check.",
        ]
    )
    metrics = {
        "mean_auc": trimmed["mean_auc"],
        "worst_auc": trimmed["worst_auc"],
        "std_auc": trimmed["std_auc"],
        "ci_low": trimmed["ci_low"],
        "ci_high": trimmed["ci_high"],
        "fold_aucs": trimmed["fold_aucs"],
        "per_seed": trimmed["per_seed"],
        "best_params": trimmed["best_params"],
        "feature_mean_auc": trimmed["feature_mean_auc"],
        "trials": trimmed["trials"],
        "n_folds": len(folds),
        "synthesizer_column": SYNTHESIZER_COLUMN,
        "trim_threshold": threshold,
        "trim_cache": repo_relative(TRIMMED_CACHE_DIR),
        "removed_sample_fraction": removed_fraction,
        "untrimmed": {
            "mean_auc": untrimmed["mean_auc"],
            "worst_auc": untrimmed["worst_auc"],
            "std_auc": untrimmed["std_auc"],
            "ci_low": untrimmed["ci_low"],
            "ci_high": untrimmed["ci_high"],
            "fold_aucs": untrimmed["fold_aucs"],
            "per_seed": untrimmed["per_seed"],
            "best_params": untrimmed["best_params"],
            "feature_mean_auc": untrimmed["feature_mean_auc"],
        },
    }
    verdict = {
        "leak": leak,
        "ready_for_training": True,
        "acoustic_shortcut": acoustic_shortcut,
        "mean_auc": float(trimmed["mean_auc"]),
        "worst_auc": float(trimmed["worst_auc"]),
        "synthesizer_column": SYNTHESIZER_COLUMN,
        "notes": notes,
        "data_manifest": repo_relative(data_manifest),
    }
    write_json(STAGE0_METRICS_PATH, metrics)
    write_json(STAGE0_VERDICT_PATH, verdict)
    print(
        f"verdict leak={verdict['leak']} ready={verdict['ready_for_training']} "
        f"acoustic_shortcut={verdict['acoustic_shortcut']} manifest={verdict['data_manifest']}",
        flush=True,
    )


if __name__ == "__main__":
    import sys

    if "--trim-recheck" in sys.argv:
        main_trim_recheck()
    else:
        main()
