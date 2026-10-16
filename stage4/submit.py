"""Скоры теста для принятой головы этапа 4. Только признаки; колонка score образца не читается."""

from __future__ import annotations

import argparse
import gc
from pathlib import Path

from stage4.gpu_env import configure_process_env

configure_process_env()

import numpy as np
import torch

import af_common
from af_common import LABEL_COLUMN, SEEDS, gpu_lock, submission_path
from jsonutil import read_json, write_json
from stage4.audio import load_wave
from stage4.cache_io import cache_dir_from_pointer, cache_is_complete, read_layer
from stage4.constants import (
    HEAD_RESULT_PATH,
    LAYER_PROBE_PATH,
    METRICS_PATH,
    MIN_AUDIO_SECONDS,
    TEST_CACHE_DIR,
)
from stage4.data import load_training_table, table_fingerprint
from stage4.encode import encode_waves, is_oom, load_frozen
from stage4.head import fit_linear, predict_linear


def _kept() -> bool:
    if not METRICS_PATH.is_file():
        return False
    payload = read_json(METRICS_PATH)
    return payload.get("kept_vs_neural") is True


def _test_cache_path(checkpoint: str, layer: int) -> Path:
    slug = checkpoint.split("/")[-1]
    return TEST_CACHE_DIR / slug / f"layer_{layer:02d}.npy"


def _read_test_cache(checkpoint: str, layer: int, names: list[str]) -> np.ndarray | None:
    cache_path = _test_cache_path(checkpoint, layer)
    stamp_path = cache_path.with_suffix(".json")
    if not cache_path.is_file() or not stamp_path.is_file():
        return None
    stamp = read_json(stamp_path)
    if stamp.get("idx") != names or stamp.get("checkpoint") != checkpoint or int(stamp.get("layer", -1)) != layer:
        return None
    return np.load(cache_path)


def _encode_test(checkpoint: str, layer: int, cap_seconds: float, ids: list[tuple[str, Path]], device: torch.device) -> np.ndarray:
    extractor, model = load_frozen(checkpoint, device)
    sample_rate = int(extractor.sampling_rate)
    max_samples = int(round(cap_seconds * sample_rate))
    min_samples = max(1, int(round(MIN_AUDIO_SECONDS * sample_rate)))
    chunks: list[np.ndarray] = []
    cursor = 0
    budget = max_samples
    try:
        while cursor < len(ids):
            waves = []
            take = 0
            limit = min(len(ids), cursor + 8)
            for index in range(cursor, limit):
                wave, _cropped = load_wave(ids[index][1], sample_rate, max_samples, min_samples)
                if waves and (len(waves) + 1) * max(len(item) for item in waves + [wave]) > budget:
                    break
                waves.append(wave)
                take += 1
            try:
                pooled = encode_waves(extractor, model, waves, device)
            except Exception as exc:
                if not is_oom(exc) or take <= 1:
                    raise
                torch.cuda.empty_cache()
                budget = max(1, budget // 2)
                print(f"test OOM, budget_samples -> {budget}", flush=True)
                continue
            if layer >= len(pooled):
                raise RuntimeError(f"layer {layer} missing from a {len(pooled)}-layer forward")
            chunks.append(pooled[layer])
            cursor += take
            print(f"test features {cursor}/{len(ids)}", flush=True)
            del pooled, waves
    finally:
        del model
        del extractor
        gc.collect()
        torch.cuda.empty_cache()
    features = np.concatenate(chunks, axis=0)
    if features.shape[0] != len(ids):
        raise RuntimeError("test feature count does not match the submission ids")
    cache_path = _test_cache_path(checkpoint, layer)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, features)
    write_json(
        cache_path.with_suffix(".json"),
        {
            "checkpoint": checkpoint,
            "layer": layer,
            "max_audio_seconds": cap_seconds,
            "idx": [name for name, _path in ids],
        },
    )
    return features


def write_submissions(manifest: str) -> None:
    if not _kept():
        print("submission skipped: kept_vs_neural is not true", flush=True)
        return
    head = read_json(HEAD_RESULT_PATH)
    probe = read_json(LAYER_PROBE_PATH)
    table = load_training_table(manifest)
    fingerprint = table_fingerprint(table)
    if head.get("fingerprint") != fingerprint or probe.get("fingerprint") != fingerprint:
        raise RuntimeError("head result does not match the training manifest")
    checkpoint = head["checkpoint"]
    layer = int(head["layer"])
    params = head["best_params"]
    found = cache_dir_from_pointer(checkpoint, fingerprint)
    if found is None:
        raise FileNotFoundError(checkpoint)
    directory, meta = found
    if not cache_is_complete(meta):
        raise RuntimeError("train feature cache is incomplete")
    train_features = read_layer(directory, layer, int(meta["n_rows"]), int(meta["hidden"]))
    labels = table[LABEL_COLUMN].to_numpy(dtype=np.int64)
    ids = af_common.read_submission_ids()
    names = [name for name, _path in ids]
    test_features = _read_test_cache(checkpoint, layer, names)
    train_index = np.arange(len(labels))
    written = []
    with gpu_lock():
        if not torch.cuda.is_available():
            raise SystemExit("CUDA is required to fit the submission head")
        device = torch.device("cuda")
        if test_features is None:
            test_features = _encode_test(
                checkpoint, layer, float(meta["max_audio_seconds"]), ids, device
            )
        for seed in SEEDS:
            model, scaler = fit_linear(train_features, labels, train_index, params, seed, device)
            scores = predict_linear(model, scaler, test_features, device)
            destination = submission_path(seed)
            af_common.write_submission_csv(destination, ids, scores)
            written.append(str(destination))
            print(f"wrote {destination}", flush=True)
            del model
        torch.cuda.empty_cache()
    write_json(
        METRICS_PATH.with_name("submission_stamp.json"),
        {"fingerprint": fingerprint, "paths": written, "checkpoint": checkpoint, "layer": layer},
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()
    write_submissions(args.manifest)


if __name__ == "__main__":
    main()
