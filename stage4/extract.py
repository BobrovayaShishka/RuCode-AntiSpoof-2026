"""Кэшировать mean-pooled признаки каждого слоя SSL. GPU только внутри gpu_lock."""

from __future__ import annotations

import argparse
import gc
import sys
import time

from stage4.gpu_env import configure_process_env

configure_process_env()

import torch
from huggingface_hub import snapshot_download

from af_common import PATH_COLUMN, SEEDS, gpu_lock, set_global_seed
from stage4.audio import load_wave, wave_sample_count
from stage4.cache_io import (
    LayerStore,
    allocate_cache_dir,
    cache_dir_from_pointer,
    cache_is_complete,
    write_pointer,
)
from stage4.constants import (
    AUDIO_MAX_SECONDS_CANDIDATES,
    EXTRACT_LOOKAHEAD,
    HF_HUB_CACHE,
    MIN_AUDIO_SECONDS,
    MODEL_CHECKPOINTS,
)
from stage4.data import load_training_table, table_fingerprint
from stage4.encode import encode_waves, fit_max_seconds, is_oom, load_frozen


def ensure_weights(checkpoint: str) -> None:
    print(f"weights {checkpoint}", flush=True)
    snapshot_download(repo_id=checkpoint, cache_dir=str(HF_HUB_CACHE))


def pack_count(lengths: list[int], budget_samples: int) -> int:
    count = 0
    longest = 0
    for length in lengths:
        new_longest = max(longest, length)
        new_count = count + 1
        if new_count * new_longest > budget_samples and count > 0:
            break
        count = new_count
        longest = new_longest
    return count


def _vram_note(cap: float, budget_samples: int, cap_samples: int, multiplier: int) -> dict | None:
    full_cap = float(AUDIO_MAX_SECONDS_CANDIDATES[0])
    full_budget = cap_samples * multiplier if cap == full_cap else None
    cropped = cap < full_cap
    reduced_batch = budget_samples < cap_samples * multiplier
    if not cropped and not reduced_batch:
        return None
    return {
        "max_audio_seconds": cap,
        "full_cap_seconds": full_cap,
        "budget_samples": int(budget_samples),
        "initial_budget_samples": int(cap_samples * multiplier),
        "cropped_audio": cropped,
        "reduced_batch": reduced_batch,
        "unused_initial_budget": full_budget,
    }


def extract(checkpoint: str, manifest: str, multiplier: int) -> None:
    if checkpoint not in MODEL_CHECKPOINTS:
        raise SystemExit(f"checkpoint is not a stage-4 constant: {checkpoint}")
    table = load_training_table(manifest)
    print(f"fingerprint {len(table)} rows", flush=True)
    fingerprint = table_fingerprint(table)
    found = cache_dir_from_pointer(checkpoint, fingerprint)
    if found is not None:
        directory, meta = found
        if cache_is_complete(meta):
            print(f"cache complete {directory}", flush=True)
            return
    else:
        directory, meta = None, None

    ensure_weights(checkpoint)
    paths = [table.at[i, PATH_COLUMN] for i in range(len(table))]
    n_rows = len(paths)
    with gpu_lock():
        if not torch.cuda.is_available():
            raise SystemExit("CUDA is required for SSL feature extraction")
        device = torch.device("cuda")
        set_global_seed(SEEDS[0])
        print(f"gpu lock held, loading {checkpoint} on {torch.cuda.get_device_name(0)}", flush=True)
        extractor, model = load_frozen(checkpoint, device)
        sample_rate = int(extractor.sampling_rate)
        min_samples = max(1, int(round(MIN_AUDIO_SECONDS * sample_rate)))
        try:
            if meta is None:
                cap = fit_max_seconds(extractor, model, device)
                print(f"max audio seconds that fit: {cap}", flush=True)
                directory = allocate_cache_dir(checkpoint, cap, fingerprint)
                cap_samples = int(round(cap * sample_rate))
                budget = cap_samples * int(multiplier)
                store = None
                rows_done = 0
                n_cropped = 0
            else:
                cap = float(meta["max_audio_seconds"])
                cap_samples = int(meta["cap_samples"])
                budget = int(meta["budget_samples"])
                rows_done = int(meta["rows_done"])
                n_cropped = int(meta.get("n_cropped", 0))
                store = LayerStore(directory)
                store.open_existing(meta)
                print(f"resume {directory} at {rows_done}/{n_rows}", flush=True)

            started = time.perf_counter()
            while rows_done < n_rows:
                end = min(n_rows, rows_done + EXTRACT_LOOKAHEAD)
                lengths = [
                    wave_sample_count(paths[index], sample_rate, cap_samples, min_samples)
                    for index in range(rows_done, end)
                ]
                while True:
                    count = pack_count(lengths, budget)
                    batch_paths = paths[rows_done : rows_done + count]
                    waves = []
                    batch_cropped = 0
                    for path in batch_paths:
                        wave, cropped = load_wave(path, sample_rate, cap_samples, min_samples)
                        waves.append(wave)
                        batch_cropped += int(cropped)
                    try:
                        pooled = encode_waves(extractor, model, waves, device)
                        break
                    except Exception as exc:
                        if not is_oom(exc) or count <= 1:
                            raise
                        torch.cuda.empty_cache()
                        gc.collect()
                        budget = max(1, budget // 2)
                        lengths = lengths[: max(1, count // 2)]
                        print(f"OOM, budget_samples -> {budget} take={len(lengths)}", flush=True)
                        if store is not None:
                            store.meta["budget_samples"] = int(budget)
                            store.save_meta()

                if store is None:
                    hidden = int(pooled[0].shape[1])
                    note = _vram_note(cap, budget, cap_samples, multiplier)
                    meta = {
                        "checkpoint": checkpoint,
                        "fingerprint": fingerprint,
                        "n_rows": n_rows,
                        "n_layers": len(pooled),
                        "hidden": hidden,
                        "rows_done": 0,
                        "max_audio_seconds": cap,
                        "cap_samples": cap_samples,
                        "sample_rate": sample_rate,
                        "budget_samples": int(budget),
                        "initial_multiplier": int(multiplier),
                        "n_cropped": 0,
                        "vram_fallback": note,
                        "layer_index_note": (
                            "0 is the encoder input after the positional convolution; "
                            "the last index is the final transformer layer"
                        ),
                        "device": torch.cuda.get_device_name(0),
                    }
                    store = LayerStore(directory)
                    store.create(meta, hidden)
                    write_pointer(checkpoint, fingerprint, directory, complete=False)

                n_cropped += batch_cropped
                store.meta["budget_samples"] = int(budget)
                store.meta["vram_fallback"] = _vram_note(cap, budget, cap_samples, multiplier)
                store.write_rows(rows_done, pooled, n_cropped)
                rows_done += count
                elapsed = time.perf_counter() - started
                rate = rows_done / elapsed if elapsed > 0 else 0.0
                if rows_done == count or rows_done % 200 < count:
                    print(
                        f"{checkpoint} {rows_done}/{n_rows} "
                        f"{rate:.2f} rows/s layers={len(pooled)}",
                        flush=True,
                    )
                del pooled, waves
            store.meta["complete"] = True
            store.save_meta()
            write_pointer(checkpoint, fingerprint, directory, complete=True)
            store.close()
            print(f"cached {checkpoint} -> {directory}", flush=True)
        finally:
            del model
            del extractor
            gc.collect()
            torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--multiplier", type=int, required=True)
    args = parser.parse_args()
    extract(args.checkpoint, args.manifest, args.multiplier)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
