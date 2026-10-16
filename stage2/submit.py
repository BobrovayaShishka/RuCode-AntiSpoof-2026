"""Переобучить принятую конфигурацию этапа 2 на всех размеченных строках и записать CSV сдачи.

Тест скорится один раз и не выбирает эпохи, семена и гиперпараметры.
Флаг --from-checkpoints пропускает обучение и читает final_checkpoints/seed_{seed}.pt.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

import af_common
from jsonutil import read_json, write_json
from stage2.bounds import (
    PIN_MEMORY_ON_CUDA,
    SPEC_COMPUTE_BATCH,
)
from stage2.datautil import (
    extract_labels,
    load_wave_from_path,
    make_store,
    open_training_table,
)
from stage2.driver import _run_locked
from stage2.features import SpecFrontend, ensure_spec_memmap
from stage2.loop import collect_scores, fit_full
from stage2.models import build_spectrogram_model, spectrogram_factories

CHECKPOINT_DIR = af_common.STAGE2_DIR / "final_checkpoints"
SUBMISSION_STAMP = af_common.STAGE2_DIR / "submission.json"


def _read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return read_json(path)


def _require_stage2_winner() -> dict:
    stage2 = _read_json(af_common.STAGE2_METRICS_PATH)
    stage3 = _read_json(af_common.STAGE3_METRICS_PATH)
    stage4 = _read_json(af_common.STAGE4_METRICS_PATH)
    if stage2.get("status") != "complete" or stage2.get("kept_vs_stage1") is not True:
        raise SystemExit("stage 2 is not the kept model")
    if stage3.get("kept_vs_stage2") is not False:
        raise SystemExit("stage 3 was kept; stage 2 must not write the submission")
    if stage4.get("kept_vs_neural") is not False:
        raise SystemExit("stage 4 was kept; stage 2 must not write the submission")
    return stage2


def _test_specs(frontend: SpecFrontend, ids: list[tuple[str, Path]], device: torch.device) -> np.ndarray:
    waves = [load_wave_from_path(path) for _name, path in ids]
    chunks: list[np.ndarray] = []
    frontend = frontend.to(device).eval()
    with torch.no_grad():
        for start in range(0, len(waves), SPEC_COMPUTE_BATCH):
            stop = min(len(waves), start + SPEC_COMPUTE_BATCH)
            batch = torch.from_numpy(np.stack(waves[start:stop], axis=0)).to(device)
            chunks.append(frontend(batch).cpu().numpy())
            print(f"test specs {stop}/{len(waves)}", flush=True)
    return np.concatenate(chunks, axis=0)


def _test_loader(params: dict, ids: list[tuple[str, Path]], device: torch.device) -> DataLoader:
    frontend = SpecFrontend(
        str(params["frontend"]),
        int(params["n_bins"]),
        int(params["n_fft"]),
        int(params["hop"]),
    )
    test_specs = _test_specs(frontend, ids, device)
    test_data = torch.from_numpy(np.ascontiguousarray(test_specs, dtype=np.float32)).unsqueeze(1)
    return DataLoader(
        TensorDataset(test_data),
        batch_size=int(params["batch_size"]),
        shuffle=False,
        num_workers=0,
        pin_memory=bool(PIN_MEMORY_ON_CUDA),
    )


def _score_checkpoints(params: dict) -> None:
    ids = af_common.read_submission_ids()
    checkpoints = [CHECKPOINT_DIR / f"seed_{int(seed)}.pt" for seed in af_common.SEEDS]
    missing = [str(path) for path in checkpoints if not path.is_file()]
    if missing:
        raise SystemExit(f"missing checkpoints: {missing}; run python -m stage2.submit without --from-checkpoints")

    def _go(device: torch.device) -> list[str]:
        test_loader = _test_loader(params, ids, device)
        written: list[str] = []
        for seed, checkpoint in zip(af_common.SEEDS, checkpoints, strict=True):
            payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
            if payload["params"] != params:
                raise SystemExit(f"{checkpoint} was trained with other params than stage2 best_params")
            model = build_spectrogram_model(
                str(params["family"]),
                int(params["lcnn_width"]),
                int(params["resnet_width"]),
                float(params["dropout"]),
            ).to(device)
            model.load_state_dict(payload["state_dict"])
            scores = collect_scores(model, test_loader, device, device.type == "cuda")
            model.cpu()
            destination = af_common.submission_path(int(seed))
            af_common.write_submission_csv(destination, ids, scores)
            written.append(str(destination))
            print(f"wrote {destination} from {checkpoint.name}", flush=True)
        return written

    _run_locked({}, _go)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--from-checkpoints",
        action="store_true",
        help="score the public test with outputs/stage2/final_checkpoints instead of refitting",
    )
    args = parser.parse_args(argv)
    metrics = _require_stage2_winner()
    params = dict(metrics["best_params"])
    if args.from_checkpoints:
        _score_checkpoints(params)
        return
    verdict = _read_json(af_common.STAGE0_VERDICT_PATH)
    table, source = open_training_table(verdict)
    labels = extract_labels(table)
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
    train_index = np.arange(len(labels), dtype=np.int64)
    ids = af_common.read_submission_ids()
    model_fn, dataset_fn = spectrogram_factories(params, specs, labels)

    def _go(device: torch.device) -> list[str]:
        test_loader = _test_loader(params, ids, device)
        written: list[str] = []
        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        for seed in af_common.SEEDS:
            state = fit_full(
                model_fn,
                dataset_fn,
                train_index,
                labels,
                seed=int(seed),
                lr=float(params["lr"]),
                weight_decay=float(params["weight_decay"]),
                max_epochs=int(params["max_epochs"]),
                batch_size=int(params["batch_size"]),
                device=device,
                seed_fn=af_common.set_global_seed,
            )
            checkpoint = CHECKPOINT_DIR / f"seed_{int(seed)}.pt"
            torch.save({"seed": int(seed), "params": params, "state_dict": state}, checkpoint)
            model = model_fn().to(device)
            model.load_state_dict(state)
            scores = collect_scores(model, test_loader, device, device.type == "cuda")
            model.cpu()
            destination = af_common.submission_path(int(seed))
            af_common.write_submission_csv(destination, ids, scores)
            written.append(str(destination))
            print(f"wrote {destination}", flush=True)
        return written

    written = _run_locked({}, _go)
    write_json(
        SUBMISSION_STAMP,
        {
            "stage": 2,
            "train_rows": len(labels),
            "train_source": source,
            "params": params,
            "epochs": int(params["max_epochs"]),
            "early_stopping": False,
            "seeds": [int(seed) for seed in af_common.SEEDS],
            "paths": written,
        },
    )


if __name__ == "__main__":
    main()
