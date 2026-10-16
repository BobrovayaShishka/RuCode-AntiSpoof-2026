"""Сначала этап 4b, затем этап 4. Ждать ready_for_training от этапа 0."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

from af_common import (
    PROJECT_ROOT,
    SEEDS,
    STAGE0_VERDICT_PATH,
    STAGE2_METRICS_PATH,
    STAGE3_METRICS_PATH,
    accept_stage,
    submission_path,
)
from jsonutil import read_json, wait_until_ready, write_json
from stage2.report import load_auc_block
from stage4.constants import (
    HEAD_RESULT_PATH,
    LAYER_PROBE_PATH,
    METRICS_PATH,
    MODEL_BATCH_MULTIPLIERS,
    MODEL_CHECKPOINTS,
    NEURAL_METRICS_POLL_SECONDS,
    NEURAL_METRICS_WAIT_SECONDS,
    VERDICT_POLL_SECONDS,
)


def _wait_for_training_gate() -> dict:
    return wait_until_ready(STAGE0_VERDICT_PATH, VERDICT_POLL_SECONDS)


def _run_step(module: str, args: list[str]) -> None:
    command = [sys.executable, "-m", module, *args]
    print(" ".join(command), flush=True)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    completed = subprocess.run(command, cwd=str(PROJECT_ROOT), env=env, check=False)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def _wait_for_neural_metrics() -> tuple[dict | None, dict | None]:
    deadline = time.monotonic() + NEURAL_METRICS_WAIT_SECONDS
    stage2 = load_auc_block(STAGE2_METRICS_PATH)
    stage3 = load_auc_block(STAGE3_METRICS_PATH)
    while (stage2 is None or stage3 is None) and time.monotonic() < deadline:
        print("waiting for stage 2/3 metrics", flush=True)
        time.sleep(NEURAL_METRICS_POLL_SECONDS)
        stage2 = load_auc_block(STAGE2_METRICS_PATH)
        stage3 = load_auc_block(STAGE3_METRICS_PATH)
    return stage2, stage3


def _reference(stage2: dict | None, stage3: dict | None) -> tuple[str | None, dict | None]:
    candidates = []
    if stage2 is not None:
        candidates.append(("stage2", stage2))
    if stage3 is not None:
        candidates.append(("stage3", stage3))
    if not candidates:
        return None, None
    name, block = max(candidates, key=lambda item: (item[1]["mean_auc"], item[1]["worst_auc"]))
    return name, block


def _write_metrics(manifest: str, verdict: dict, kept, reference_name, reference, stage2, stage3, submission_paths, skipped) -> None:
    head = read_json(HEAD_RESULT_PATH)
    probe = read_json(LAYER_PROBE_PATH)
    aggregate = head["aggregate"]
    payload = {
        "stage": 4,
        "manifest": manifest,
        "leak": verdict.get("leak"),
        "acoustic_shortcut": verdict.get("acoustic_shortcut"),
        "ready_for_training": verdict.get("ready_for_training"),
        "checkpoint": head["checkpoint"],
        "layer": head["layer"],
        "best_params": head["best_params"],
        "mean_auc": aggregate["mean_auc"],
        "worst_auc": aggregate["worst_auc"],
        "std_auc": aggregate["std_auc"],
        "ci_low": aggregate["ci_low"],
        "ci_high": aggregate["ci_high"],
        "fold_aucs": aggregate["fold_aucs"],
        "per_seed": head["per_seed"],
        "seed_mean_aucs": head["seed_mean_aucs"],
        "seed_std_of_mean_auc": head["seed_std_of_mean_auc"],
        "probe_chosen": probe.get("chosen"),
        "vram_fallback": {
            checkpoint: (probe.get("models") or {}).get(checkpoint, {}).get("vram_fallback")
            for checkpoint in MODEL_CHECKPOINTS
        },
        "stage2": stage2,
        "stage3": stage3,
        "reference_stage": reference_name,
        "reference_mean_auc": None if reference is None else reference["mean_auc"],
        "reference_worst_auc": None if reference is None else reference["worst_auc"],
        "kept_vs_neural": kept,
        "submission_paths": submission_paths,
        "submission_skipped_reason": skipped,
    }
    write_json(METRICS_PATH, payload)


def main() -> None:
    verdict = _wait_for_training_gate()
    manifest = str(verdict["data_manifest"])
    manifest_args = ["--manifest", manifest]
    for checkpoint in MODEL_CHECKPOINTS:
        _run_step(
            "stage4.extract",
            [*manifest_args, "--checkpoint", checkpoint, "--multiplier", str(MODEL_BATCH_MULTIPLIERS[checkpoint])],
        )
    _run_step("stage4.probe", manifest_args)
    _run_step("stage4.head", manifest_args)

    stage2, stage3 = _wait_for_neural_metrics()
    reference_name, reference = _reference(stage2, stage3)
    head = read_json(HEAD_RESULT_PATH)
    if reference is None:
        kept = None
        skipped = "stage 2 and stage 3 metrics were still missing"
    else:
        kept = bool(accept_stage(reference, head["aggregate"]))
        skipped = None if kept else "kept_vs_neural is false"
    _write_metrics(manifest, verdict, kept, reference_name, reference, stage2, stage3, None, skipped)
    if kept is True:
        _run_step("stage4.submit", manifest_args)
        submission_paths = [str(submission_path(seed)) for seed in SEEDS]
        for path in submission_paths:
            if not Path(path).is_file():
                raise SystemExit(f"submission was not written: {path}")
        _write_metrics(
            manifest, verdict, kept, reference_name, reference, stage2, stage3, submission_paths, None
        )
    print(f"metrics {METRICS_PATH} kept_vs_neural={kept}", flush=True)


if __name__ == "__main__":
    main()
