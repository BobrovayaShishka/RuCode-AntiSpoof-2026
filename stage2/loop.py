"""Общий цикл обучения нейронных этапов. Early stopping смотрит AUC фолда LOSO."""

from __future__ import annotations

import gc
import time
from collections.abc import Callable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from stage2.bounds import (
    DROP_LAST,
    EARLY_STOP_PATIENCE,
    NEGATIVE_LABEL,
    NUM_WORKERS,
    OPTIMIZER_NAME,
    PIN_MEMORY_ON_CUDA,
    POSITIVE_LABEL,
    USE_POS_WEIGHT,
)


def positive_weight(labels: np.ndarray, indices: np.ndarray, device: torch.device) -> torch.Tensor | None:
    if not USE_POS_WEIGHT:
        return None
    subset = labels[np.asarray(indices)]
    n_pos = float(np.sum(subset == POSITIVE_LABEL))
    n_neg = float(np.sum(subset == NEGATIVE_LABEL))
    if n_pos <= 0.0 or n_neg <= 0.0:
        return None
    return torch.tensor([n_neg / n_pos], dtype=torch.float32, device=device)


@torch.no_grad()
def collect_auc(model, loader, device, auc_fn, use_amp: bool) -> float:
    model.eval()
    scores: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device, non_blocking=True)
        with torch.amp.autocast(device.type, enabled=use_amp):
            logits = model(batch_x).view(-1)
        scores.append(torch.sigmoid(logits.float()).cpu().numpy())
        targets.append(batch_y.view(-1).numpy())
    y_true = np.concatenate(targets)
    y_score = np.concatenate(scores)
    return float(auc_fn(y_true, y_score))


def fit_epochs(
    model: nn.Module,
    train_loader,
    *,
    optimizer,
    scaler,
    device: torch.device,
    use_amp: bool,
    max_epochs: int,
    compute_loss,
    prepare_batch=None,
    evaluate=None,
    patience: int = EARLY_STOP_PATIENCE,
    on_epoch=None,
):
    """Шаг AMP и опциональный early stopping по скалярному монитору.

    compute_loss внутри autocast, prepare_batch перед ним.
    При evaluate лучшие веса возвращаются на модель.
    Без него проходят все эпохи, возвращается state dict на CPU.
    """
    best_auc = -1.0
    stale = 0
    best_state = None
    for epoch in range(int(max_epochs)):
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            if prepare_batch is not None:
                batch_x, batch_y = prepare_batch(batch_x, batch_y)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device.type, enabled=use_amp):
                loss = compute_loss(model, batch_x, batch_y)
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite loss")
            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
        if evaluate is None:
            if on_epoch is not None:
                on_epoch(epoch)
            continue
        auc = float(evaluate())
        if auc > best_auc:
            best_auc = auc
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= int(patience):
                break
    if evaluate is None:
        return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    if best_state is None:
        raise RuntimeError("training produced no epoch")
    model.load_state_dict(best_state)
    return None


def _adamw(model, lr: float, weight_decay: float):
    if OPTIMIZER_NAME != "adamw":
        raise RuntimeError(f"unsupported optimizer {OPTIMIZER_NAME}")
    return torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))


def _loader(dataset, batch_size: int, shuffle: bool, generator, pin: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=shuffle,
        generator=generator,
        num_workers=NUM_WORKERS,
        drop_last=DROP_LAST,
        pin_memory=pin,
    )


def fit_fold(
    model_fn: Callable[[], nn.Module],
    dataset_fn: Callable[[np.ndarray], Dataset],
    train_index: np.ndarray,
    val_index: np.ndarray,
    labels: np.ndarray,
    *,
    seed: int,
    lr: float,
    weight_decay: float,
    max_epochs: int,
    batch_size: int,
    device: torch.device,
    auc_fn,
    seed_fn,
    return_scores: bool = False,
):
    seed_fn(int(seed))
    use_amp = device.type == "cuda"
    pin = bool(PIN_MEMORY_ON_CUDA and use_amp)
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    train_loader = _loader(dataset_fn(train_index), batch_size, True, generator, pin)
    val_loader = _loader(dataset_fn(val_index), batch_size, False, None, pin)
    model = None
    optimizer = None
    scaler = None
    try:
        model = model_fn().to(device)
        weight = positive_weight(labels, train_index, device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=weight)
        optimizer = _adamw(model, lr, weight_decay)
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if use_amp else None

        def compute_loss(net, batch_x, batch_y):
            return criterion(net(batch_x).view(-1), batch_y.view(-1))

        fit_epochs(
            model,
            train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            use_amp=use_amp,
            max_epochs=int(max_epochs),
            compute_loss=compute_loss,
            evaluate=lambda: collect_auc(model, val_loader, device, auc_fn, use_amp),
        )
        if return_scores:
            return collect_scores(model, val_loader, device, use_amp)
        return collect_auc(model, val_loader, device, auc_fn, use_amp)
    finally:
        if model is not None:
            model.cpu()
            del model
        del optimizer, scaler, train_loader, val_loader
        gc.collect()
        if use_amp:
            torch.cuda.empty_cache()


def run_folds(
    model_fn,
    dataset_fn,
    folds,
    labels,
    *,
    seed: int,
    lr: float,
    weight_decay: float,
    max_epochs: int,
    batch_size: int,
    device: torch.device,
    auc_fn,
    seed_fn,
    on_fold=None,
) -> dict[str, float]:
    fold_aucs: dict[str, float] = {}
    for fold in folds:
        started = time.perf_counter()
        auc = fit_fold(
            model_fn,
            dataset_fn,
            np.asarray(fold.train_index),
            np.asarray(fold.val_index),
            labels,
            seed=seed,
            lr=lr,
            weight_decay=weight_decay,
            max_epochs=max_epochs,
            batch_size=batch_size,
            device=device,
            auc_fn=auc_fn,
            seed_fn=seed_fn,
        )
        elapsed = time.perf_counter() - started
        fold_id = str(fold.fold_id)
        fold_aucs[fold_id] = float(auc)
        if on_fold is not None:
            on_fold(fold_id, float(auc), elapsed)
    return fold_aucs


def fit_full(
    model_fn: Callable[[], nn.Module],
    dataset_fn: Callable[[np.ndarray], Dataset],
    train_index: np.ndarray,
    labels: np.ndarray,
    *,
    seed: int,
    lr: float,
    weight_decay: float,
    max_epochs: int,
    batch_size: int,
    device: torch.device,
    seed_fn,
) -> dict[str, torch.Tensor]:
    """Обучить все размеченные строки max_epochs эпох. В сдаче нет holdout, early stopping выключен."""
    seed_fn(int(seed))
    use_amp = device.type == "cuda"
    pin = bool(PIN_MEMORY_ON_CUDA and use_amp)
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    train_loader = DataLoader(
        dataset_fn(train_index),
        batch_size=int(batch_size),
        shuffle=True,
        generator=generator,
        num_workers=NUM_WORKERS,
        drop_last=DROP_LAST,
        pin_memory=pin,
    )
    model = None
    optimizer = None
    scaler = None
    try:
        model = model_fn().to(device)
        weight = positive_weight(labels, train_index, device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=weight)
        optimizer = _adamw(model, lr, weight_decay)
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if use_amp else None

        def compute_loss(net, batch_x, batch_y):
            return criterion(net(batch_x).view(-1), batch_y.view(-1))

        def on_epoch(epoch: int) -> None:
            print(f"full fit seed {seed} epoch {epoch + 1}/{int(max_epochs)}", flush=True)

        return fit_epochs(
            model,
            train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            use_amp=use_amp,
            max_epochs=int(max_epochs),
            compute_loss=compute_loss,
            on_epoch=on_epoch,
        )
    finally:
        if model is not None:
            model.cpu()
            del model
        del optimizer, scaler, train_loader
        gc.collect()
        if use_amp:
            torch.cuda.empty_cache()


@torch.no_grad()
def collect_scores(model, loader, device, use_amp: bool) -> np.ndarray:
    model.eval()
    chunks: list[np.ndarray] = []
    for batch in loader:
        batch_x = batch[0] if isinstance(batch, (tuple, list)) else batch
        batch_x = batch_x.to(device, non_blocking=True)
        with torch.amp.autocast(device.type, enabled=use_amp):
            logits = model(batch_x).view(-1)
        chunks.append(torch.sigmoid(logits.float()).cpu().numpy())
    if not chunks:
        raise RuntimeError("score loader was empty")
    return np.concatenate(chunks)
