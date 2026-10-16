"""Проверка форм на CPU и один фолд. Блокировку GPU и тест не трогает."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

import af_common
from af_common import binary_auc, iter_loso_folds, load_loso_table
from stage2.features import SpecFrontend
from stage2.loop import fit_fold
from stage2.models import LCNN, ResNet18
from stage3.frontend import RawWaveCNN, frontend_report


class _Tiny(Dataset):
    def __init__(self, specs, labels, indices):
        self.specs = specs
        self.labels = labels
        self.indices = np.asarray(indices)

    def __len__(self):
        return int(self.indices.shape[0])

    def __getitem__(self, i):
        j = int(self.indices[i])
        return self.specs[j], self.labels[j]


def _seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def main() -> None:
    table = load_loso_table(af_common.TRAIN_PROTOCOL_PATH)
    folds = iter_loso_folds(table)
    if len(folds) < 2:
        raise SystemExit(f"expected multiple LOSO folds, got {len(folds)}")
    for shape in ((1, 40, 80), (1, 80, 188)):
        batch = torch.randn(2, *shape)
        assert LCNN(16, 0.1)(batch).shape == (2,)
        assert ResNet18(32, 0.1)(batch).shape == (2,)
    spec = SpecFrontend("logmel", 40, 512, 256).eval()
    with torch.no_grad():
        mel = spec(torch.zeros(2, 16000))
    assert mel.ndim == 3 and torch.isfinite(mel).all()
    lfcc = SpecFrontend("lfcc", 40, 512, 256).eval()
    with torch.no_grad():
        cep = lfcc(torch.zeros(2, 16000))
    assert cep.shape[1] == 40 and torch.isfinite(cep).all()
    wave = torch.randn(2, 16000)
    assert RawWaveCNN(16, 256, 128, 3, 0.1)(wave).shape == (2,)
    report = frontend_report()
    assert report["external"] is False

    n = 32
    specs = torch.zeros(n, 1, 16, 16)
    labels = torch.zeros(n)
    specs[n // 2 :] = 1
    labels[n // 2 :] = 1
    device = torch.device("cpu")

    def model_fn():
        return LCNN(8, 0.0)

    def dataset_fn(indices):
        return _Tiny(specs, labels, indices)

    auc = fit_fold(
        model_fn,
        dataset_fn,
        np.arange(n),
        np.arange(n),
        labels.numpy(),
        seed=0,
        lr=1e-2,
        weight_decay=0.0,
        max_epochs=4,
        batch_size=8,
        device=device,
        auc_fn=binary_auc,
        seed_fn=_seed,
    )
    if auc < 0.8:
        raise SystemExit(f"smoke AUC too low: {auc}")
    print(f"smoke ok auc={auc:.3f} frontend={report['source']}", flush=True)


if __name__ == "__main__":
    main()
