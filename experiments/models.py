"""LCNN-64 со стеком признаков этапа 2 и сменной головой пулинга.

Mean + BCE берёт stage2.models.LCNN без изменений. Остальные головы меняют только пул и лосс.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

import af_common
from experiments.config import (
    BATCH_DIM,
    CENTER_ROWS,
    CHANNEL_DIM,
    COLLAPSED_CHANNEL_DIM,
    COLLAPSED_TIME_DIM,
    FEATURE_FLATTEN_START,
    FLATTEN_ALL,
    FREQ_DIM,
    GLOBAL_POOL_SIZE,
    GRAPH_LEAKY_SLOPE,
    LAST_DIM,
    LOGIT_DIM,
    LOSS_AM,
    LOSS_BCE,
    LOSS_OC,
    N_CLASSES,
    ONE,
    PAIR_WIDTH,
    POOL_AASIST,
    POOL_ASP,
    POOL_MEAN,
    SEQ_FEAT_DIM,
    SEQ_NODE_DIM,
    STAT_COUNT,
    TIME_DIM,
    TWO,
    VARIANCE_EPS,
)
from stage2.models import LCNN


class MeanPooling(nn.Module):
    """Глобальное среднее по частоте и времени. То же сжатие, что у пула LCNN."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.output_dim = int(channels)
        self.pool = nn.AdaptiveAvgPool2d(GLOBAL_POOL_SIZE)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.pool(features).flatten(FEATURE_FLATTEN_START)


class AttentiveStatPooling(nn.Module):
    """Внимание по времени, затем взвешенные среднее и СКО.

    Карта LCNN сначала усредняется по частоте. Это та же временная ось, что
    сжимает baseline mean, но с обучаемыми весами.
    """

    def __init__(self, channels: int, attention_hidden: int) -> None:
        super().__init__()
        self.output_dim = int(channels) * STAT_COUNT
        self.score = nn.Sequential(
            nn.Linear(int(channels), int(attention_hidden)),
            nn.Tanh(),
            nn.Linear(int(attention_hidden), LOGIT_DIM),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        sequence = features.mean(dim=FREQ_DIM)
        sequence = sequence.transpose(COLLAPSED_CHANNEL_DIM, COLLAPSED_TIME_DIM)
        logits = self.score(sequence).squeeze(LAST_DIM)
        weights = torch.softmax(logits, dim=SEQ_NODE_DIM).unsqueeze(LAST_DIM)
        mean = torch.sum(weights * sequence, dim=SEQ_NODE_DIM)
        centered = sequence - mean.unsqueeze(SEQ_NODE_DIM)
        variance = torch.sum(weights * centered.square(), dim=SEQ_NODE_DIM)
        std = torch.sqrt(variance.clamp_min(VARIANCE_EPS))
        return torch.cat((mean, std), dim=CHANNEL_DIM)


class _HeteroGraphLayer(nn.Module):
    """Один слой гетерогенного графового внимания по двум множествам узлов."""

    def __init__(self, channels: int, hidden: int) -> None:
        super().__init__()
        self.proj = nn.Linear(int(channels), int(hidden))
        self.score = nn.Linear(int(hidden) * PAIR_WIDTH, LOGIT_DIM)
        self.out = nn.Linear(int(hidden), int(channels))

    def _attend(self, query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
        batch = query.shape[BATCH_DIM]
        n_query = query.shape[SEQ_NODE_DIM]
        n_key = key.shape[SEQ_NODE_DIM]
        hidden = query.shape[SEQ_FEAT_DIM]
        query_exp = query.unsqueeze(SEQ_FEAT_DIM).expand(batch, n_query, n_key, hidden)
        key_exp = key.unsqueeze(SEQ_NODE_DIM).expand(batch, n_query, n_key, hidden)
        logits = self.score(torch.cat((query_exp, key_exp), dim=LAST_DIM)).squeeze(LAST_DIM)
        weights = torch.softmax(F.leaky_relu(logits, GRAPH_LEAKY_SLOPE), dim=LAST_DIM)
        return torch.matmul(weights, key)

    def forward(self, spectral: torch.Tensor, temporal: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        spec_h = self.proj(spectral)
        temp_h = self.proj(temporal)
        spec_upd = torch.maximum(self._attend(spec_h, spec_h), self._attend(spec_h, temp_h))
        temp_upd = torch.maximum(self._attend(temp_h, temp_h), self._attend(temp_h, spec_h))
        return spectral + self.out(spec_upd), temporal + self.out(temp_upd)


class AASISTLitePooling(nn.Module):
    """Спектрально-временное графовое внимание по карте LCNN, затем mean+max.

    Узлы: максимум по другой оси, как в AASIST. Глубина и hidden ищет Optuna.
    Это не энкодер RawNet2 AASIST.
    """

    def __init__(self, channels: int, n_layers: int, hidden: int) -> None:
        super().__init__()
        self.output_dim = int(channels) * STAT_COUNT
        self.layers = nn.ModuleList(
            _HeteroGraphLayer(int(channels), int(hidden)) for _ in range(int(n_layers))
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        spectral = features.amax(dim=TIME_DIM).transpose(CHANNEL_DIM, FREQ_DIM)
        temporal = features.amax(dim=FREQ_DIM).transpose(COLLAPSED_CHANNEL_DIM, COLLAPSED_TIME_DIM)
        for layer in self.layers:
            spectral, temporal = layer(spectral, temporal)
        nodes = torch.cat((spectral, temporal), dim=SEQ_NODE_DIM)
        mean = nodes.mean(dim=SEQ_NODE_DIM)
        maximum = nodes.amax(dim=SEQ_NODE_DIM)
        return torch.cat((mean, maximum), dim=CHANNEL_DIM)


def build_pooling(hparams: dict, channels: int) -> nn.Module:
    pooling = hparams["pooling"]
    if pooling == POOL_MEAN:
        return MeanPooling(channels)
    if pooling == POOL_ASP:
        return AttentiveStatPooling(channels, int(hparams["attention_hidden"]))
    if pooling == POOL_AASIST:
        return AASISTLitePooling(channels, int(hparams["graph_layers"]), int(hparams["graph_hidden"]))
    raise ValueError(f"unknown pooling {pooling}")


class MeanBCEModel(nn.Module):
    """Точная LCNN этапа 2: mean pool, dropout, один логит, BCE-with-logits."""

    def __init__(self, hparams: dict) -> None:
        super().__init__()
        self.inner = LCNN(width=int(hparams["lcnn_width"]), dropout=float(hparams["dropout"]))

    def training_loss(self, specs: torch.Tensor, labels: torch.Tensor, pos_weight: torch.Tensor | None) -> torch.Tensor:
        logits = self.inner(specs).reshape(FLATTEN_ALL).float()
        target = labels.reshape(FLATTEN_ALL).float()
        return F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight)

    def spoof_scores(self, specs: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.inner(specs).reshape(FLATTEN_ALL).float())


class ContinuedLCNN(nn.Module):
    """Стек LCNN этапа 2 с выбранным пулом и BCE, AM-Softmax или OC-Softmax."""

    def __init__(self, hparams: dict) -> None:
        super().__init__()
        width = int(hparams["lcnn_width"])
        donor = LCNN(width=width, dropout=float(hparams["dropout"]))
        self.features = donor.features
        self.pooling = build_pooling(hparams, width)
        self.drop = nn.Dropout(float(hparams["dropout"]))
        self.loss_name = str(hparams["loss"])
        pooled = int(self.pooling.output_dim)
        if self.loss_name == LOSS_BCE:
            self.fc = nn.Linear(pooled, LOGIT_DIM)
        elif self.loss_name == LOSS_AM:
            self.am_scale = float(hparams["am_scale"])
            self.am_margin = float(hparams["am_margin"])
            self.am_weight = nn.Parameter(torch.empty(N_CLASSES, pooled))
            nn.init.xavier_uniform_(self.am_weight)
        elif self.loss_name == LOSS_OC:
            bonafide_margin = float(hparams["oc_margin_bonafide"])
            spoof_margin = float(hparams["oc_margin_spoof"])
            if bonafide_margin <= spoof_margin:
                raise ValueError("OC-Softmax bonafide margin must be above the spoof margin")
            self.oc_scale = float(hparams["oc_scale"])
            self.oc_margin_bonafide = bonafide_margin
            self.oc_margin_spoof = spoof_margin
            self.oc_center = nn.Parameter(torch.empty(CENTER_ROWS, pooled))
            nn.init.xavier_uniform_(self.oc_center)
        else:
            raise ValueError(f"unknown loss {self.loss_name}")
        del donor

    def embed(self, specs: torch.Tensor) -> torch.Tensor:
        return self.drop(self.pooling(self.features(specs)))

    def training_loss(self, specs: torch.Tensor, labels: torch.Tensor, pos_weight: torch.Tensor | None) -> torch.Tensor:
        embedding = self.embed(specs)
        target = labels.reshape(FLATTEN_ALL).long()
        if self.loss_name == LOSS_BCE:
            logits = self.fc(embedding).reshape(FLATTEN_ALL).float()
            return F.binary_cross_entropy_with_logits(
                logits,
                labels.reshape(FLATTEN_ALL).float(),
                pos_weight=pos_weight,
            )
        if self.loss_name == LOSS_AM:
            return _am_softmax_loss(embedding.float(), target, self.am_weight, self.am_scale, self.am_margin)
        return _oc_softmax_loss(
            embedding.float(),
            target,
            self.oc_center,
            self.oc_scale,
            self.oc_margin_bonafide,
            self.oc_margin_spoof,
        )

    def spoof_scores(self, specs: torch.Tensor) -> torch.Tensor:
        embedding = self.embed(specs).float()
        if self.loss_name == LOSS_BCE:
            return torch.sigmoid(self.fc(embedding).reshape(FLATTEN_ALL))
        if self.loss_name == LOSS_AM:
            return _am_spoof_probability(embedding, self.am_weight, self.am_scale)
        return _oc_spoof_score(embedding, self.oc_center)


def build_model(hparams: dict) -> nn.Module:
    if hparams["pooling"] == POOL_MEAN and hparams["loss"] == LOSS_BCE:
        return MeanBCEModel(hparams)
    return ContinuedLCNN(hparams)


def _am_softmax_loss(
    embedding: torch.Tensor,
    labels: torch.Tensor,
    weight: torch.Tensor,
    scale: float,
    margin: float,
) -> torch.Tensor:
    cosine = _normalized_cosine(embedding, weight)
    adjusted = cosine.clone()
    adjusted[torch.arange(labels.shape[BATCH_DIM], device=labels.device), labels] = (
        cosine[torch.arange(labels.shape[BATCH_DIM], device=labels.device), labels] - float(margin)
    )
    return F.cross_entropy(adjusted * float(scale), labels)


def _am_spoof_probability(embedding: torch.Tensor, weight: torch.Tensor, scale: float) -> torch.Tensor:
    cosine = _normalized_cosine(embedding, weight)
    probabilities = torch.softmax(cosine * float(scale), dim=CHANNEL_DIM)
    return probabilities[:, af_common.LABEL_SPOOF]


def _oc_softmax_loss(
    embedding: torch.Tensor,
    labels: torch.Tensor,
    center: torch.Tensor,
    scale: float,
    margin_bonafide: float,
    margin_spoof: float,
) -> torch.Tensor:
    cosine = _center_cosine(embedding, center)
    bonafide = labels == af_common.LABEL_BONAFIDE
    directed = torch.where(
        bonafide,
        float(margin_bonafide) - cosine,
        cosine - float(margin_spoof),
    )
    return F.softplus(float(scale) * directed).mean()


def _oc_spoof_score(embedding: torch.Tensor, center: torch.Tensor) -> torch.Tensor:
    """Косинус до bona fide перевести в скор spoof на отрезке [0, 1]."""
    cosine = _center_cosine(embedding, center)
    return (ONE - cosine) / TWO


def _normalized_cosine(embedding: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return F.normalize(embedding, dim=CHANNEL_DIM) @ F.normalize(weight, dim=CHANNEL_DIM).transpose(BATCH_DIM, CHANNEL_DIM)


def _center_cosine(embedding: torch.Tensor, center: torch.Tensor) -> torch.Tensor:
    return (F.normalize(embedding, dim=CHANNEL_DIM) * F.normalize(center, dim=CHANNEL_DIM)).sum(dim=CHANNEL_DIM)
