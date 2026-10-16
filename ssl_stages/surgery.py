"""Заморозка, last-K и LoRA на SSL-энкодере этапа 4.

Цели: проекции внимания загруженного backbone. Отсутствующая проекция: ошибка.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from ssl_stages.constants import (
    FINETUNE_APPLY_SPEC_AUGMENT,
    FINETUNE_LAYERDROP,
    FINETUNE_MASK_FEATURE_PROB,
    FINETUNE_MASK_TIME_PROB,
    GRADIENT_CHECKPOINTING,
    HEAD_OUT_FEATURES,
    LORA_TARGET_SUFFIXES,
    STRATEGY_LAST_K,
    STRATEGY_LORA,
    TRAIN_DTYPE_NAME,
)


class LoRALinear(nn.Module):
    """Замороженный линейный слой плюс низкоранговый остаток. Остаток стартует с нуля."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float) -> None:
        super().__init__()
        if rank < 1:
            raise ValueError("LoRA rank must be positive")
        self.base = base
        self.in_features = int(base.in_features)
        self.out_features = int(base.out_features)
        self.rank = int(rank)
        self.scale = float(alpha) / float(rank)
        self.lora_down = nn.Parameter(torch.empty(self.rank, self.in_features))
        self.lora_up = nn.Parameter(torch.empty(self.out_features, self.rank))
        nn.init.kaiming_uniform_(self.lora_down)
        nn.init.zeros_(self.lora_up)
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)

    @property
    def weight(self) -> torch.Tensor:
        return self.base.weight

    @property
    def bias(self) -> torch.Tensor | None:
        return self.base.bias

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = F.linear(F.linear(inputs, self.lora_down), self.lora_up) * self.scale
        return self.base(inputs) + residual


def train_dtype() -> torch.dtype:
    if TRAIN_DTYPE_NAME == "float32":
        return torch.float32
    raise ValueError(TRAIN_DTYPE_NAME)


def transformer_layers(model: nn.Module) -> nn.ModuleList:
    encoder = getattr(model, "encoder", None)
    layers = None if encoder is None else getattr(encoder, "layers", None)
    if not isinstance(layers, nn.ModuleList) or len(layers) == 0:
        raise RuntimeError(f"{type(model).__name__} has no transformer layers")
    return layers


def attention_target_names(layer: nn.Module) -> list[str]:
    attention = getattr(layer, "attention", None)
    if attention is None:
        raise RuntimeError(f"{type(layer).__name__} has no attention module")
    found = []
    for name in LORA_TARGET_SUFFIXES:
        module = getattr(attention, name, None)
        if isinstance(module, nn.Linear):
            found.append(name)
    if not found:
        seen = [name for name, _module in attention.named_modules()]
        raise RuntimeError(
            f"none of {LORA_TARGET_SUFFIXES} are Linear modules on attention; saw {seen}"
        )
    return found


def _disable_pretraining_noise(model: nn.Module) -> None:
    config = model.config
    config.mask_time_prob = float(FINETUNE_MASK_TIME_PROB)
    config.mask_feature_prob = float(FINETUNE_MASK_FEATURE_PROB)
    config.apply_spec_augment = bool(FINETUNE_APPLY_SPEC_AUGMENT)
    config.layerdrop = float(FINETUNE_LAYERDROP)


def load_backbone(checkpoint: str, device: torch.device):
    """Загрузить чекпоинт этапа 4 для обучения. При необходимости скачать в кэш этапа 4."""
    from transformers import AutoFeatureExtractor, AutoModel

    from stage4.constants import HF_HUB_CACHE
    from stage4.extract import ensure_weights

    ensure_weights(checkpoint)
    kwargs = {"cache_dir": str(HF_HUB_CACHE)}
    extractor = AutoFeatureExtractor.from_pretrained(checkpoint, **kwargs)
    model = AutoModel.from_pretrained(checkpoint, dtype=train_dtype(), **kwargs)
    _disable_pretraining_noise(model)
    if GRADIENT_CHECKPOINTING and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    model.to(device)
    return extractor, model


def _freeze_all(model: nn.Module) -> None:
    for parameter in model.parameters():
        parameter.requires_grad_(False)


def unfreeze_last_k(model: nn.Module, k: int) -> int:
    layers = transformer_layers(model)
    count = int(k)
    if count < 1 or count > len(layers):
        raise ValueError(f"K={count} is outside 1..{len(layers)}")
    _freeze_all(model)
    for layer in list(layers)[-count:]:
        for parameter in layer.parameters():
            parameter.requires_grad_(True)
    return count


def install_lora(model: nn.Module, rank: int, alpha: float) -> int:
    _freeze_all(model)
    replaced = 0
    for layer in transformer_layers(model):
        attention = layer.attention
        for name in attention_target_names(layer):
            setattr(attention, name, LoRALinear(getattr(attention, name), int(rank), float(alpha)))
            replaced += 1
    if replaced == 0:
        raise RuntimeError("LoRA replaced no attention projections")
    return replaced


def apply_strategy(model: nn.Module, params: dict) -> str:
    strategy = str(params["strategy"])
    if strategy == STRATEGY_LAST_K:
        count = unfreeze_last_k(model, int(params["k"]))
        return f"unfroze last {count} transformer layers"
    if strategy == STRATEGY_LORA:
        replaced = install_lora(model, int(params["lora_rank"]), float(params["lora_alpha"]))
        return f"LoRA on {replaced} attention projections"
    raise ValueError(strategy)


def eval_frozen_subtrees(model: nn.Module) -> None:
    """Dropout выключен в модулях без обучаемых весов."""
    model.train()
    for module in model.modules():
        parameters = list(module.parameters())
        if parameters and all(not parameter.requires_grad for parameter in parameters):
            module.eval()


def make_head(hidden_size: int, device: torch.device) -> nn.Linear:
    return nn.Linear(int(hidden_size), int(HEAD_OUT_FEATURES)).to(device)


def trainable_parameters(model: nn.Module, head: nn.Module) -> list[nn.Parameter]:
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    parameters.extend(parameter for parameter in head.parameters() if parameter.requires_grad)
    if not parameters:
        raise RuntimeError("no trainable parameters")
    return parameters


def snapshot_trainable(model: nn.Module, head: nn.Module) -> dict[str, torch.Tensor]:
    state = {}
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            state["backbone." + name] = parameter.detach().cpu().clone()
    for name, parameter in head.named_parameters():
        state["head." + name] = parameter.detach().cpu().clone()
    return state


def restore_trainable(model: nn.Module, head: nn.Module, state: dict[str, torch.Tensor]) -> None:
    mapping = {"backbone": model, "head": head}
    for key, tensor in state.items():
        prefix, name = key.split(".", 1)
        module = mapping[prefix]
        parameter = dict(module.named_parameters())[name]
        parameter.data.copy_(tensor.to(device=parameter.device, dtype=parameter.dtype))


def feature_sample_rate(checkpoint: str) -> int:
    from transformers import AutoFeatureExtractor

    from stage4.constants import HF_HUB_CACHE
    from stage4.extract import ensure_weights

    ensure_weights(checkpoint)
    extractor = AutoFeatureExtractor.from_pretrained(checkpoint, cache_dir=str(HF_HUB_CACHE))
    return int(extractor.sampling_rate)


def assert_backbone_api(checkpoint: str, device: torch.device) -> int:
    """Упасть до Optuna, если у энкодера нет проекций внимания для адаптации."""
    _extractor, model = load_backbone(checkpoint, device)
    try:
        layers = transformer_layers(model)
        attention_target_names(layers[0])
        return len(layers)
    finally:
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
