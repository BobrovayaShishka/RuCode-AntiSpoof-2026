"""Именованные константы этапов 10 и 11.

Константы протокола в af_common. Границы здесь выбирает Optuna.
Параметры головы этапа 4 читаются из того исследования, не копируются литералами.
"""

from __future__ import annotations

from af_common import MIN_MEAN_AUC_GAIN, N_OPTUNA_TRIALS, SEED_AGGREGATION, SEEDS
from experiments.config import OPTUNA_SEED_INDEX, STRONG_MEAN_AUC_GAIN
from stage4.constants import WAVLM_CHECKPOINT, XLSR_CHECKPOINT

# Реэкспорт. Этапы 10 и 11 вторую копию протокола не держат.
__all__ = [
    "MIN_MEAN_AUC_GAIN",
    "N_OPTUNA_TRIALS",
    "SEEDS",
    "STRONG_MEAN_AUC_GAIN",
]

SSL_STAGES_CODE_VERSION = 1
STAGE_PARTIAL = 10
STAGE_ENSEMBLE = 11

# Этап 3 отклонён против этапа 2, этап 4 против того же победителя.
# Ансамбль поэтому смешивается с этапом 2.
BASELINE_STAGE = 2
BASELINE_STAGE_NAME = "stage2_lcnn_lfcc"
BASELINE_STAGE_REASON = (
    "outputs/stage3/metrics.json kept_vs_stage2 is false and "
    "outputs/stage4/metrics.json kept_vs_neural is false, so the accepted "
    "simpler neural model is the stage-2 LCNN + LFCC"
)

# Замороженный SSL, который этап 10 должен превзойти. Не перенастраивается.
FROZEN_SSL_BASELINE_NAME = "stage4_xlsr_frozen_linear"

XLSR_MODEL_ID = XLSR_CHECKPOINT
WAVLM_MODEL_ID = WAVLM_CHECKPOINT
XLSR_SINGLE_NAME = "xlsr_frozen_linear"
WAVLM_SINGLE_NAME = "wavlm_frozen_linear"
ENSEMBLE_NAME = "equal_rank_xlsr_wavlm_stage2"

STRATEGY_LAST_K = "last_k_layers"
STRATEGY_LORA = "lora"
STRATEGIES = (STRATEGY_LAST_K, STRATEGY_LORA)

# Границы. Optuna выбирает одно значение, не рабочая точка по умолчанию.
LAST_K_CHOICES = (1, 2, 4)
LORA_RANK_CHOICES = (4, 8, 16)
LORA_ALPHA_CHOICES = (8, 16, 32)
LORA_TARGET_SUFFIXES = ("q_proj", "k_proj", "v_proj", "out_proj")
LORA_SCOPE = "all_attention_projections"

FINETUNE_LR_LOW = 1e-6
FINETUNE_LR_HIGH = 1e-4
FINETUNE_EPOCH_CHOICES = (1, 2, 3)
FINETUNE_BATCH_CHOICES = (1, 2)
FINETUNE_BATCH_FLOOR = 1

# Последнее скрытое состояние. Last-K и LoRA меняют этот вектор.
# Голова этапа 4 читает более ранний слой и остаётся baseline.
READOUT_HIDDEN_INDEX = -1

# Маски предобучения не гиперпараметры антиспуфа. Выключены, чтобы семя
# управляло обновлениями. Не ищутся.
FINETUNE_MASK_TIME_PROB = 0.0
FINETUNE_MASK_FEATURE_PROB = 0.0
FINETUNE_APPLY_SPEC_AUGMENT = False
FINETUNE_LAYERDROP = 0.0

TRAIN_DTYPE_NAME = "float32"
TRAIN_ALLOW_TF32 = False
GRADIENT_CHECKPOINTING = True
GRAD_CLIP_NORM = 1.0
DETERMINISTIC_ALGORITHMS = True
POOL_DENOM_FLOOR = 1.0
HEAD_OUT_FEATURES = 1

# Внутренний монитор early-stop: последний синтезатор train, не holdout LOSO.
INNER_MONITOR_POSITION = -1

# Смешивание рангов. Вес смеси не подбирается.
FUSION_NAME = "equal_rank_average"
FUSION_WEIGHTS_FIT = False
RANK_KIND = "mergesort"
RANK_BASE = 1
MOST_FOLDS_THRESHOLD = 0.5

PRIMARY_AGGREGATION = SEED_AGGREGATION
PRIMARY_MEAN_NOTE = (
    "Primary mean_auc is the mean over seeds of the mean over LOSO folds. "
    "That equals the mean over folds of the seed-averaged fold AUC. "
    "Primary worst_auc is the minimum seed-averaged fold AUC, "
    "not the minimum of the per-seed worsts. std and the bootstrap CI are over folds."
)

OPTUNA_SAMPLER_SEED = int(SEEDS[OPTUNA_SEED_INDEX])
HEARTBEAT_NAME = "heartbeat.txt"
RESULT_NAME = "result.json"
OPTUNA_STATE_NAME = "optuna_state.json"
PARTIAL_SEEDS_NAME = "partial_seeds.json"
SUBMISSION_STAMP_NAME = "submission.json"
OOF_DIRNAME = "oof"

DECISION_ACCEPT = "ACCEPT"
DECISION_REJECT = "REJECT"

# Соседний json рядом с submission_seed_{seed}.csv.
SIDECAR_SUFFIX = ".json"
