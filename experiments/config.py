"""Именованные константы этапов 5-9.

Числа LCNN + LFCC ниже: победитель этапа 2, не границы поиска.
Новые ручки следующих этапов: только границы, выбор пишет decisions.json.

MIN_MEAN_AUC_GAIN (0.005) порог приёмки. STRONG_MEAN_AUC_GAIN (0.01)
только в тексте решения, порог не подменяет.
"""

from __future__ import annotations

from af_common import (
    MIN_MEAN_AUC_GAIN,
    N_OPTUNA_TRIALS,
    SCORE_FLOAT_FORMAT,
    SEED_AGGREGATION,
    SEEDS,
    SUBMISSION_ID_COLUMN,
    SUBMISSION_SCORE_COLUMN,
)
from stage2.bounds import (
    DROP_LAST,  # noqa: F401
    EARLY_STOP_PATIENCE,
    FRONTEND_LFCC,
    NUM_WORKERS,  # noqa: F401
    OPTIMIZER_NAME,
    PIN_MEMORY_ON_CUDA,  # noqa: F401
    SPEC_COMPUTE_BATCH,  # noqa: F401
    TARGET_LENGTH,  # noqa: F401
    TARGET_SR,  # noqa: F401
)

# Реэкспорт констант протокола. Этапы 5-9 вторую копию не держат.
# В SEEDS входит 0, семя известной сдачи.

STRONG_MEAN_AUC_GAIN = 0.01
DECISIONS_VERSION = 1
CONTINUATION_CODE_VERSION = 1
OPTUNA_SEED_INDEX = 0
FIRST_STAGE = 5
LAST_STAGE = 9
N_LABELED_ROWS = 25380

STATUS_RUNNING = "running"
STATUS_COMPLETE = "complete"
ROLE_REFERENCE = "reference"
ROLE_NEW = "new"
DECISION_SOURCE = "loso_only"
FOLDS_NAME = "loso-synthesizer"
DECISIONS_FILENAME = "decisions.json"
CONTINUATION_DIRNAME = "continuation"
FEAT_CACHE_DIRNAME = "feat_cache"
STAGE2_DIRNAME = "stage2"
HEARTBEAT_TEMPLATE = "stage{stage}_heartbeat.txt"
CHECKPOINT_DIRNAME = "final_checkpoints"
DISPLAY_METRIC_FORMAT = ".4f"
DISPLAY_THRESHOLD_FORMAT = ".3f"
DISPLAY_INDEX_OFFSET = 1
DISPLAY_SECONDS_FORMAT = ".0f"
JSON_INDENT = 2
STALE_STEP = 1
STALE_START = 0
INITIAL_BEST_AUC = -1.0
DEVICE_INDEX = 0

# Победитель этапа 2 из outputs/stage2/metrics.json. Публичные 97.65 в отбор не входят.
BASELINE_NAME = "lcnn64_lfcc_mean_bce"
BASELINE_SOURCE = "prior winner, test 97.65 not used for selection"
BASELINE_LCNN_WIDTH = 64
BASELINE_N_BINS = 40
BASELINE_N_FFT = 1024
BASELINE_HOP = 512
BASELINE_LR = 0.0020925467114224455
BASELINE_WEIGHT_DECAY = 0.0005310044665511727
BASELINE_DROPOUT = 0.4230032577549638
BASELINE_BATCH_SIZE = 32
BASELINE_MAX_EPOCHS = 10
BASELINE_MEAN_AUC = 0.9894454984360125
BASELINE_WORST_AUC = 0.9492874337005305
BASELINE_STD_AUC = 0.01990202775067736
BASELINE_CI_LOW = 0.9733055674554603
BASELINE_CI_HIGH = 0.9991444189446486
BASELINE_WORST_FOLD_ID = "A06"
_RECORDED_SEED_MEANS = (
    0.9915704814361485,
    0.9846860465116279,
    0.9920799673602612,
)

POOL_MEAN = "mean"
POOL_ASP = "attentive_stat"
POOL_AASIST = "aasist_lite"
LOSS_BCE = "bce"
LOSS_AM = "am_softmax"
LOSS_OC = "oc_softmax"

# Оси карты признаков: батч, канал, частота, время.
BATCH_DIM = 0
CHANNEL_DIM = 1
FREQ_DIM = 2
TIME_DIM = 3
# Оси последовательности: батч, узлы, признаки или батч, время, каналы.
SEQ_NODE_DIM = 1
SEQ_FEAT_DIM = 2
COLLAPSED_CHANNEL_DIM = 1
COLLAPSED_TIME_DIM = 2
LAST_DIM = -1
FEATURE_FLATTEN_START = 1
FLATTEN_ALL = -1
SPEC_RANK_WITHOUT_CHANNEL = 3
WAVE_RANK = 2
LOGIT_DIM = 1
N_CLASSES = 2
CENTER_ROWS = 1
STAT_COUNT = 2
PAIR_WIDTH = 2
GLOBAL_POOL_SIZE = 1
CONV_IN_CHANNELS = 1
CONV_SAME_PAD_DIVISOR = 2
GRAPH_LEAKY_SLOPE = 0.2
VARIANCE_EPS = 1e-6
NORM_EPS = 1e-6
ONE = 1.0
TWO = 2.0
PEAK_LIMIT = 1.0
PAD_LEFT = 0
START_INDEX = 0
QUANT_MIN = 0
REMAINDER_NONE = 0

# Attentive statistics pooling: границы, не выбранный hidden.
ATTENTION_HIDDEN_LOW = 16
ATTENTION_HIDDEN_HIGH = 128
ATTENTION_HIDDEN_STEP = 16

# Графовое внимание AASIST-lite. Границы, не выбранная глубина.
# GRAPH_LEAKY_SLOPE: наклон GAT, архитектурная константа.
GRAPH_LAYERS_LOW = 1
GRAPH_LAYERS_HIGH = 3
GRAPH_HIDDEN_LOW = 32
GRAPH_HIDDEN_HIGH = 128
GRAPH_HIDDEN_STEP = 32

# Отступы и масштабы лосса. Границы, не бумажные значения как подобранные.
AM_SCALE_LOW = 5.0
AM_SCALE_HIGH = 40.0
AM_MARGIN_LOW = 0.05
AM_MARGIN_HIGH = 0.5
OC_SCALE_LOW = 5.0
OC_SCALE_HIGH = 40.0
OC_MARGIN_SPOOF_LOW = 0.0
OC_MARGIN_SPOOF_HIGH = 0.4
OC_MARGIN_GAP_LOW = 0.2
OC_MARGIN_GAP_HIGH = 0.8
OC_MARGIN_BONAFIDE_CAP = 1.0

# Сила RawBoost. Вариант и коэффициенты выбирает Optuna.
RAWBOOST_CONVOLUTIVE = "convolutive"
RAWBOOST_IMPULSIVE = "impulsive"
RAWBOOST_SERIES = "series"
RAWBOOST_VARIANTS = (RAWBOOST_CONVOLUTIVE, RAWBOOST_IMPULSIVE, RAWBOOST_SERIES)
RAWBOOST_STRENGTH_LOW = 0.05
RAWBOOST_STRENGTH_HIGH = 0.9
RAWBOOST_N_TAPS_CHOICES = (3, 5, 7)
RAWBOOST_IMPULSE_RATE_LOW = 1e-3
RAWBOOST_IMPULSE_RATE_HIGH = 5e-2
RAWBOOST_APPLY_P_LOW = 0.3
RAWBOOST_APPLY_P_HIGH = 1.0

# Симуляция телефонного тракта: полоса, 8 кГц, mu-law (G.711).
# Optuna выбирает смесь и вероятность. Только train.
TELEPHONY_SAMPLE_RATE = 8000
TELEPHONE_BAND_LOW_HZ = 300.0
TELEPHONE_BAND_HIGH_HZ = 3400.0
MULAW_MU = 255.0
MULAW_MAX_CODE = 255
CODEC_BANDLIMIT = "bandlimit"
CODEC_MULAW = "mulaw"
CODEC_TELEPHONE = "telephone"
CODEC_CHOICES = (CODEC_BANDLIMIT, CODEC_MULAW, CODEC_TELEPHONE)
CODEC_APPLY_P_LOW = 0.3
CODEC_APPLY_P_HIGH = 1.0
INTERPOLATE_MODE = "linear"
ALIGN_CORNERS = False

# Только проверка форм на CPU, не гиперпараметр обучения.
SMOKE_BATCH = 2
SMOKE_FRAMES = 64

STAGE_NOTEBOOKS = {
    5: "notebooks/05_attentive_stat_pooling.ipynb",
    6: "notebooks/06_aasist_lite_pooling.ipynb",
    7: "notebooks/07_oc_softmax.ipynb",
    8: "notebooks/08_rawboost.ipynb",
    9: "notebooks/09_codec_simulation.ipynb",
}

TRIAL_BUDGET_NOTE = (
    "Each new arm that introduces hyperparameters receives exactly N_OPTUNA_TRIALS. "
    "Arms inside one stage share that budget; a more complex arm does not receive more trials. "
    "The reference arm is the frozen previous winner and is not re-tuned. "
    "Optuna trials use the first protocol seed and every LOSO fold. "
    "The chosen trial is then refit on every seed in SEEDS. "
    "BCE has no new loss knobs, so it is the frozen reference rather than twenty identical trials."
)

FORBIDDEN_DECISION_KEYS = (
    "leaderboard",
    "public_test",
    "test_auc",
    "test_score",
    "test_labels",
)

def baseline_seed_mean_aucs() -> dict[str, float]:
    if len(_RECORDED_SEED_MEANS) != len(SEEDS):
        raise RuntimeError("recorded seed means must align with SEEDS")
    return {str(int(seed)): float(value) for seed, value in zip(SEEDS, _RECORDED_SEED_MEANS, strict=True)}


def baseline_hparams() -> dict:
    """Зафиксированный победитель этапа 2. Следующие этапы копируют его и добавляют искомые ручки."""
    return {
        "pooling": POOL_MEAN,
        "loss": LOSS_BCE,
        "frontend": FRONTEND_LFCC,
        "lcnn_width": BASELINE_LCNN_WIDTH,
        "n_bins": BASELINE_N_BINS,
        "n_fft": BASELINE_N_FFT,
        "hop": BASELINE_HOP,
        "lr": BASELINE_LR,
        "weight_decay": BASELINE_WEIGHT_DECAY,
        "dropout": BASELINE_DROPOUT,
        "batch_size": BASELINE_BATCH_SIZE,
        "max_epochs": BASELINE_MAX_EPOCHS,
        "optimizer": OPTIMIZER_NAME,
        "rawboost": None,
        "codec": None,
    }


def baseline_metrics() -> dict:
    """Сводка LOSO этапа 2. Не скор публичного теста."""
    return {
        "mean_auc": BASELINE_MEAN_AUC,
        "worst_fold": BASELINE_WORST_AUC,
        "worst_auc": BASELINE_WORST_AUC,
        "std": BASELINE_STD_AUC,
        "std_auc": BASELINE_STD_AUC,
        "ci_low": BASELINE_CI_LOW,
        "ci_high": BASELINE_CI_HIGH,
        "worst_fold_id": BASELINE_WORST_FOLD_ID,
        "seed_mean_aucs": baseline_seed_mean_aucs(),
        "source": "outputs/stage2/metrics.json aggregate",
    }


def protocol_block() -> dict:
    return {
        "n_trials": int(N_OPTUNA_TRIALS),
        "seeds": [int(seed) for seed in SEEDS],
        "min_mean_auc_gain": float(MIN_MEAN_AUC_GAIN),
        "strong_mean_auc_gain": float(STRONG_MEAN_AUC_GAIN),
        "folds": FOLDS_NAME,
        "decision_source": DECISION_SOURCE,
        "optuna_seed": int(SEEDS[OPTUNA_SEED_INDEX]),
        "early_stop_patience": int(EARLY_STOP_PATIENCE),
        "final_fit_epochs": int(BASELINE_MAX_EPOCHS),
        "final_fit_early_stopping": False,
        "aggregation": SEED_AGGREGATION,
        "trial_budget_note": TRIAL_BUDGET_NOTE,
    }
