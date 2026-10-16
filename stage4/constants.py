"""Чекпоинты и границы поиска этапа 4.

Константы протокола в af_common. Здесь границы поиска и лимиты памяти, не рабочая точка.
"""

from __future__ import annotations

from af_common import PROJECT_ROOT, STAGE4_DIR, STAGE4_METRICS_PATH, SUBMISSION_COLUMNS

CACHE_ROOT = STAGE4_DIR / "cache"
HF_CACHE_DIR = PROJECT_ROOT / "checkpoints" / "hf"
HF_HUB_CACHE = HF_CACHE_DIR / "hub"

LAYER_PROBE_PATH = STAGE4_DIR / "layer_probe.json"
PROBE_STATE_PATH = STAGE4_DIR / "probe_state.json"
HEAD_OPTUNA_PATH = STAGE4_DIR / "head_optuna.json"
HEAD_RESULT_PATH = STAGE4_DIR / "head_result.json"
METRICS_PATH = STAGE4_METRICS_PATH

# Базовые чекпоинты. XLS-R 1B/2B и WavLM large не влезают в 4 ГБ.
WAVLM_CHECKPOINT = "microsoft/wavlm-base"
XLSR_CHECKPOINT = "facebook/wav2vec2-xls-r-300m"
MODEL_CHECKPOINTS = (WAVLM_CHECKPOINT, XLSR_CHECKPOINT)

# Сколько максимально длинных реплик класть в один forward.
# Уменьшается, если батч не влезает в VRAM.
WAVLM_BATCH_MULTIPLIER = 4
XLSR_BATCH_MULTIPLIER = 2
MODEL_BATCH_MULTIPLIERS = {
    WAVLM_CHECKPOINT: WAVLM_BATCH_MULTIPLIER,
    XLSR_CHECKPOINT: XLSR_BATCH_MULTIPLIER,
}

# Максимальная длина клипа. Максимум train меньше 14 с, первый кандидат не режет.
# Более короткие: запасной вариант при нехватке VRAM.
AUDIO_MAX_SECONDS_CANDIDATES = (14.0, 10.0, 8.0, 6.0, 4.0)
MIN_AUDIO_SECONDS = 0.25
EXTRACT_LOOKAHEAD = 16

ALLOW_TF32 = True
GPU_DTYPE_NAME = "float16"

# Solver LogReg для pooled-признаков 768/1024. saga не сходился и был сильно медленнее.
PROBE_LOGREG_SOLVER = "lbfgs"
PROBE_REFERENCE = "last_layer"

SCALE_POOLED_FEATURES = True

HEAD_LR_LOW = 1e-4
HEAD_LR_HIGH = 1e-1
HEAD_WEIGHT_DECAY_LOW = 1e-6
HEAD_WEIGHT_DECAY_HIGH = 1e-2
HEAD_EPOCHS_LOW = 5
HEAD_EPOCHS_HIGH = 30
HEAD_BATCH_SIZES = (64, 128, 256, 512)
HEAD_POS_WEIGHT_LOW = 0.05
HEAD_POS_WEIGHT_HIGH = 8.0
HEAD_OPTIMIZER = "adamw"

VERDICT_POLL_SECONDS = 15.0
NEURAL_METRICS_WAIT_SECONDS = 180.0
NEURAL_METRICS_POLL_SECONDS = 10.0

LAYER_FILE = "layer_{index:02d}.f32"
ACTIVE_FILE = "active.json"
META_FILE = "meta.json"
TEST_CACHE_DIR = CACHE_ROOT / "test"