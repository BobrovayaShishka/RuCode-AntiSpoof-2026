"""Границы поиска CNN по сырой волне. Кнопки обучения общие с этапом 2."""

from stage2.bounds import (
    BATCH_SIZE_CHOICES,
    DROPOUT_HIGH,
    DROPOUT_LOW,
    EARLY_STOP_PATIENCE,
    EPOCH_HIGH,
    EPOCH_LOW,
    LR_HIGH,
    LR_LOW,
    OOM_BATCH_SIZE_FLOOR,
    WEIGHT_DECAY_HIGH,
    WEIGHT_DECAY_LOW,
)

RAW_WIDTH_CHOICES = (16, 32, 64)
RAW_KERNEL_CHOICES = (256, 512, 1024)
RAW_STRIDE_CHOICES = (128, 256)
RAW_BLOCK_CHOICES = (3, 4, 5)
BLOCK_KERNEL = 3
BLOCK_STRIDE = 2
STUDY_CODE = "s3v1"

RAW_SEARCH_BOUNDS = {
    "width": RAW_WIDTH_CHOICES,
    "kernel": RAW_KERNEL_CHOICES,
    "stride": RAW_STRIDE_CHOICES,
    "n_blocks": RAW_BLOCK_CHOICES,
    "block_kernel": BLOCK_KERNEL,
    "block_stride": BLOCK_STRIDE,
    "lr": (LR_LOW, LR_HIGH),
    "weight_decay": (WEIGHT_DECAY_LOW, WEIGHT_DECAY_HIGH),
    "dropout": (DROPOUT_LOW, DROPOUT_HIGH),
    "batch_size": BATCH_SIZE_CHOICES,
    "max_epochs": (EPOCH_LOW, EPOCH_HIGH),
    "early_stop_patience": EARLY_STOP_PATIENCE,
    "oom_batch_size_floor": OOM_BATCH_SIZE_FLOOR,
}
