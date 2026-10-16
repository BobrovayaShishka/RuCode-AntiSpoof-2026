"""Окружение процесса для весов Hugging Face и аллокатора CUDA.

Импортировать до torch и transformers. CUDA не инициализируется.
"""

from __future__ import annotations

import os

from stage4.constants import HF_CACHE_DIR, HF_HUB_CACHE


def configure_process_env() -> None:
    HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    HF_HUB_CACHE.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(HF_CACHE_DIR)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(HF_HUB_CACHE)
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
