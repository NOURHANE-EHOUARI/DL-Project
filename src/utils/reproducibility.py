"""
reproducibility.py — Seed management and deterministic training utilities.

Owner: Student A
Phase: 2 — Preprocessing & Infrastructure (Week 2–3)
"""

from __future__ import annotations

import logging
import os
import random

import numpy as np
import torch

logger = logging.getLogger(__name__)


def set_seed(seed: int = 42) -> None:
    """Set random seeds across all libraries for reproducible runs.

    Args:
        seed: The random seed value. Default: 42.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # multi-GPU

    # Deterministic CuDNN behaviour (may slow training slightly)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ["PYTHONHASHSEED"] = str(seed)

    logger.info("Global seed set to %d", seed)


def get_device(prefer_gpu: bool = True) -> torch.device:
    """Return the best available device.

    Args:
        prefer_gpu: If True, returns CUDA device when available.

    Returns:
        torch.device — "cuda", "mps", or "cpu"
    """
    if prefer_gpu and torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info("Using GPU: %s", torch.cuda.get_device_name(0))
    elif prefer_gpu and torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using Apple MPS")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU")
    return device