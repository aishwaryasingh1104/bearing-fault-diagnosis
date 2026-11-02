"""
src/utils/reproducibility.py
Utilities for setting global random seeds and guaranteeing reproducibility.
"""
from __future__ import annotations

import os
import random

import numpy as np


def set_seed(seed: int = 42) -> None:
    """Set Python, NumPy, and PyTorch (CPU + CUDA) random seeds.

    Parameters
    ----------
    seed:
        Integer seed value.  Use the same value across all training runs
        to obtain reproducible results.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

    # Optional PyTorch seeding – imported lazily to avoid hard dependency
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        # Deterministic mode (may slow training slightly)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def get_device(prefer_gpu: bool = True) -> "torch.device":
    """Return a ``torch.device`` object.

    Automatically selects CUDA when available and *prefer_gpu* is ``True``.

    Parameters
    ----------
    prefer_gpu:
        When ``True`` (default) return a CUDA device if one is detected.
    """
    import torch

    if prefer_gpu and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return device


def device_info() -> dict:
    """Return a dictionary with device information suitable for logging."""
    try:
        import torch

        info: dict = {
            "pytorch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
        }
        if torch.cuda.is_available():
            info["cuda_device"] = torch.cuda.get_device_name(0)
            info["cuda_version"] = torch.version.cuda
        return info
    except ImportError:
        return {"pytorch_version": "not installed", "cuda_available": False}
