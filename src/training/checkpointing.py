"""
src/training/checkpointing.py
Model checkpoint saving and loading utilities.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import torch

logger = logging.getLogger("bfd.checkpointing")


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    epoch: int,
    metrics: Dict[str, Any],
    config: Dict[str, Any],
    checkpoint_dir: str | Path,
    label_map: Optional[Dict[str, int]] = None,
    tag: str = "best",
) -> Path:
    """Save a model checkpoint.

    Parameters
    ----------
    model:
        PyTorch model.
    optimizer:
        Optimizer (saved for resumable training).
    epoch:
        Current training epoch.
    metrics:
        Dict of validation metrics at this epoch.
    config:
        Experiment configuration (for reproducibility).
    checkpoint_dir:
        Directory to save the checkpoint file.
    label_map:
        Label-to-index mapping (embedded in the checkpoint for
        compatibility checks on load).
    tag:
        Short descriptor appended to the filename (e.g. ``'best'``,
        ``'final'``).

    Returns
    -------
    Path
        Path of the saved checkpoint file.
    """
    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"checkpoint_{tag}_epoch{epoch:03d}_{timestamp}.pt"
    path = ckpt_dir / filename

    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
        "metrics": metrics,
        "config": config,
        "label_map": label_map or {},
        "model_class": model.__class__.__name__,
        "timestamp": timestamp,
    }
    torch.save(payload, path)
    logger.info("Checkpoint saved to %s (epoch %d)", path, epoch)
    return path


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: Optional[torch.device] = None,
    strict: bool = True,
) -> Dict[str, Any]:
    """Load a checkpoint and restore model (and optionally optimizer) state.

    Parameters
    ----------
    path:
        Path to the ``.pt`` checkpoint file.
    model:
        Model whose ``state_dict`` will be restored.
    optimizer:
        Optional optimizer to restore.
    device:
        Map-location device.  When ``None``, uses the device the model is on.
    strict:
        Passed to ``model.load_state_dict()``.  Set to ``False`` for
        partial loading (e.g. backbone-only).

    Returns
    -------
    dict
        The full checkpoint payload (metrics, config, label_map, …).

    Raises
    ------
    FileNotFoundError
        When the checkpoint file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    map_location = device or next(model.parameters()).device
    payload = torch.load(path, map_location=map_location, weights_only=False)

    # Compatibility check
    saved_class = payload.get("model_class", "")
    current_class = model.__class__.__name__
    if saved_class and saved_class != current_class:
        logger.warning(
            "Model class mismatch: checkpoint has '%s', current model is '%s'.",
            saved_class, current_class,
        )

    model.load_state_dict(payload["model_state_dict"], strict=strict)
    if optimizer and payload.get("optimizer_state_dict"):
        optimizer.load_state_dict(payload["optimizer_state_dict"])

    logger.info(
        "Loaded checkpoint from %s (epoch %d, metrics=%s)",
        path, payload.get("epoch", -1), payload.get("metrics", {}),
    )
    return payload


def list_checkpoints(checkpoint_dir: str | Path) -> list:
    """List all checkpoint files in a directory, newest first."""
    ckpt_dir = Path(checkpoint_dir)
    if not ckpt_dir.exists():
        return []
    return sorted(ckpt_dir.glob("checkpoint_*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
