"""
src/training/trainer.py
Generic training and evaluation loop for all model variants.

Features
--------
* Class-weighted loss for imbalanced datasets.
* Learning-rate scheduling (cosine annealing or step).
* Early stopping on validation loss.
* Checkpoint saving (best and final).
* Progress callback (callable) for Streamlit live updates.
* Experiment result logging to JSON.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .losses import cross_entropy_loss
from .checkpointing import save_checkpoint
from ..evaluation.metrics import compute_metrics

logger = logging.getLogger("bfd.trainer")


# ---------------------------------------------------------------------------
# Experiment result container
# ---------------------------------------------------------------------------

class ExperimentResult:
    """Stores the complete history and final metrics of one training run."""

    def __init__(self, experiment_name: str, config: Dict):
        self.experiment_name = experiment_name
        self.config = config
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        self.val_metrics: List[Dict] = []
        self.best_epoch: int = 0
        self.best_val_loss: float = float("inf")
        self.best_metrics: Dict = {}
        self.training_time_s: float = 0.0
        self.checkpoint_path: Optional[str] = None
        self.warnings: List[str] = []
        self.status: str = "pending"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "experiment_name": self.experiment_name,
            "status": self.status,
            "best_epoch": self.best_epoch,
            "best_val_loss": self.best_val_loss,
            "best_metrics": self.best_metrics,
            "training_time_s": round(self.training_time_s, 2),
            "checkpoint_path": self.checkpoint_path,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "warnings": self.warnings,
            "config": self.config,
        }

    def save(self, output_dir: str | Path) -> Path:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"{self.experiment_name.replace(' ', '_')}_result.json"
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)
        return path


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    """Model trainer supporting classification and transfer learning.

    Parameters
    ----------
    model:
        PyTorch nn.Module.
    device:
        ``torch.device`` (cpu or cuda).
    config:
        Training hyper-parameter dict (from ``configs/default.yaml``).
    experiment_name:
        Human-readable name for this run.
    label_map:
        Label-to-integer mapping (for embedding in checkpoints).
    progress_callback:
        Optional callable ``(epoch, total_epochs, train_loss, val_loss)``
        called at the end of each epoch (used by Streamlit).
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        config: Dict,
        experiment_name: str = "experiment",
        label_map: Optional[Dict[str, int]] = None,
        progress_callback: Optional[Callable] = None,
    ):
        self.model = model.to(device)
        self.device = device
        self.config = config
        self.experiment_name = experiment_name
        self.label_map = label_map or {}
        self.progress_callback = progress_callback

        train_cfg = config.get("training", config)

        # Optimizer
        lr = float(train_cfg.get("learning_rate", 1e-3))
        wd = float(train_cfg.get("weight_decay", 1e-4))
        opt_name = train_cfg.get("optimizer", "adam").lower()
        if opt_name == "adamw":
            self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        elif opt_name == "sgd":
            self.optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd, momentum=0.9)
        else:
            self.optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

        # Scheduler
        sched_name = train_cfg.get("scheduler", "cosine").lower()
        T_max = int(train_cfg.get("scheduler_T_max", train_cfg.get("epochs", 30)))
        if sched_name == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=T_max)
        elif sched_name == "step":
            # BUG-08: step_size and gamma are now configurable (no more magic numbers)
            step_size = int(train_cfg.get("lr_step_size", 10))
            gamma     = float(train_cfg.get("lr_step_gamma", 0.5))
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=step_size, gamma=gamma
            )
        else:
            self.scheduler = None

        self.epochs = int(train_cfg.get("epochs", 30))
        self.batch_size = int(train_cfg.get("batch_size", 64))
        self.patience = int(train_cfg.get("early_stopping_patience", 10))
        self.checkpoint_dir = train_cfg.get("checkpoint_dir", "artifacts/checkpoints")
        self.class_weighted = train_cfg.get("class_weighted_loss", True)
        # BUG-09: gradient-clip norm is now configurable
        self.grad_clip_norm = float(train_cfg.get("grad_clip_norm", 5.0))

        self.result = ExperimentResult(experiment_name, config)

    # ------------------------------------------------------------------
    # Data loaders
    # ------------------------------------------------------------------

    def _make_loader(
        self, windows: np.ndarray, labels: np.ndarray, shuffle: bool
    ) -> DataLoader:
        X = torch.tensor(windows[:, np.newaxis, :], dtype=torch.float32)
        y = torch.tensor(labels, dtype=torch.long)
        ds = TensorDataset(X, y)
        return DataLoader(ds, batch_size=self.batch_size, shuffle=shuffle,
                          num_workers=0, pin_memory=self.device.type == "cuda")

    def _make_multichannel_loader(
        self, windows: np.ndarray, labels: np.ndarray, shuffle: bool
    ) -> DataLoader:
        """For EWT-pre-processed windows of shape (N, n_components, L)."""
        X = torch.tensor(windows, dtype=torch.float32)
        y = torch.tensor(labels, dtype=torch.long)
        ds = TensorDataset(X, y)
        return DataLoader(ds, batch_size=self.batch_size, shuffle=shuffle,
                          num_workers=0, pin_memory=self.device.type == "cuda")

    # ------------------------------------------------------------------
    # Core training loop
    # ------------------------------------------------------------------

    def train(
        self,
        train_windows: np.ndarray,
        train_labels: np.ndarray,
        val_windows: np.ndarray,
        val_labels: np.ndarray,
        num_classes: int,
        multichannel: bool = False,
    ) -> ExperimentResult:
        """Run the full training loop.

        Parameters
        ----------
        train_windows:
            ``(N_train, L)`` or ``(N_train, C, L)`` if *multichannel*.
        train_labels:
            ``(N_train,)`` integer labels.
        val_windows, val_labels:
            Validation data.
        num_classes:
            Total number of classes (needed for loss weight computation).
        multichannel:
            When ``True``, expects ``(N, C, L)`` inputs (EWT pre-processed).
        """
        loader_fn = self._make_multichannel_loader if multichannel else self._make_loader

        train_loader = loader_fn(train_windows, train_labels, shuffle=True)
        val_loader   = loader_fn(val_windows,   val_labels,   shuffle=False)

        # Loss function
        criterion = cross_entropy_loss(
            class_weighted=self.class_weighted,
            labels=train_labels,
            num_classes=num_classes,
            device=self.device,
        )

        best_val_loss = float("inf")
        patience_counter = 0
        t0 = time.time()
        self.result.status = "running"

        for epoch in range(1, self.epochs + 1):
            # --- Train ---
            self.model.train()
            train_loss = self._run_epoch(train_loader, criterion, training=True)

            # --- Validate ---
            self.model.eval()
            val_loss, val_preds, val_true = self._run_eval(val_loader, criterion)

            # --- Metrics ---
            metrics = compute_metrics(val_true, val_preds, num_classes=num_classes)
            metrics["val_loss"] = round(val_loss, 6)
            metrics["train_loss"] = round(train_loss, 6)
            metrics["epoch"] = epoch

            self.result.train_losses.append(round(train_loss, 6))
            self.result.val_losses.append(round(val_loss, 6))
            self.result.val_metrics.append(metrics)

            # Scheduler step
            if self.scheduler:
                self.scheduler.step()

            # Progress callback
            if self.progress_callback:
                self.progress_callback(epoch, self.epochs, train_loss, val_loss, metrics)

            logger.info(
                "Epoch %d/%d | train_loss=%.4f | val_loss=%.4f | val_acc=%.4f",
                epoch, self.epochs, train_loss, val_loss, metrics.get("accuracy", 0.0),
            )

            # --- Early stopping & checkpoint ---
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self.result.best_epoch = epoch
                self.result.best_val_loss = val_loss
                self.result.best_metrics = metrics

                ckpt_path = save_checkpoint(
                    model=self.model,
                    optimizer=self.optimizer,
                    epoch=epoch,
                    metrics=metrics,
                    config=self.config,
                    checkpoint_dir=self.checkpoint_dir,
                    label_map=self.label_map,
                    tag="best",
                )
                self.result.checkpoint_path = str(ckpt_path)
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    logger.info("Early stopping triggered at epoch %d.", epoch)
                    self.result.warnings.append(f"Early stopping at epoch {epoch}.")
                    break

        self.result.training_time_s = time.time() - t0
        self.result.status = "completed"
        logger.info(
            "Training complete. Best epoch: %d, best val acc: %.4f",
            self.result.best_epoch,
            self.result.best_metrics.get("accuracy", 0.0),
        )
        return self.result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _run_epoch(self, loader: DataLoader, criterion: nn.Module, training: bool) -> float:
        total_loss = 0.0
        n_batches = 0
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(self.device, non_blocking=True)
            y_batch = y_batch.to(self.device, non_blocking=True)
            if training:
                self.optimizer.zero_grad()
            logits = self.model(X_batch)
            loss = criterion(logits, y_batch)
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip_norm)
                self.optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        return total_loss / max(n_batches, 1)

    def _run_eval(
        self, loader: DataLoader, criterion: nn.Module
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        total_loss = 0.0
        all_preds: List[np.ndarray] = []
        all_true:  List[np.ndarray] = []

        with torch.no_grad():
            for X_batch, y_batch in loader:
                X_batch = X_batch.to(self.device, non_blocking=True)
                y_batch = y_batch.to(self.device, non_blocking=True)
                logits = self.model(X_batch)
                loss = criterion(logits, y_batch)
                total_loss += loss.item()
                preds = logits.argmax(dim=1).cpu().numpy()
                all_preds.append(preds)
                all_true.append(y_batch.cpu().numpy())

        preds_arr = np.concatenate(all_preds) if all_preds else np.array([])
        true_arr  = np.concatenate(all_true)  if all_true  else np.array([])
        n_batches = len(loader)
        return total_loss / max(n_batches, 1), preds_arr, true_arr
