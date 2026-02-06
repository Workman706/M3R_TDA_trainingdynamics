"""Training loop for classification tasks with periodic snapshots."""
from __future__ import annotations

import logging
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.base import TaskDataset
from src.snapshots.writer import SnapshotWriter

logger = logging.getLogger(__name__)


def train(
    model: nn.Module,
    train_dataset: TaskDataset,
    val_dataset: TaskDataset,
    snapshot_writer: SnapshotWriter,
    *,
    lr: float = 1e-3,
    weight_decay: float = 1.0,
    batch_size: int = 512,
    max_steps: int = 5000,
    save_every: int = 500,
    log_every: int = 100,
    device: str = "cpu",
) -> nn.Module:
    """Train model with AdamW, cross-entropy loss, periodic snapshots.

    Returns the trained model.
    """
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    step = 0
    model.train()
    t0 = time.time()

    while step < max_steps:
        for inputs, labels in loader:
            if step >= max_steps:
                break
            inputs, labels = inputs.to(device), labels.to(device)
            logits = model(inputs)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1

            if step % log_every == 0:
                train_acc = (logits.argmax(-1) == labels).float().mean().item()
                val_loss, val_acc = _evaluate(model, val_loader, criterion, device)
                elapsed = time.time() - t0
                logger.info(
                    "step=%d  train_loss=%.4f  train_acc=%.4f  "
                    "val_loss=%.4f  val_acc=%.4f  elapsed=%.1fs",
                    step, loss.item(), train_acc, val_loss, val_acc, elapsed,
                )

            if step % save_every == 0:
                snapshot_writer.save(
                    model, optimizer, step, loss.item(),
                    extra={"device": device},
                )

    # Final snapshot (only if not already saved at this step)
    if step % save_every != 0:
        snapshot_writer.save(model, optimizer, step, loss.item(), extra={"final": True})
    logger.info("Training complete at step %d", step)
    return model


def _evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            logits = model(inputs)
            total_loss += criterion(logits, labels).item() * len(labels)
            correct += (logits.argmax(-1) == labels).sum().item()
            total += len(labels)
    model.train()
    return total_loss / total, correct / total
