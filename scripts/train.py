"""CLI entry point: train a model and save snapshots.

Usage:
    python -m scripts.train --config configs/default.yaml
"""
from __future__ import annotations

import argparse
import logging
import sys

import torch

from src.utils.config import PipelineConfig
from src.utils.seed import set_seed
from src.data.modular_addition import ModularAdditionDataset
from src.models.modular_addition import ModularTransformerFactory
from src.snapshots.writer import DefaultSnapshotWriter
from src.train.trainer import train

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train model with snapshots")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()

    cfg = PipelineConfig.from_yaml(args.config)
    logger.info("Run ID: %s", cfg.run_id)
    logger.info("Run dir: %s", cfg.run_dir)
    cfg.run_dir.mkdir(parents=True, exist_ok=True)

    set_seed(cfg.training.seed)

    # Dataset
    train_ds = ModularAdditionDataset(
        p=cfg.task.p, train_frac=cfg.task.train_frac,
        train=True, seed=cfg.training.seed,
    )
    val_ds = ModularAdditionDataset(
        p=cfg.task.p, train_frac=cfg.task.train_frac,
        train=False, seed=cfg.training.seed,
    )
    logger.info("Dataset: p=%d, train=%d, val=%d", cfg.task.p, len(train_ds), len(val_ds))

    # Model
    factory = ModularTransformerFactory()
    model = factory.create(cfg.model, num_classes=train_ds.num_classes)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("Model: %s (%d params)", cfg.model.name, n_params)

    # Snapshot writer
    writer = DefaultSnapshotWriter(run_dir=cfg.run_dir, seed=cfg.training.seed)

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Device: %s", device)

    # Train
    train(
        model=model,
        train_dataset=train_ds,
        val_dataset=val_ds,
        snapshot_writer=writer,
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
        batch_size=cfg.training.batch_size,
        max_steps=cfg.training.max_steps,
        save_every=cfg.snapshots.save_every,
        log_every=cfg.training.log_every,
        device=device,
    )
    logger.info("Done. Snapshots in: %s", cfg.run_dir / "snapshots")


if __name__ == "__main__":
    main()
