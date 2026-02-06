"""Snapshot writer: saves training checkpoints and TDA-friendly weight bundles."""
from __future__ import annotations

import json
import logging
import subprocess
import time
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import torch
from safetensors.torch import save_file as safetensors_save

logger = logging.getLogger(__name__)


def _git_hash() -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return None


class SnapshotWriter(ABC):
    """Interface for writing training snapshots to disk."""

    @abstractmethod
    def save(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
             step: int, loss: float, extra: Dict[str, Any] | None = None) -> Path:
        ...


class DefaultSnapshotWriter(SnapshotWriter):
    """Saves both a full checkpoint (.pt) and a safetensors weight bundle (.safetensors).

    Also appends a line to manifest.jsonl with metadata.
    """

    def __init__(self, run_dir: Path, seed: int):
        self.run_dir = run_dir
        self.seed = seed
        self.snap_dir = run_dir / "snapshots"
        self.snap_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = run_dir / "manifest.jsonl"
        self._start_time = time.time()

    def save(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
             step: int, loss: float, extra: Dict[str, Any] | None = None) -> Path:
        prefix = f"step_{step:07d}"
        ckpt_path = self.snap_dir / f"{prefix}.pt"
        safe_path = self.snap_dir / f"{prefix}.safetensors"

        # Full training checkpoint (for resuming)
        torch.save({
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
        }, ckpt_path)

        # TDA-friendly weight bundle in float32
        weights_f32 = {k: v.detach().cpu().float() for k, v in model.state_dict().items()}
        safetensors_save(weights_f32, str(safe_path))

        # Manifest entry
        entry = {
            "step": step,
            "loss": float(loss),
            "seed": self.seed,
            "wallclock": time.time() - self._start_time,
            "checkpoint": str(ckpt_path),
            "safetensors": str(safe_path),
            "git_hash": _git_hash(),
        }
        if extra:
            entry.update(extra)
        with open(self.manifest_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

        logger.info("Snapshot saved: step=%d, loss=%.4f -> %s", step, loss, safe_path)
        return safe_path
