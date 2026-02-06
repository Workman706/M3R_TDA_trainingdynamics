"""Modular addition dataset: classify (a, b) -> (a+b) mod p."""
from __future__ import annotations

import numpy as np
import torch

from src.data.base import TaskDataset


class ModularAdditionDataset(TaskDataset):
    """Dataset of (a, b) -> (a + b) mod p for a, b in {0, ..., p-1}.

    Args:
        p: prime modulus
        train_frac: fraction of all p*p pairs used for training
        train: if True return training split, else validation
        seed: random seed for reproducible split
    """

    def __init__(self, p: int, train_frac: float = 0.5,
                 train: bool = True, seed: int = 42):
        self.p = p
        rng = np.random.RandomState(seed)
        # All (a, b) pairs
        pairs = np.array([(a, b) for a in range(p) for b in range(p)])
        labels = (pairs[:, 0] + pairs[:, 1]) % p
        n = len(pairs)
        idx = rng.permutation(n)
        split = int(n * train_frac)
        sel = idx[:split] if train else idx[split:]
        self.inputs = torch.tensor(pairs[sel], dtype=torch.long)
        self.labels = torch.tensor(labels[sel], dtype=torch.long)

    @property
    def num_classes(self) -> int:
        return self.p

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple:
        return self.inputs[idx], self.labels[idx]
