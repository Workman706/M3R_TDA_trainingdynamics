"""Small transformer for modular addition classification."""
from __future__ import annotations

import math
import torch
import torch.nn as nn

from src.models.base import ModelFactory
from src.utils.config import ModelConfig


class ModularTransformer(nn.Module):
    """Minimal transformer: embed two tokens, apply transformer encoder, classify.

    Architecture: Embedding -> positional encoding -> TransformerEncoder -> mean pool -> linear.
    Input: (batch, 2) integer tensor with values in {0, ..., num_classes-1}.
    Output: (batch, num_classes) logits.
    """

    def __init__(self, num_classes: int, d_model: int = 128,
                 n_heads: int = 4, n_layers: int = 2, dropout: float = 0.0):
        super().__init__()
        self.embed = nn.Embedding(num_classes, d_model)
        # Learnable positional encoding for 2 positions
        self.pos_enc = nn.Parameter(torch.randn(1, 2, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=4 * d_model,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 2)
        h = self.embed(x) + self.pos_enc  # (batch, 2, d_model)
        h = self.transformer(h)            # (batch, 2, d_model)
        h = h.mean(dim=1)                  # (batch, d_model)
        return self.head(h)                # (batch, num_classes)


class ModularTransformerFactory(ModelFactory):
    def create(self, config: ModelConfig, num_classes: int) -> nn.Module:
        return ModularTransformer(
            num_classes=num_classes,
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_layers=config.n_layers,
            dropout=config.dropout,
        )
