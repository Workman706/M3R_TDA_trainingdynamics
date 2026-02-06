"""Abstract model factory."""
from abc import ABC, abstractmethod

import torch.nn as nn

from src.utils.config import ModelConfig


class ModelFactory(ABC):
    """Creates a model given config and task metadata."""

    @abstractmethod
    def create(self, config: ModelConfig, num_classes: int) -> nn.Module:
        ...
