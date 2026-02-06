"""Weight extractor: selects and reshapes tensors from a snapshot for TDA."""
from __future__ import annotations

import re
import logging
from abc import ABC, abstractmethod
from typing import Dict, List

import numpy as np
from safetensors import safe_open

logger = logging.getLogger(__name__)


class WeightExtractor(ABC):
    """Interface for extracting tensors from a snapshot file."""

    @abstractmethod
    def extract(self, path: str) -> Dict[str, np.ndarray]:
        """Return {tensor_name: numpy array in float32}."""
        ...


class RegexWeightExtractor(WeightExtractor):
    """Extracts tensors whose keys match any of the given regex patterns.

    Each matched tensor is returned as a float32 numpy array with its original shape.
    """

    def __init__(self, patterns: List[str]):
        self.patterns = [re.compile(p) for p in patterns]

    def extract(self, path: str) -> Dict[str, np.ndarray]:
        result = {}
        with safe_open(path, framework="numpy") as f:
            for key in f.keys():
                if any(p.fullmatch(key) for p in self.patterns):
                    arr = f.get_tensor(key)
                    result[key] = arr.astype(np.float32)
        if not result:
            avail = []
            with safe_open(path, framework="numpy") as f:
                avail = list(f.keys())
            raise ValueError(
                f"No tensors matched patterns {[p.pattern for p in self.patterns]}. "
                f"Available keys: {avail}"
            )
        logger.info("Extracted %d tensors from %s", len(result), path)
        return result
