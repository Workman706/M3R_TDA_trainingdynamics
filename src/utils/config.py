"""Configuration dataclasses and YAML loading."""
from __future__ import annotations

import uuid
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union

import yaml

logger = logging.getLogger(__name__)


@dataclass
class TaskConfig:
    name: str = "modular_addition"
    p: int = 97
    train_frac: float = 0.5


@dataclass
class ModelConfig:
    name: str = "modular_transformer"
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 2
    dropout: float = 0.0


@dataclass
class TrainingConfig:
    seed: int = 42
    lr: float = 1e-3
    weight_decay: float = 1.0
    batch_size: int = 512
    max_steps: int = 5000
    log_every: int = 100


@dataclass
class SnapshotConfig:
    save_every: int = 500
    output_dir: str = "runs"
    tensor_keys: List[str] = field(default_factory=lambda: [".*weight"])


@dataclass
class FiltrationConfig:
    name: str = "lower_star"
    use_abs: bool = True
    max_dim: int = 1


@dataclass
class PHConfig:
    backend: str = "gudhi"
    max_homology_dim: int = 1


@dataclass
class FeatureConfig:
    betti_grid_size: int = 100
    betti_grid_range: Union[str, List[float]] = "auto"
    summary_stats: bool = True


@dataclass
class TimeSeriesConfig:
    method: str = "spline"
    smoothing_factor: Optional[float] = None
    spline_degree: int = 3


@dataclass
class TDAConfig:
    filtration: FiltrationConfig = field(default_factory=FiltrationConfig)
    ph: PHConfig = field(default_factory=PHConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    timeseries: TimeSeriesConfig = field(default_factory=TimeSeriesConfig)


@dataclass
class PipelineConfig:
    task: TaskConfig = field(default_factory=TaskConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    snapshots: SnapshotConfig = field(default_factory=SnapshotConfig)
    tda: TDAConfig = field(default_factory=TDAConfig)
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])

    @property
    def run_dir(self) -> Path:
        return Path(self.snapshots.output_dir) / self.run_id

    @staticmethod
    def from_yaml(path: str | Path) -> "PipelineConfig":
        with open(path) as f:
            raw = yaml.safe_load(f)
        return _build_config(raw)


def _build_config(raw: dict) -> PipelineConfig:
    tda_raw = raw.get("tda", {})
    tda = TDAConfig(
        filtration=FiltrationConfig(**tda_raw.get("filtration", {})),
        ph=PHConfig(**tda_raw.get("ph", {})),
        features=FeatureConfig(**tda_raw.get("features", {})),
        timeseries=TimeSeriesConfig(**tda_raw.get("timeseries", {})),
    )
    return PipelineConfig(
        task=TaskConfig(**raw.get("task", {})),
        model=ModelConfig(**raw.get("model", {})),
        training=TrainingConfig(**raw.get("training", {})),
        snapshots=SnapshotConfig(**raw.get("snapshots", {})),
        tda=tda,
    )
