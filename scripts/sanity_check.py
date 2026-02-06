"""Quick sanity check: small p, few steps, full pipeline.

Usage:
    python -m scripts.sanity_check
"""
from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path

import numpy as np
import torch

from src.utils.seed import set_seed
from src.data.modular_addition import ModularAdditionDataset
from src.models.modular_addition import ModularTransformerFactory
from src.utils.config import ModelConfig
from src.snapshots.writer import DefaultSnapshotWriter
from src.snapshots.extractor import RegexWeightExtractor
from src.train.trainer import train
from src.tda.filtrations.lower_star import LowerStarFiltration
from src.tda.ph.gudhi_backend import GUDHIPHComputer
from src.tda.features.betti import BettiFeatureExtractor
from src.tda.timeseries.spline import SplineFitter

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    set_seed(0)
    p = 7  # small prime for fast test
    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = Path(tmpdir)
        logger.info("Sanity check with p=%d in %s", p, run_dir)

        # 1. Dataset
        train_ds = ModularAdditionDataset(p=p, train_frac=0.5, train=True, seed=0)
        val_ds = ModularAdditionDataset(p=p, train_frac=0.5, train=False, seed=0)
        assert len(train_ds) > 0 and len(val_ds) > 0
        logger.info("CHECK: datasets OK (train=%d, val=%d)", len(train_ds), len(val_ds))

        # 2. Model
        cfg = ModelConfig(d_model=32, n_heads=2, n_layers=1)
        model = ModularTransformerFactory().create(cfg, num_classes=p)
        x, y = train_ds[0]
        out = model(x.unsqueeze(0))
        assert out.shape == (1, p)
        logger.info("CHECK: model forward pass OK, output shape=%s", out.shape)

        # 3. Training with snapshots
        writer = DefaultSnapshotWriter(run_dir=run_dir, seed=0)
        train(model, train_ds, val_ds, writer,
              lr=1e-3, weight_decay=0.1, batch_size=16,
              max_steps=50, save_every=10, log_every=10)
        manifest = run_dir / "manifest.jsonl"
        assert manifest.exists()
        entries = [json.loads(l) for l in manifest.read_text().splitlines()]
        assert len(entries) >= 5  # steps 10,20,30,40,50 + final
        logger.info("CHECK: %d snapshots saved", len(entries))

        # 4. Weight extraction
        extractor = RegexWeightExtractor(patterns=[".*weight"])
        tensors = extractor.extract(entries[0]["safetensors"])
        assert len(tensors) > 0
        logger.info("CHECK: extracted %d tensors from snapshot", len(tensors))

        # 5. Lower-star filtration
        filt = LowerStarFiltration(use_abs=True, max_dim=1)
        for name, w in tensors.items():
            fc = filt.build(w)
            assert fc.num_vertices == np.prod(w.shape)
            assert len(fc.simplices) > fc.num_vertices  # should have edges too
            logger.info("CHECK: filtration for %s: %d simplices, %d vertices",
                        name, len(fc.simplices), fc.num_vertices)
            break  # just check one

        # 6. Persistent homology
        ph = GUDHIPHComputer()
        fc = filt.build(list(tensors.values())[0])
        diagrams = ph.compute(fc, max_dim=1)
        assert 0 in diagrams and 1 in diagrams
        logger.info("CHECK: PH computed, H0=%d pairs, H1=%d pairs",
                    len(diagrams[0].pairs), len(diagrams[1].pairs))

        # 7. Feature extraction
        feat_ext = BettiFeatureExtractor(grid_size=50, max_dim=1, summary_stats=True)
        feats = feat_ext.extract(diagrams)
        assert len(feats.feature_vector) == len(feats.feature_names)
        assert len(feats.feature_vector) > 0
        logger.info("CHECK: %d features extracted", len(feats.feature_vector))

        # 8. Spline fitting (need multiple snapshots)
        all_feats = []
        steps = []
        for entry in entries:
            t = extractor.extract(entry["safetensors"])
            w = list(t.values())[0]
            fc = filt.build(w)
            d = ph.compute(fc, max_dim=1)
            f = feat_ext.extract(d)
            all_feats.append(f.feature_vector)
            steps.append(entry["step"])
        features = np.stack(all_feats)
        steps_arr = np.array(steps)

        fitter = SplineFitter(degree=3)
        result = fitter.fit(steps_arr, features)
        assert result.smoothed.shape == features.shape
        logger.info("CHECK: spline fitting OK, smoothed shape=%s", result.smoothed.shape)

        logger.info("ALL SANITY CHECKS PASSED")


if __name__ == "__main__":
    main()
