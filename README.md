# TDA Training Dynamics

Studying neural network training dynamics through the lens of persistent homology (Topological Data Analysis).

This pipeline trains a small transformer on modular addition, periodically saves weight snapshots, then performs TDA analysis: lower-star filtration → persistent homology → Betti curve / summary features → spline fitting over training time.

## Install

```bash
pip install -r requirements.txt
```

Requires Python 3.9+.

## Quick Start

### 1. Train a model

```bash
python -m scripts.train --config configs/default.yaml
```

This trains a small transformer on `(a+b) mod 97`, saving snapshots every 500 steps into `runs/<run_id>/snapshots/`. The run ID is printed at startup.

### 2. Run TDA analysis

```bash
python -m scripts.analyze --config configs/default.yaml --run-dir runs/<run_id>
```

This loads each snapshot, extracts weight tensors, builds lower-star filtrations, computes persistent homology (H0 and H1), extracts Betti curves and persistence statistics, fits smoothing splines over training time, and saves:

- `results.npz` — steps, losses, feature matrix, smoothed features, feature names
- `spline_coefficients.json` — spline knots and coefficients
- `summary_plot.png` — loss curve, persistence trajectories, Betti heatmap

### 3. Sanity check

```bash
python -m scripts.sanity_check
```

Runs a fast end-to-end test with `p=7` and 50 training steps.

## Project Structure

```
configs/
  default.yaml          # All pipeline parameters
src/
  data/
    base.py             # TaskDataset ABC
    modular_addition.py # (a+b) mod p dataset
  models/
    base.py             # ModelFactory ABC
    modular_addition.py # Small transformer
  train/
    trainer.py          # Training loop with snapshots
  snapshots/
    writer.py           # Checkpoint + safetensors + manifest.jsonl
    extractor.py        # Regex-based tensor selection from safetensors
  tda/
    filtrations/
      base.py           # Filtration ABC + FilteredComplex dataclass
      lower_star.py     # Lower-star filtration on grid graphs
    ph/
      base.py           # PHComputer ABC + PersistenceDiagram dataclass
      gudhi_backend.py  # GUDHI SimplexTree backend
    features/
      base.py           # FeatureExtractor ABC + TDAFeatures dataclass
      betti.py          # Betti curves + persistence summary stats
    timeseries/
      base.py           # TimeSeriesFitter ABC + FitResult dataclass
      spline.py         # Scipy UnivariateSpline fitting
  utils/
    config.py           # YAML config → dataclasses
    seed.py             # Deterministic seeding
scripts/
  train.py              # CLI: train model
  analyze.py            # CLI: TDA analysis
  sanity_check.py       # CLI: fast end-to-end test
```

## Configuration

All parameters are in `configs/default.yaml`. Key sections:

- **task**: dataset choice, `p`, train fraction
- **model**: architecture, dimensions, layers
- **training**: seed, lr, weight decay, batch size, max steps
- **snapshots**: save frequency, output dir, tensor key patterns
- **tda.filtration**: lower-star settings (use_abs, max simplex dim)
- **tda.ph**: backend, max homology dimension
- **tda.features**: Betti grid size/range, summary stats toggle
- **tda.timeseries**: spline degree, smoothing factor

## Extending

Each component is defined by an ABC. To swap in a new implementation:

| Component | ABC | Default |
|-----------|-----|---------|
| Dataset | `TaskDataset` | `ModularAdditionDataset` |
| Model | `ModelFactory` | `ModularTransformerFactory` |
| Snapshot writer | `SnapshotWriter` | `DefaultSnapshotWriter` |
| Weight extractor | `WeightExtractor` | `RegexWeightExtractor` |
| Filtration | `Filtration` | `LowerStarFiltration` |
| PH backend | `PHComputer` | `GUDHIPHComputer` |
| Features | `FeatureExtractor` | `BettiFeatureExtractor` |
| Time series | `TimeSeriesFitter` | `SplineFitter` |
