# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research project studying neural network training dynamics via persistent homology (TDA). The pipeline trains a transformer on modular addition, saves weight snapshots, then performs topological analysis: lower-star filtration → persistent homology → Betti curves / summary features → spline fitting over training time.

## Repository Structure

- `configs/` — YAML configuration files (single config controls entire pipeline)
- `src/` — All source code, using absolute imports (`from src.x import y`)
  - `data/` — Task datasets (ABC: `TaskDataset`, default: `ModularAdditionDataset`)
  - `models/` — Model factories (ABC: `ModelFactory`, default: `ModularTransformerFactory`)
  - `train/` — Training loop with periodic snapshot saving
  - `snapshots/` — Snapshot writer (`.pt` checkpoint + `.safetensors` + `manifest.jsonl`) and weight extractor
  - `tda/filtrations/` — Filtration builders (ABC: `Filtration`, default: `LowerStarFiltration`)
  - `tda/ph/` — Persistent homology backends (ABC: `PHComputer`, default: `GUDHIPHComputer`)
  - `tda/features/` — Feature extractors (ABC: `FeatureExtractor`, default: `BettiFeatureExtractor`)
  - `tda/timeseries/` — Time-series fitters (ABC: `TimeSeriesFitter`, default: `SplineFitter`)
  - `utils/` — Config dataclasses, seed utilities
- `scripts/` — CLI entry points: `train.py`, `analyze.py`, `sanity_check.py`
- `runs/` — Output directory for training runs (gitignored)
- `requirements.txt` — Python dependencies

## Key Commands

```bash
pip install -r requirements.txt                          # Install deps
python -m scripts.train --config configs/default.yaml    # Train + save snapshots
python -m scripts.analyze --config configs/default.yaml --run-dir runs/<id>  # TDA analysis
python -m scripts.sanity_check                           # Fast end-to-end test (p=7, 50 steps)
```

## Architecture Notes

- Every major component has an ABC in `base.py` and a default implementation; swap by subclassing
- Config is parsed from YAML into nested dataclasses (`PipelineConfig`)
- Snapshots: `.pt` for training resume, `.safetensors` (float32) for TDA, `manifest.jsonl` for metadata
- Lower-star filtration uses grid adjacency (4-neighbor for 2D, chain for 1D)
- Dependencies: PyTorch, GUDHI, safetensors, scipy, numpy, pandas, matplotlib, pyyaml
