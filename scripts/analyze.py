"""CLI entry point: offline TDA analysis of saved snapshots.

Usage:
    python -m scripts.analyze --config configs/default.yaml --run-dir runs/<run_id>

Reads manifest.jsonl, loads each snapshot, runs:
  extract weights -> lower-star filtration -> PH -> Betti features -> spline fit
Saves results to results.npz and generates a summary plot.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.utils.config import PipelineConfig
from src.snapshots.extractor import RegexWeightExtractor
from src.tda.filtrations.lower_star import LowerStarFiltration
from src.tda.ph.gudhi_backend import GUDHIPHComputer
from src.tda.features.betti import BettiFeatureExtractor
from src.tda.timeseries.spline import SplineFitter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="TDA analysis of training snapshots")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--run-dir", type=str, required=True,
                        help="Path to run directory containing manifest.jsonl")
    args = parser.parse_args()

    cfg = PipelineConfig.from_yaml(args.config)
    run_dir = Path(args.run_dir)
    manifest_path = run_dir / "manifest.jsonl"
    if not manifest_path.exists():
        raise FileNotFoundError(f"No manifest.jsonl found in {run_dir}")

    # Load manifest
    entries = []
    with open(manifest_path) as f:
        for line in f:
            entries.append(json.loads(line.strip()))
    logger.info("Found %d snapshots in manifest", len(entries))

    # Build pipeline components
    extractor = RegexWeightExtractor(patterns=cfg.snapshots.tensor_keys)
    filtration = LowerStarFiltration(
        use_abs=cfg.tda.filtration.use_abs,
        max_dim=cfg.tda.filtration.max_dim,
    )
    ph_computer = GUDHIPHComputer()
    feature_extractor = BettiFeatureExtractor(
        grid_size=cfg.tda.features.betti_grid_size,
        grid_range=cfg.tda.features.betti_grid_range,
        max_dim=cfg.tda.ph.max_homology_dim,
        summary_stats=cfg.tda.features.summary_stats,
    )
    fitter = SplineFitter(
        smoothing_factor=cfg.tda.timeseries.smoothing_factor,
        degree=cfg.tda.timeseries.spline_degree,
    )

    # Process each snapshot
    steps_list = []
    losses_list = []
    all_features = []
    feature_names = None

    for entry in entries:
        step = entry["step"]
        loss = entry["loss"]
        safe_path = entry["safetensors"]
        logger.info("Processing step %d ...", step)

        # Extract weight tensors
        tensors = extractor.extract(safe_path)

        # For each tensor, compute TDA features and concatenate
        snapshot_features = []
        for tensor_name, weights in sorted(tensors.items()):
            # Build filtered complex
            fc = filtration.build(weights)
            # Compute persistence
            diagrams = ph_computer.compute(fc, max_dim=cfg.tda.ph.max_homology_dim)
            # Extract features
            feats = feature_extractor.extract(diagrams)
            snapshot_features.append(feats.feature_vector)
            if feature_names is None:
                # Prefix feature names with tensor name
                feature_names = [
                    f"{tensor_name}/{n}" for n in feats.feature_names
                ]
            elif len(all_features) == 0:
                # First snapshot: build full feature names across all tensors
                feature_names.extend(
                    [f"{tensor_name}/{n}" for n in feats.feature_names]
                )

        combined = np.concatenate(snapshot_features)
        all_features.append(combined)
        steps_list.append(step)
        losses_list.append(loss)

    steps = np.array(steps_list)
    features = np.stack(all_features)  # (n_snapshots, n_features)
    logger.info("Feature matrix: %s", features.shape)

    # Spline fitting
    fit_result = fitter.fit(steps, features)
    logger.info("Spline fitting complete")

    # Save results
    out_path = run_dir / "results.npz"
    np.savez(
        out_path,
        steps=steps,
        losses=np.array(losses_list),
        features=features,
        smoothed=fit_result.smoothed,
        feature_names=np.array(feature_names if feature_names else []),
    )
    # Also save spline coefficients as JSON
    coeff_path = run_dir / "spline_coefficients.json"
    with open(coeff_path, "w") as f:
        json.dump(fit_result.coefficients, f, indent=2)
    logger.info("Results saved to %s", out_path)

    # Generate summary plot
    _plot_summary(run_dir, steps, features, fit_result.smoothed, losses_list, feature_names)
    logger.info("Analysis complete for %s", run_dir)


def _plot_summary(run_dir: Path, steps, features, smoothed, losses, feature_names):
    """Generate a summary plot with loss curve and a few selected feature trajectories."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Loss curve
    axes[0, 0].plot(steps, losses, "o-", markersize=3)
    axes[0, 0].set_title("Training Loss")
    axes[0, 0].set_xlabel("Step")
    axes[0, 0].set_ylabel("Loss")

    # Pick a few interesting feature dimensions to plot
    n_features = features.shape[1]
    # Plot summary stats if available (look for "total_pers" features)
    pers_indices = [i for i, n in enumerate(feature_names or []) if "total_pers" in n]
    betti_indices = [i for i, n in enumerate(feature_names or []) if "betti_0_t50" in n]

    # Plot 1: total persistence over time
    ax = axes[0, 1]
    if pers_indices:
        for idx in pers_indices[:4]:
            name = feature_names[idx] if feature_names else str(idx)
            short = name.split("/")[-1] if "/" in name else name
            ax.plot(steps, features[:, idx], "o", markersize=3, alpha=0.5)
            ax.plot(steps, smoothed[:, idx], "-", label=short)
        ax.legend(fontsize=7)
    ax.set_title("Total Persistence over Time")
    ax.set_xlabel("Step")

    # Plot 2: a sample Betti curve value over time
    ax = axes[1, 0]
    sample_indices = list(range(min(5, n_features)))
    for idx in sample_indices:
        name = feature_names[idx] if feature_names else str(idx)
        short = name.split("/")[-1] if "/" in name else name
        ax.plot(steps, features[:, idx], "o", markersize=2, alpha=0.4)
        ax.plot(steps, smoothed[:, idx], "-", label=short, linewidth=1)
    ax.legend(fontsize=6)
    ax.set_title("Sample Betti Curve Values")
    ax.set_xlabel("Step")

    # Plot 3: Betti curve heatmap (first tensor, H0)
    ax = axes[1, 1]
    # Find the block of betti_0 features for the first tensor
    h0_indices = [i for i, n in enumerate(feature_names or []) if "betti_0_t" in n]
    if h0_indices and features.shape[0] > 1:
        block = features[:, h0_indices[: min(100, len(h0_indices))]]
        im = ax.imshow(block.T, aspect="auto", origin="lower",
                       extent=[steps[0], steps[-1], 0, block.shape[1]])
        ax.set_title("Betti-0 Curve Heatmap (first tensor)")
        ax.set_xlabel("Step")
        ax.set_ylabel("Grid index")
        fig.colorbar(im, ax=ax)
    else:
        ax.set_title("(Not enough data for heatmap)")

    fig.tight_layout()
    fig.savefig(run_dir / "summary_plot.png", dpi=150)
    plt.close(fig)
    logger.info("Plot saved to %s", run_dir / "summary_plot.png")


if __name__ == "__main__":
    main()
