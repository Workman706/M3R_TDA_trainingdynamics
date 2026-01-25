# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a research project studying neural network training dynamics from the perspective of persistent homology (Topological Data Analysis). The project is in early development stages.

## Repository Structure

- `model/`: Neural network model implementations
  - `model1.py`: Primary model file (currently empty, intended for neural network implementation)

## Development Context

### Current State
- The repository is on the `toymodel` branch (main branch exists for PRs)
- This is an early-stage research codebase with minimal structure
- No dependency management files (requirements.txt, setup.py, etc.) are currently present

### Expected Development Direction
Given the project's focus on training dynamics and persistent homology, future development will likely involve:
- Neural network implementations (likely PyTorch or TensorFlow)
- Persistent homology computation libraries (e.g., GUDHI, Ripser)
- Training dynamic analysis code
- Visualization tools for topological features
- Experiment tracking and data analysis notebooks

### Recommendations for Future Development
When adding code to this repository:
- Add a `requirements.txt` or `pyproject.toml` to track dependencies when packages are introduced
- Consider organizing code into modules: `models/`, `topology/`, `experiments/`, `utils/`
- Add Jupyter notebooks to `notebooks/` or `experiments/` for exploratory analysis
- Document mathematical concepts and TDA methodology as they're implemented
