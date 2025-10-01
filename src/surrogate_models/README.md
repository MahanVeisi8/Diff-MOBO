# surrogate_models

Lightweight regressors to approximate CFD metrics (e.g., lift/drag) for fast screening.

## Contents
- `MLP/`, `residual/` — model families & training code
- `train_best_surrogate_model.ipynb` — training walkthrough
- `surrogate_model_scores.ipynb` — comparison/metrics
- `Surrogate_Model_weigths/` — weights (recommend moving to `artifacts/models/` with Git LFS)

## Usage
- Train in notebooks, export a lightweight `.pt`/`.pkl`
- Outer loop can switch to `--mode surrogate` to use it for rapid evaluation
