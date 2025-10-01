# Notebooks

Structured demos and experiments. Keep notebooks small and task-focused.

## Layout
- `00_overview/` — End-to-end walkthroughs
- `10_generation/` — Training & sampling diffusion models
- `20_evaluation/` — Metrics, ablations, plots
- `30_openfoam/` — CFD coupling demos (when needed)
- `90_archive/` — Older/alternate experiments (kept for reference)

## Cloud notebooks
- **Colab (Main Flow):**  
  GenerativeAirfoil(Main_Flow).ipynb  
  https://colab.research.google.com/drive/1q9PG53IRJTT7E0KyeFvqGBovLTQtffT9?usp=sharing

- **Kaggle (Diffusion / DPP experiments):**
  - Mahan: https://www.kaggle.com/code/mahanveisi/generativeairfoil
  - Bardia: https://www.kaggle.com/code/bardiyakariminia/generative-dpp-airfoil

> If a Kaggle notebook isn’t accessible to you, **please contact Mahan** to grant access.

## Tips
- Keep heavy training artifacts out of Git (use `artifacts/` + Git LFS).
- Clear runtime state before commit (no large outputs in cells).
