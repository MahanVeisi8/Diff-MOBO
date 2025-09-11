# Generative Airfoil Project

This repository contains code and resources for generating and evaluating airfoil designs using various methods (OpenFoam, Bayesian Optimization, neural surrogates, diffusion models, etc.).

## Repository Structure

```
.
├── src/                  # Source code (Python scripts, shell scripts)
├── notebooks/            # Jupyter notebooks for experiments and demos
├── openfoam/             # OpenFoam scripts, simulation setups, and results
├── diffusion/            # Diffusion-model notebooks and related files
├── docker/               # Docker-related files (ignored by Git)
├── .gitignore            # Ignore patterns (including large Docker tarball)
├── LICENSE               # Project license
└── README.md             # This file
```

## Getting Started

1. **Clone the repository**

   ```bash
   git clone https://github.com/your-username/Generative-Airfoil.git
   cd Generative-Airfoil
   ```

2. **Install dependencies** (e.g., Python packages)

   ```bash
   pip install -r requirements.txt
   ```

3. **Explore code and notebooks**

   * `src/` contains Python and shell scripts for airfoil generation, optimization, and evaluation.
   * `notebooks/` has Jupyter notebooks illustrating workflows.

4. **Run directory tree report** (optional)

   ```bash
   python tree_report.py .
   ```

## Ignored Files

* Large Docker image archive in `docker/` (5 GB) is excluded via `.gitignore`.

## License

Distributed under the MIT License. See `LICENSE` for details.
