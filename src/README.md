# src

Source code for the Generative Airfoil project.

---

## Modules

- **diffusion_core/**  
  Core diffusion model implementation (datasets, UNet1D, diffusion loop, utilities).

- **OpenFoam/**  
  Scripts and simulation setups for coupling with OpenFOAM (kept as-is).

- **optimization_loop/**  
  Outer loop orchestrator. Generates airfoil candidates → evaluates them (surrogate or OpenFOAM) → saves results.  
  See [optimization_loop/README.md](./optimization_loop/README.md) for details.

- **surrogate_models/**  
  Surrogate models for fast performance approximation (MLP, residual nets, etc.).

---

## Notes

- Large outputs, weights, and results should go in `artifacts/` (tracked with Git LFS, not in this folder).
- For Docker/OpenFOAM setup, see the [setup guide](optimization_loop/docker_openfoam_setup_tutorial_full.md).
- Main entry point:  
  ```bash
  python src/optimization_loop/outerloop_creation.py
