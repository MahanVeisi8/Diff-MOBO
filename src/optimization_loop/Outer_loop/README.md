# optimization_loop

Glue code that:
1. Generates airfoil candidates (diffusion or from a seed set)
2. Evaluates them (surrogate or OpenFOAM-in-Docker)
3. Saves a single bundle with shapes + performances

---

## Main script
`outerloop_creation.py`

---

## CLI (current)

```bash
python src/optimization_loop/outerloop_creation.py \
  --container airfoil_mount \
  --n_samples 64 \
  --out artifacts/run_$(date +%Y%m%d_%H%M%S)/results.npy \
  --mode openfoam      # or surrogate
````

---

## Inputs/Outputs

* Reads/writes under `artifacts/run_YYYYmmdd_HHMMSS/`
* Produces `results.npy` with: `latents`, `shapes`, `performances`

---

## Setup Instructions

If running with **OpenFOAM in Docker**, follow the step-by-step guide here:

👉 [Docker + OpenFOAM Setup Guide](./docker_openfoam_setup_tutorial_full.md)

---

## Notes

* Container name can be passed via `--container` or the `AIRFOIL_CONTAINER` env var.
* Dependencies:

  ```bash
  pip install -r src/optimization_loop/requirements.txt
  ```

```


