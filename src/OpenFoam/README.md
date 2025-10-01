# OpenFoam (as-is)

Do not rename this directory.

## What's here
- `Airfoil_simulation_1/2/3/` — pipeline variants used in different experiments
- `performance_finding.py` — container-side entry point called by the outer loop
- `surrogate/` — (legacy) quick surrogate utilities
- `Airfoil_full/`, `Airfoil_design/` — supporting scripts/templates

## Container workflow (summary)
1. Load image and start container mounting repo at `/home/airfoil_UANA`
2. Inside container:
   ```bash
   source /opt/openfoam5/etc/bashrc
   pip3 install --upgrade pip
   pip3 install tensorflow==1.15
   pip3 install pymoo