
### `/src/diffusion_core/README.md`

# diffusion_core

Lightweight 1D diffusion for parametric airfoil shapes.

## Modules
- `datasets.py` — Load/normalize parametric shape vectors (X ∈ ℝ^N)
- `model.py` — `Unet1D` backbone(s)
- `diffusion.py` — Gaussian diffusion process (train/sample)
- `utils.py` — Helpers (EMA, checkpoint IO, seeding, etc.)

## Usage (programmatic)
```python
from diffusion_core.model import Unet1D
from diffusion_core.diffusion import GaussianDiffusion1D

model = Unet1D(dim=64, dim_mults=(1,2,4))
gauss = GaussianDiffusion1D(model, timesteps=1000)
# gauss.train(...) / gauss.sample(...)
```
