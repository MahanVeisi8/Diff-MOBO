import torch
import numpy as np
import os , sys
# Add the project root (two levels up from this script)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))

sys.path.append(project_root)
from Gen_src.diffusion import GaussianDiffusion1D
from Gen_src.model import Unet1D
import torch
import os,sys


# Same architecture as in training
model = Unet1D(
    dim=32,
    dim_mults=(2, 4, 8, 16),
    channels=2,  # X and Y
    dropout=0.1
).cuda()  # or .to(device)

# Create the same diffusion wrapper
diffusion = GaussianDiffusion1D(
    model,
    seq_length=192,      # must match your training setup
    objective='pred_noise',
    timesteps=1000
).cuda()  # or .to(device)

# Load checkpoint
checkpoint_path = rf"/home/bardiya/projects/diffusion_air_manifolding/codes/creative-generativeai-diffusion/model_weigths/model_epoch_375.pt"
model.load_state_dict(torch.load(checkpoint_path ,weights_only=True))
model.eval()
print("Loaded model weights from:", checkpoint_path)
